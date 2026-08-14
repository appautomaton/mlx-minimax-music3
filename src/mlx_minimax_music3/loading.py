"""Strict, shard-aware loading for MLX model components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .checkpoint import TensorInfo, inspect_safetensors, load_json_object
from .config import (
    ConditionEncoderConfig,
    FlowTransformerConfig,
    Qwen3Config,
    RVQDepthDecoderConfig,
    VocoderConfig,
)
from .manifest import CheckpointManifest
from .models.condition_encoder import ConditionEncoder
from .models.flow_transformer import FlowTransformer
from .models.qwen3 import Qwen3ForCausalLM
from .models.rvq_depth import RVQDepthDecoder
from .models.vocoder import Vocoder
from .quantization import apply_q8_topology


class CheckpointLayoutError(ValueError):
    """Raised when stored tensors do not exactly match a model component."""


@dataclass(frozen=True, slots=True)
class StoredTensor:
    file: Path
    info: TensorInfo


@dataclass(frozen=True, slots=True)
class ComponentLayout:
    """Validated files and tensor locations for one component directory."""

    directory: Path
    files: tuple[Path, ...]
    tensors: dict[str, StoredTensor]


def _checkpoint_model[ModuleT: nn.Module](
    checkpoint: str | Path,
    component: str,
    model: ModuleT,
) -> tuple[ModuleT, CheckpointManifest]:
    manifest = CheckpointManifest.read(Path(checkpoint) / "manifest.json")
    if manifest.profile == "q8":
        apply_q8_topology(component, model, manifest)
    return model, manifest


def _safe_component_path(directory: Path, filename: str) -> Path:
    if not filename or Path(filename).is_absolute():
        raise CheckpointLayoutError(f"Invalid shard path in index: {filename!r}")
    candidate = (directory / filename).resolve()
    if candidate.parent != directory:
        raise CheckpointLayoutError(
            f"Shard path must stay directly inside {directory}: {filename!r}"
        )
    if candidate.suffix != ".safetensors":
        raise CheckpointLayoutError(f"Indexed shard is not SafeTensors: {filename!r}")
    if not candidate.is_file():
        raise CheckpointLayoutError(f"Indexed shard does not exist: {candidate}")
    return candidate


def discover_component_layout(directory: str | Path) -> ComponentLayout:
    """Discover and cross-check one sharded or single-file component."""

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise CheckpointLayoutError(f"Component directory does not exist: {directory}")

    index_paths = sorted(directory.glob("*.safetensors.index.json"))
    if len(index_paths) > 1:
        raise CheckpointLayoutError(
            f"Multiple SafeTensors indexes found under {directory}"
        )

    indexed_locations: dict[str, Path] | None = None
    if index_paths:
        index = load_json_object(index_paths[0])
        raw_weight_map = index.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise CheckpointLayoutError(
                f"SafeTensors index has no non-empty weight_map: {index_paths[0]}"
            )
        indexed_locations = {}
        for name, filename in raw_weight_map.items():
            if not isinstance(name, str) or not isinstance(filename, str):
                raise CheckpointLayoutError(
                    f"weight_map must map tensor names to shard names: {index_paths[0]}"
                )
            indexed_locations[name] = _safe_component_path(directory, filename)
        files = tuple(sorted(set(indexed_locations.values())))
    else:
        files = tuple(sorted(directory.glob("*.safetensors")))
        if not files:
            raise CheckpointLayoutError(
                f"No SafeTensors checkpoint found under {directory}"
            )

    unindexed_files = set(directory.glob("*.safetensors")) - set(files)
    if unindexed_files:
        examples = ", ".join(path.name for path in sorted(unindexed_files)[:3])
        raise CheckpointLayoutError(
            f"Unindexed SafeTensors files under {directory}: {examples}"
        )

    tensors: dict[str, StoredTensor] = {}
    for path in files:
        file_info = inspect_safetensors(path)
        for tensor in file_info.tensors:
            if tensor.name in tensors:
                previous = tensors[tensor.name].file
                raise CheckpointLayoutError(
                    f"Duplicate tensor {tensor.name!r} in {previous.name} and {path.name}"
                )
            tensors[tensor.name] = StoredTensor(file=path, info=tensor)

    if indexed_locations is not None:
        stored_names = set(tensors)
        indexed_names = set(indexed_locations)
        missing = indexed_names - stored_names
        unexpected = stored_names - indexed_names
        misplaced = {
            name
            for name in stored_names & indexed_names
            if tensors[name].file != indexed_locations[name]
        }
        if missing or unexpected or misplaced:
            raise CheckpointLayoutError(
                "SafeTensors index does not match shard headers: "
                f"{len(missing)} missing, {len(unexpected)} unexpected, "
                f"{len(misplaced)} misplaced"
            )

    return ComponentLayout(
        directory=directory,
        files=files,
        tensors=tensors,
    )


def validate_model_layout(
    model: nn.Module,
    layout: ComponentLayout,
    *,
    allowed_dtypes: frozenset[str] | None = None,
) -> None:
    """Validate all names, shapes, and optional source dtypes before allocation."""

    parameters = dict(tree_flatten(model.parameters()))
    wanted = set(parameters)
    stored = set(layout.tensors)
    missing = wanted - stored
    unexpected = stored - wanted
    if missing or unexpected:
        raise CheckpointLayoutError(
            "Checkpoint does not match the model tree: "
            f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {sorted(unexpected)[:3]})"
        )

    shape_mismatches = []
    dtype_mismatches = []
    for name, parameter in parameters.items():
        tensor = layout.tensors[name].info
        if tuple(parameter.shape) != tensor.shape:
            shape_mismatches.append(
                f"{name}: model={tuple(parameter.shape)}, checkpoint={tensor.shape}"
            )
        if allowed_dtypes is not None and tensor.dtype not in allowed_dtypes:
            dtype_mismatches.append(f"{name}: {tensor.dtype}")
    if shape_mismatches:
        raise CheckpointLayoutError(
            "Checkpoint tensor shapes do not match the model: "
            + "; ".join(shape_mismatches[:3])
        )
    if dtype_mismatches:
        raise CheckpointLayoutError(
            f"Checkpoint contains unsupported dtypes {sorted(allowed_dtypes)}: "
            + "; ".join(dtype_mismatches[:3])
        )


def load_component_weights(
    model: nn.Module,
    directory: str | Path,
    *,
    allowed_dtypes: frozenset[str] | None = None,
    materialize: bool = True,
) -> nn.Module:
    """Strictly load a component one shard at a time.

    Header validation happens before any payload is read. Each shard replaces
    only its corresponding model leaves, and the completed model is optionally
    materialized before return so first-token latency cannot hide page faults.
    """

    layout = discover_component_layout(directory)
    validate_model_layout(model, layout, allowed_dtypes=allowed_dtypes)
    if not materialize:
        return model

    for path in layout.files:
        weights = mx.load(str(path))
        expected_names = {
            name for name, stored in layout.tensors.items() if stored.file == path
        }
        if set(weights) != expected_names:
            raise CheckpointLayoutError(
                f"MLX loaded a different tensor set than the validated header: {path}"
            )
        ordered = [(name, weights[name]) for name in sorted(weights)]
        model.load_weights(ordered, strict=False)
        mx.eval([value for _, value in ordered])
        del ordered, weights

    mx.eval(model.parameters())
    return model


def load_language_model(
    checkpoint: str | Path, *, materialize: bool = True
) -> Qwen3ForCausalLM:
    """Instantiate and load the official dense Qwen3 component."""

    directory = Path(checkpoint) / "language_model"
    config = Qwen3Config.from_file(directory / "config.json")
    model, manifest = _checkpoint_model(
        checkpoint,
        "language_model",
        Qwen3ForCausalLM(config),
    )
    return load_component_weights(
        model,
        directory,
        allowed_dtypes=(
            frozenset({"BF16", "U32"})
            if manifest.profile == "q8"
            else frozenset({"BF16"})
        ),
        materialize=materialize,
    )


def load_rvq_depth_decoder(
    checkpoint: str | Path, *, materialize: bool = True
) -> RVQDepthDecoder:
    """Instantiate and load the official dense RVQ depth decoder component."""

    directory = Path(checkpoint) / "rvq_depth_decoder"
    config = RVQDepthDecoderConfig.from_file(directory / "config.json")
    model, manifest = _checkpoint_model(
        checkpoint,
        "rvq_depth_decoder",
        RVQDepthDecoder(config),
    )
    return load_component_weights(
        model,
        directory,
        allowed_dtypes=(
            frozenset({"BF16", "U32"})
            if manifest.profile == "q8"
            else frozenset({"BF16"})
        ),
        materialize=materialize,
    )


def load_condition_encoder(
    checkpoint: str | Path, *, materialize: bool = True
) -> ConditionEncoder:
    """Instantiate and load an MLX-native dense condition encoder."""

    directory = Path(checkpoint) / "condition_encoder"
    config = ConditionEncoderConfig.from_file(directory / "config.json")
    model, _ = _checkpoint_model(
        checkpoint,
        "condition_encoder",
        ConditionEncoder(config),
    )
    return load_component_weights(
        model,
        directory,
        allowed_dtypes=frozenset({"F32"}),
        materialize=materialize,
    )


def load_flow_transformer(
    checkpoint: str | Path, *, materialize: bool = True
) -> FlowTransformer:
    """Instantiate and load an MLX-native dense flow transformer."""

    directory = Path(checkpoint) / "transformer"
    config = FlowTransformerConfig.from_file(directory / "config.json")
    model, _ = _checkpoint_model(
        checkpoint,
        "transformer",
        FlowTransformer(config),
    )
    return load_component_weights(
        model,
        directory,
        allowed_dtypes=frozenset({"F32"}),
        materialize=materialize,
    )


def load_vocoder(
    checkpoint: str | Path, *, materialize: bool = True
) -> Vocoder:
    """Instantiate and load an MLX-native dense waveform decoder."""

    directory = Path(checkpoint) / "vocoder"
    config = VocoderConfig.from_file(directory / "config.json")
    model, _ = _checkpoint_model(
        checkpoint,
        "vocoder",
        Vocoder(config),
    )
    return load_component_weights(
        model,
        directory,
        allowed_dtypes=frozenset({"F32"}),
        materialize=materialize,
    )
