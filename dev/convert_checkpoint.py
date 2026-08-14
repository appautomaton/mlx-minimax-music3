"""Convert the official component checkpoint to MLX-native dense layouts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mlx_minimax_music3.checkpoint import TensorInfo, inspect_safetensors
from mlx_minimax_music3.loading import discover_component_layout
from mlx_minimax_music3.manifest import (
    CheckpointManifest,
    ComponentManifest,
    ManifestFile,
    sha256_file,
)

SOURCE_REPOSITORY = "MiniMaxAI/MiniMax-Music3"
SOURCE_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
WEIGHT_COMPONENTS = (
    "language_model",
    "rvq_depth_decoder",
    "condition_encoder",
    "transformer",
    "vocoder",
)
UNCHANGED_COMPONENTS = frozenset({"language_model", "rvq_depth_decoder"})


@dataclass(frozen=True, slots=True)
class PlannedTensor:
    source_name: str
    output_name: str
    source_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: str
    operation: str


def _transpose_conv_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) != 3:
        raise ValueError(f"Expected rank-3 convolution weight, got {shape}")
    return shape[0], shape[2], shape[1]


def _transpose_conv_transpose_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) != 3:
        raise ValueError(f"Expected rank-3 transposed-conv weight, got {shape}")
    return shape[1], shape[2], shape[0]


def plan_tensor(component: str, tensor: TensorInfo) -> PlannedTensor | None:
    """Describe a lossless dense mapping without loading the tensor payload."""

    name = tensor.name
    shape = tensor.shape
    if component in UNCHANGED_COMPONENTS:
        operation = "identity"
        output_name = name
        output_shape = shape
    elif component == "condition_encoder" and name == "proj.weight" or component == "transformer" and name in {
        "preprocess_conv.weight",
        "postprocess_conv.weight",
    }:
        operation = "conv1d"
        output_name = name
        output_shape = _transpose_conv_shape(shape)
    elif component == "vocoder" and name.endswith(".weight_g"):
        return None
    elif component == "vocoder" and name.endswith(".weight_v"):
        output_name = name.removesuffix("_v")
        if ".conv_t1." in name:
            operation = "weight_norm_conv_transpose1d"
            output_shape = _transpose_conv_transpose_shape(shape)
        else:
            operation = "weight_norm_conv1d"
            output_shape = _transpose_conv_shape(shape)
    elif component == "vocoder" and name == "dec_in_proj.weight":
        operation = "conv1d"
        output_name = name
        output_shape = _transpose_conv_shape(shape)
    else:
        operation = "identity"
        output_name = name
        output_shape = shape
    return PlannedTensor(
        source_name=name,
        output_name=output_name,
        source_shape=shape,
        output_shape=output_shape,
        dtype=tensor.dtype,
        operation=operation,
    )


def plan_component(source: str | Path, component: str) -> tuple[PlannedTensor, ...]:
    layout = discover_component_layout(Path(source) / component)
    planned = []
    for stored in layout.tensors.values():
        mapping = plan_tensor(component, stored.info)
        if mapping is not None:
            planned.append(mapping)
    output_names = [mapping.output_name for mapping in planned]
    if len(output_names) != len(set(output_names)):
        raise ValueError(f"Dense mapping creates duplicate tensors in {component}")
    if component == "vocoder":
        source_names = set(layout.tensors)
        for mapping in planned:
            if mapping.operation.startswith("weight_norm_"):
                weight_g = mapping.source_name.removesuffix("weight_v") + "weight_g"
                if weight_g not in source_names:
                    raise ValueError(
                        f"Missing weight-normalization scale for {mapping.source_name}"
                    )
    return tuple(sorted(planned, key=lambda mapping: mapping.output_name))


def plan_checkpoint(source: str | Path) -> dict[str, tuple[PlannedTensor, ...]]:
    """Validate every dense mapping using headers only."""

    return {
        component: plan_component(source, component)
        for component in WEIGHT_COMPONENTS
    }


def _normalized_weight(weight_v: mx.array, weight_g: mx.array) -> mx.array:
    norm = mx.sqrt(mx.sum(mx.square(weight_v), axis=(1, 2), keepdims=True))
    return weight_v * (weight_g / norm)


def _convert_weights(
    component: str,
    source_weights: dict[str, mx.array],
    mappings: tuple[PlannedTensor, ...],
) -> dict[str, mx.array]:
    converted = {}
    for mapping in mappings:
        value = source_weights[mapping.source_name]
        if mapping.operation == "conv1d":
            value = mx.transpose(value, (0, 2, 1))
        elif mapping.operation.startswith("weight_norm_"):
            weight_g_name = (
                mapping.source_name.removesuffix("weight_v") + "weight_g"
            )
            value = _normalized_weight(value, source_weights[weight_g_name])
            axes = (
                (1, 2, 0)
                if mapping.operation == "weight_norm_conv_transpose1d"
                else (0, 2, 1)
            )
            value = mx.transpose(value, axes)
        converted[mapping.output_name] = value
    return converted


def _atomic_link_or_copy(source: Path, destination: Path) -> None:
    if destination.is_file() and os.path.samefile(source, destination):
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    try:
        temporary.replace(destination)
    finally:
        # A rename may be a no-op when both paths already link to one inode.
        temporary.unlink(missing_ok=True)


def _converted_file_matches(
    destination: Path,
    source_file: Path,
    mappings: tuple[PlannedTensor, ...],
) -> bool:
    if not destination.is_file():
        return False
    try:
        info = inspect_safetensors(destination)
    except (OSError, ValueError):
        return False
    if info.metadata.get("format") != "mlx-minimax-music3":
        return False
    if info.metadata.get("mapping_version") != "1":
        return False
    if info.metadata.get("source") != source_file.name:
        return False
    actual = {
        tensor.name: (tensor.shape, tensor.dtype) for tensor in info.tensors
    }
    expected = {
        mapping.output_name: (mapping.output_shape, mapping.dtype)
        for mapping in mappings
    }
    return actual == expected


def _convert_component(
    source_root: Path,
    destination_root: Path,
    component: str,
    mappings: tuple[PlannedTensor, ...],
) -> list[tuple[Path, Path]]:
    source_layout = discover_component_layout(source_root / component)
    destination_directory = destination_root / component
    destination_directory.mkdir(parents=True, exist_ok=True)
    output_pairs = []
    for source_file in source_layout.files:
        destination = destination_directory / source_file.name
        if component in UNCHANGED_COMPONENTS:
            print(f"{component}: link {source_file.name}", flush=True)
            _atomic_link_or_copy(source_file, destination)
        else:
            names = {
                name
                for name, stored in source_layout.tensors.items()
                if stored.file == source_file
            }
            shard_mappings = tuple(
                mapping for mapping in mappings if mapping.source_name in names
            )
            if _converted_file_matches(
                destination, source_file, shard_mappings
            ):
                print(f"{component}: reuse {source_file.name}", flush=True)
            else:
                print(f"{component}: convert {source_file.name}", flush=True)
                source_weights = mx.load(str(source_file))
                converted = _convert_weights(
                    component, source_weights, shard_mappings
                )
                mx.eval(converted)
                temporary = destination.with_name(
                    destination.stem + ".tmp.safetensors"
                )
                temporary.unlink(missing_ok=True)
                mx.save_safetensors(
                    str(temporary),
                    converted,
                    metadata={
                        "format": "mlx-minimax-music3",
                        "mapping_version": "1",
                        "source": source_file.name,
                    },
                )
                temporary.replace(destination)
                del converted, source_weights
                mx.clear_cache()
        output_pairs.append((source_file, destination))

    for index in (source_root / component).glob("*.safetensors.index.json"):
        shutil.copy2(index, destination_directory / index.name)
        output_pairs.append((index, destination_directory / index.name))
    return output_pairs


def _copy_metadata(source_root: Path, destination_root: Path) -> list[tuple[Path, Path]]:
    copied = []
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or ".cache" in source.parts:
            continue
        if source.suffix == ".safetensors" or source.name.endswith(
            ".safetensors.index.json"
        ):
            continue
        if source.name in {"inventory.json", "manifest.json"}:
            continue
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append((source, destination))
    return copied


def _manifest_file(
    source_root: Path,
    destination_root: Path,
    source: Path,
    destination: Path,
) -> ManifestFile:
    tensor_count = 0
    dtypes: tuple[str, ...] = ()
    if destination.suffix == ".safetensors":
        info = inspect_safetensors(destination)
        tensor_count = len(info.tensors)
        dtypes = tuple(sorted({tensor.dtype for tensor in info.tensors}))
    source_digest = sha256_file(source)
    destination_digest = (
        source_digest if os.path.samefile(source, destination) else sha256_file(destination)
    )
    return ManifestFile(
        path=destination.relative_to(destination_root).as_posix(),
        size=destination.stat().st_size,
        sha256=destination_digest,
        tensor_count=tensor_count,
        dtypes=dtypes,
        source_path=source.relative_to(source_root).as_posix(),
        source_sha256=source_digest,
    )


def convert_checkpoint(source: Path, destination: Path) -> CheckpointManifest:
    source = source.resolve()
    destination = destination.resolve()
    if (
        source == destination
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Source and destination checkpoints must not be nested")
    destination.mkdir(parents=True, exist_ok=True)
    plans = plan_checkpoint(source)
    copied = _copy_metadata(source, destination)
    for component, mappings in plans.items():
        copied.extend(
            _convert_component(source, destination, component, mappings)
        )

    grouped: dict[str, list[ManifestFile]] = defaultdict(list)
    for source_file, destination_file in copied:
        relative = destination_file.relative_to(destination)
        component = relative.parts[0] if len(relative.parts) > 1 else "root"
        grouped[component].append(
            _manifest_file(
                source,
                destination,
                source_file,
                destination_file,
            )
        )
    manifest = CheckpointManifest(
        profile="dense",
        source_repository=SOURCE_REPOSITORY,
        source_revision=SOURCE_REVISION,
        components=tuple(
            ComponentManifest(
                name=name,
                files=tuple(sorted(files, key=lambda file: file.path)),
            )
            for name, files in sorted(grouped.items())
        ),
    )
    manifest.write(destination / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Validate and print tensor mappings without reading payloads.",
    )
    args = parser.parse_args()
    plans = plan_checkpoint(args.source)
    if args.plan:
        summary = {
            component: {
                "source_tensors": len(
                    discover_component_layout(args.source / component).tensors
                ),
                "output_tensors": len(mappings),
                "operations": dict(
                    sorted(
                        {
                            operation: sum(
                                mapping.operation == operation
                                for mapping in mappings
                            )
                            for operation in {
                                mapping.operation for mapping in mappings
                            }
                        }.items()
                    )
                ),
            }
            for component, mappings in plans.items()
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.destination is None:
        parser.error("destination is required unless --plan is used")
    manifest = convert_checkpoint(args.source, args.destination)
    print(args.destination / "manifest.json")
    print(
        f"{sum(len(component.files) for component in manifest.components)} files"
    )


if __name__ == "__main__":
    main()
