"""Create a selective affine-q8 checkpoint from validated MLX-dense weights."""

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
from mlx_minimax_music3.quantization import (
    is_q8_module,
    qualified_module_name,
)

QUANTIZED_COMPONENTS = frozenset({"language_model", "rvq_depth_decoder"})
QUANTIZATION_MODE = "affine"
QUANTIZATION_BITS = 8
QUANTIZATION_GROUP_SIZE = 64


@dataclass(frozen=True, slots=True)
class PlannedOutput:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class TensorQuantizationPlan:
    source_name: str
    module_name: str | None
    outputs: tuple[PlannedOutput, ...]

    @property
    def quantized(self) -> bool:
        return self.module_name is not None


def plan_tensor(component: str, tensor: TensorInfo) -> TensorQuantizationPlan:
    """Plan one source tensor without reading its payload."""

    module_path = tensor.name.removesuffix(".weight")
    if tensor.name.endswith(".weight") and is_q8_module(component, module_path):
        if tensor.dtype != "BF16":
            raise ValueError(
                f"q8 allowlisted tensor must be BF16: {component}.{tensor.name}"
            )
        if len(tensor.shape) != 2:
            raise ValueError(
                f"q8 allowlisted tensor must be rank 2: {component}.{tensor.name}"
            )
        output_dims, input_dims = tensor.shape
        if input_dims % QUANTIZATION_GROUP_SIZE:
            raise ValueError(
                f"q8 input dimension must be divisible by "
                f"{QUANTIZATION_GROUP_SIZE}: {component}.{tensor.name}"
            )
        packed_dims = input_dims * QUANTIZATION_BITS // 32
        group_dims = input_dims // QUANTIZATION_GROUP_SIZE
        return TensorQuantizationPlan(
            source_name=tensor.name,
            module_name=qualified_module_name(component, module_path),
            outputs=(
                PlannedOutput(tensor.name, (output_dims, packed_dims), "U32"),
                PlannedOutput(
                    f"{module_path}.scales",
                    (output_dims, group_dims),
                    tensor.dtype,
                ),
                PlannedOutput(
                    f"{module_path}.biases",
                    (output_dims, group_dims),
                    tensor.dtype,
                ),
            ),
        )
    return TensorQuantizationPlan(
        source_name=tensor.name,
        module_name=None,
        outputs=(PlannedOutput(tensor.name, tensor.shape, tensor.dtype),),
    )


def plan_component(
    source: str | Path,
    component: str,
) -> tuple[TensorQuantizationPlan, ...]:
    layout = discover_component_layout(Path(source) / component)
    plans = tuple(
        plan_tensor(component, stored.info)
        for _, stored in sorted(layout.tensors.items())
    )
    output_names = [output.name for plan in plans for output in plan.outputs]
    if len(output_names) != len(set(output_names)):
        raise ValueError(f"q8 mapping creates duplicate tensors in {component}")
    if not any(plan.quantized for plan in plans):
        raise ValueError(f"q8 policy selected no tensors in {component}")
    return plans


def _quantize_weights(
    source_weights: dict[str, mx.array],
    plans: tuple[TensorQuantizationPlan, ...],
) -> dict[str, mx.array]:
    converted = {}
    for plan in plans:
        value = source_weights[plan.source_name]
        if not plan.quantized:
            converted[plan.source_name] = value
            continue
        weight, scales, biases = mx.quantize(
            value,
            group_size=QUANTIZATION_GROUP_SIZE,
            bits=QUANTIZATION_BITS,
            mode=QUANTIZATION_MODE,
        )
        converted[plan.outputs[0].name] = weight
        converted[plan.outputs[1].name] = scales
        converted[plan.outputs[2].name] = biases
    return converted


def _atomic_link_or_copy(source: Path, destination: Path) -> None:
    if destination.is_file() and os.path.samefile(source, destination):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _expected_shard(
    plans: tuple[TensorQuantizationPlan, ...],
) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        output.name: (output.shape, output.dtype)
        for plan in plans
        for output in plan.outputs
    }


def _converted_shard_matches(
    destination: Path,
    source: Path,
    plans: tuple[TensorQuantizationPlan, ...],
) -> bool:
    if not destination.is_file():
        return False
    try:
        info = inspect_safetensors(destination)
    except (OSError, ValueError):
        return False
    expected_metadata = {
        "format": "mlx-minimax-music3",
        "profile": "q8",
        "quantization_mode": QUANTIZATION_MODE,
        "quantization_bits": str(QUANTIZATION_BITS),
        "quantization_group_size": str(QUANTIZATION_GROUP_SIZE),
        "source": source.name,
    }
    actual = {
        tensor.name: (tensor.shape, tensor.dtype) for tensor in info.tensors
    }
    return info.metadata == expected_metadata and actual == _expected_shard(plans)


def _write_quantized_shard(
    source: Path,
    destination: Path,
    plans: tuple[TensorQuantizationPlan, ...],
) -> None:
    if _converted_shard_matches(destination, source, plans):
        print(f"q8: reuse {source.parent.name}/{source.name}", flush=True)
        return
    print(f"q8: convert {source.parent.name}/{source.name}", flush=True)
    source_weights = mx.load(str(source))
    converted = _quantize_weights(source_weights, plans)
    mx.eval(converted)
    temporary = destination.with_name(destination.stem + ".tmp.safetensors")
    temporary.unlink(missing_ok=True)
    mx.save_safetensors(
        str(temporary),
        converted,
        metadata={
            "format": "mlx-minimax-music3",
            "profile": "q8",
            "quantization_mode": QUANTIZATION_MODE,
            "quantization_bits": str(QUANTIZATION_BITS),
            "quantization_group_size": str(QUANTIZATION_GROUP_SIZE),
            "source": source.name,
        },
    )
    temporary.replace(destination)
    del converted, source_weights
    mx.synchronize()
    mx.clear_cache()


def _write_index(
    destination: Path,
    weight_map: dict[str, str],
    total_size: int,
) -> None:
    data = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _record_for_output(
    destination_root: Path,
    destination: Path,
    source_record: ManifestFile,
) -> ManifestFile:
    tensor_count = 0
    dtypes: tuple[str, ...] = ()
    if destination.suffix == ".safetensors":
        info = inspect_safetensors(destination)
        tensor_count = len(info.tensors)
        dtypes = tuple(sorted({tensor.dtype for tensor in info.tensors}))
    return ManifestFile(
        path=destination.relative_to(destination_root).as_posix(),
        size=destination.stat().st_size,
        sha256=sha256_file(destination),
        tensor_count=tensor_count,
        dtypes=dtypes,
        source_path=source_record.source_path,
        source_sha256=source_record.source_sha256,
    )


def quantize_checkpoint(
    source: Path,
    destination: Path,
    *,
    verify_source_digests: bool = False,
) -> CheckpointManifest:
    """Convert one managed dense checkpoint into a resumable q8 profile."""

    source = source.resolve()
    destination = destination.resolve()
    if (
        source == destination
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Source and destination checkpoints must not be nested")
    dense_manifest = CheckpointManifest.read(source / "manifest.json")
    if dense_manifest.profile != "dense":
        raise ValueError("q8 conversion requires an MLX-dense source checkpoint")
    dense_manifest.verify(source, digests=verify_source_digests)
    destination.mkdir(parents=True, exist_ok=True)

    source_records = {
        record.path: record
        for component in dense_manifest.components
        for record in component.files
    }
    output_records: dict[str, ManifestFile] = {}
    quantized_modules = set()

    for path, source_record in sorted(source_records.items()):
        component = Path(path).parts[0] if len(Path(path).parts) > 1 else "root"
        if component in QUANTIZED_COMPONENTS and (
            path.endswith((".safetensors", ".safetensors.index.json"))
        ):
            continue
        source_file = source / path
        destination_file = destination / path
        _atomic_link_or_copy(source_file, destination_file)
        output_records[path] = _record_for_output(
            destination,
            destination_file,
            source_record,
        )

    for component in sorted(QUANTIZED_COMPONENTS):
        layout = discover_component_layout(source / component)
        component_plans = plan_component(source, component)
        plans_by_name = {plan.source_name: plan for plan in component_plans}
        weight_map = {}
        total_size = 0
        for source_file in layout.files:
            shard_names = {
                name
                for name, stored in layout.tensors.items()
                if stored.file == source_file
            }
            shard_plans = tuple(
                plans_by_name[name] for name in sorted(shard_names)
            )
            destination_file = destination / component / source_file.name
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            _write_quantized_shard(source_file, destination_file, shard_plans)
            relative = destination_file.relative_to(destination).as_posix()
            output_records[relative] = _record_for_output(
                destination,
                destination_file,
                source_records[relative],
            )
            info = inspect_safetensors(destination_file)
            total_size += sum(tensor.byte_size for tensor in info.tensors)
            for plan in shard_plans:
                if plan.module_name is not None:
                    quantized_modules.add(plan.module_name)
                for output in plan.outputs:
                    weight_map[output.name] = source_file.name

        index_records = [
            record
            for path, record in source_records.items()
            if path.startswith(f"{component}/")
            and path.endswith(".safetensors.index.json")
        ]
        if len(index_records) > 1:
            raise ValueError(f"Multiple dense indexes found for {component}")
        if index_records:
            source_record = index_records[0]
            destination_index = destination / source_record.path
            _write_index(destination_index, weight_map, total_size)
            output_records[source_record.path] = _record_for_output(
                destination,
                destination_index,
                source_record,
            )

    grouped: dict[str, list[ManifestFile]] = defaultdict(list)
    for record in output_records.values():
        parts = Path(record.path).parts
        component = parts[0] if len(parts) > 1 else "root"
        grouped[component].append(record)
    manifest = CheckpointManifest(
        profile="q8",
        source_repository=dense_manifest.source_repository,
        source_revision=dense_manifest.source_revision,
        components=tuple(
            ComponentManifest(
                name=name,
                files=tuple(sorted(records, key=lambda record: record.path)),
            )
            for name, records in sorted(grouped.items())
        ),
        quantized_modules=tuple(sorted(quantized_modules)),
        quantization_mode=QUANTIZATION_MODE,
        quantization_bits=QUANTIZATION_BITS,
        quantization_group_size=QUANTIZATION_GROUP_SIZE,
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
        help="Validate and print the q8 tensor policy without reading payloads.",
    )
    parser.add_argument(
        "--verify-source-digests",
        action="store_true",
        help="Hash every dense source file before conversion.",
    )
    args = parser.parse_args()
    plans = {
        component: plan_component(args.source, component)
        for component in sorted(QUANTIZED_COMPONENTS)
    }
    if args.plan:
        summary = {
            component: {
                "source_tensors": len(component_plans),
                "output_tensors": sum(
                    len(plan.outputs) for plan in component_plans
                ),
                "quantized_modules": sum(
                    plan.quantized for plan in component_plans
                ),
            }
            for component, component_plans in plans.items()
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.destination is None:
        parser.error("destination is required unless --plan is used")
    manifest = quantize_checkpoint(
        args.source,
        args.destination,
        verify_source_digests=args.verify_source_digests,
    )
    print(args.destination / "manifest.json")
    print(
        f"{len(manifest.quantized_modules)} modules across "
        f"{sum(len(component.files) for component in manifest.components)} files"
    )


if __name__ == "__main__":
    main()
