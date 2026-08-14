"""Verify an MLX-dense checkpoint against its official source using only MLX."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx

from dev.convert_checkpoint import PlannedTensor, plan_checkpoint
from mlx_minimax_music3.loading import discover_component_layout
from mlx_minimax_music3.manifest import CheckpointManifest, sha256_file


class DenseVerificationError(ValueError):
    """Raised when a dense checkpoint is not equivalent to its source."""


@dataclass(frozen=True, slots=True)
class DenseVerificationSummary:
    source_files: int
    dense_files: int
    identity_tensors: int
    transposed_tensors: int
    folded_weight_norm_tensors: int
    hardlinked_files: int
    digests_verified: bool

    @property
    def tensors_verified(self) -> int:
        return (
            self.identity_tensors
            + self.transposed_tensors
            + self.folded_weight_norm_tensors
        )


def _expected_tensor(
    mapping: PlannedTensor,
    source_weights: dict[str, mx.array],
) -> mx.array:
    value = source_weights[mapping.source_name]
    if mapping.operation == "identity":
        return value
    if mapping.operation == "conv1d":
        return mx.transpose(value, (0, 2, 1))
    if mapping.operation.startswith("weight_norm_"):
        weight_g_name = (
            mapping.source_name.removesuffix("weight_v") + "weight_g"
        )
        try:
            weight_g = source_weights[weight_g_name]
        except KeyError as error:
            raise DenseVerificationError(
                f"Missing weight-normalization scale: {weight_g_name}"
            ) from error
        norm = mx.sqrt(mx.sum(mx.square(value), axis=(1, 2), keepdims=True))
        value = value * (weight_g / norm)
        axes = (
            (1, 2, 0)
            if mapping.operation == "weight_norm_conv_transpose1d"
            else (0, 2, 1)
        )
        return mx.transpose(value, axes)
    raise DenseVerificationError(
        f"Unsupported dense mapping operation: {mapping.operation}"
    )


def _verify_source_digests(
    source: Path,
    manifest: CheckpointManifest,
) -> None:
    records = {
        (record.source_path, record.source_sha256)
        for component in manifest.components
        for record in component.files
    }
    for source_path, expected_digest in sorted(records):
        if source_path is None or expected_digest is None:
            raise DenseVerificationError(
                "Dense manifest is missing source provenance"
            )
        path = (source / source_path).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise DenseVerificationError(f"Missing source file: {source_path}")
        if sha256_file(path) != expected_digest:
            raise DenseVerificationError(
                f"Source SHA-256 mismatch: {source_path}"
            )


def _validate_component_names(
    component: str,
    source_names: set[str],
    dense_names: set[str],
    mappings: tuple[PlannedTensor, ...],
) -> None:
    expected_dense = {mapping.output_name for mapping in mappings}
    if dense_names != expected_dense:
        raise DenseVerificationError(
            f"Dense tensor set mismatch for {component}: "
            f"missing={sorted(expected_dense - dense_names)[:3]}, "
            f"unexpected={sorted(dense_names - expected_dense)[:3]}"
        )
    consumed_source = {mapping.source_name for mapping in mappings}
    omitted_source = source_names - consumed_source
    expected_omitted = (
        {
            mapping.source_name.removesuffix("weight_v") + "weight_g"
            for mapping in mappings
            if mapping.operation.startswith("weight_norm_")
        }
        if component == "vocoder"
        else set()
    )
    if omitted_source != expected_omitted:
        raise DenseVerificationError(
            f"Source tensor set mismatch for {component}: "
            f"unaccounted={sorted(omitted_source - expected_omitted)[:3]}, "
            f"missing={sorted(expected_omitted - omitted_source)[:3]}"
        )


def verify_dense_checkpoint(
    source: str | Path,
    dense: str | Path,
    *,
    verify_digests: bool = False,
) -> DenseVerificationSummary:
    """Verify every stored dense tensor against the declared source mapping."""

    source = Path(source).resolve()
    dense = Path(dense).resolve()
    if source == dense:
        raise DenseVerificationError("Source and dense checkpoints must differ")

    manifest = CheckpointManifest.read(dense / "manifest.json")
    if manifest.profile != "dense":
        raise DenseVerificationError(
            f"Expected a dense manifest, got {manifest.profile!r}"
        )
    manifest.verify(dense, digests=verify_digests)
    if verify_digests:
        _verify_source_digests(source, manifest)

    plans = plan_checkpoint(source)
    source_file_count = 0
    dense_file_count = 0
    identity_tensors = 0
    transposed_tensors = 0
    folded_tensors = 0
    hardlinked_files = 0

    for component, mappings in plans.items():
        source_layout = discover_component_layout(source / component)
        dense_layout = discover_component_layout(dense / component)
        source_file_count += len(source_layout.files)
        dense_file_count += len(dense_layout.files)
        _validate_component_names(
            component,
            set(source_layout.tensors),
            set(dense_layout.tensors),
            mappings,
        )

        mappings_by_file: dict[Path, list[PlannedTensor]] = {
            path: [] for path in source_layout.files
        }
        for mapping in mappings:
            mappings_by_file[source_layout.tensors[mapping.source_name].file].append(
                mapping
            )

        for source_file, shard_mappings in mappings_by_file.items():
            dense_file = dense / component / source_file.name
            if not dense_file.is_file():
                raise DenseVerificationError(
                    f"Missing dense shard: {component}/{source_file.name}"
                )
            if os.path.samefile(source_file, dense_file):
                if any(mapping.operation != "identity" for mapping in shard_mappings):
                    raise DenseVerificationError(
                        f"Transformed shard is unexpectedly hardlinked: {dense_file}"
                    )
                hardlinked_files += 1
                identity_tensors += len(shard_mappings)
                continue

            source_weights = mx.load(str(source_file))
            dense_weights = mx.load(str(dense_file))
            for mapping in sorted(
                shard_mappings, key=lambda value: value.output_name
            ):
                expected = _expected_tensor(mapping, source_weights)
                actual = dense_weights[mapping.output_name]
                matches = mx.array_equal(actual, expected)
                mx.eval(matches)
                if not bool(matches.item()):
                    maximum_error = mx.max(
                        mx.abs(
                            actual.astype(mx.float32)
                            - expected.astype(mx.float32)
                        )
                    )
                    mx.eval(maximum_error)
                    raise DenseVerificationError(
                        f"Dense tensor mismatch: {component}.{mapping.output_name} "
                        f"({mapping.operation}, max_abs={maximum_error.item():.8g})"
                    )
                if mapping.operation == "identity":
                    identity_tensors += 1
                elif mapping.operation == "conv1d":
                    transposed_tensors += 1
                else:
                    folded_tensors += 1
            del dense_weights, source_weights
            mx.synchronize()
            mx.clear_cache()

    return DenseVerificationSummary(
        source_files=source_file_count,
        dense_files=dense_file_count,
        identity_tensors=identity_tensors,
        transposed_tensors=transposed_tensors,
        folded_weight_norm_tensors=folded_tensors,
        hardlinked_files=hardlinked_files,
        digests_verified=verify_digests,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dense", type=Path)
    parser.add_argument(
        "--verify-digests",
        action="store_true",
        help="Hash every source and dense file in addition to tensor checks.",
    )
    args = parser.parse_args()
    summary = verify_dense_checkpoint(
        args.source,
        args.dense,
        verify_digests=args.verify_digests,
    )
    report = asdict(summary)
    report["tensors_verified"] = summary.tensors_verified
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
