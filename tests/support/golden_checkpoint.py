"""Build a deterministic miniature Music 3 checkpoint for regression tests."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mlx.utils import tree_flatten

from dev.quantize_checkpoint import quantize_checkpoint
from mlx_minimax_music3.checkpoint import inspect_safetensors
from mlx_minimax_music3.config import (
    ConditionEncoderConfig,
    FlowTransformerConfig,
    Qwen3Config,
    RVQDepthDecoderConfig,
    VocoderConfig,
)
from mlx_minimax_music3.manifest import (
    CheckpointManifest,
    ComponentManifest,
    ManifestFile,
    sha256_file,
)
from mlx_minimax_music3.models.condition_encoder import ConditionEncoder
from mlx_minimax_music3.models.flow_transformer import FlowTransformer
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM
from mlx_minimax_music3.models.rvq_depth import RVQDepthDecoder
from mlx_minimax_music3.models.vocoder import Vocoder

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "music3_golden_v1.json"
)
_FIXTURE_FORMAT = "mlx-minimax-music3-golden"
_FIXTURE_FORMAT_VERSION = 1
_DTYPES = {
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}


@dataclass(frozen=True, slots=True)
class GoldenCheckpoints:
    """Dense and selective-q8 paths generated from one deterministic fixture."""

    dense: Path
    q8: Path


def load_golden_contract() -> dict[str, Any]:
    """Load and minimally validate the versioned golden fixture contract."""

    data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Golden fixture must contain a JSON object")
    if data.get("format") != _FIXTURE_FORMAT:
        raise ValueError(f"Unsupported golden fixture format: {data.get('format')!r}")
    if data.get("format_version") != _FIXTURE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported golden fixture version: "
            f"{data.get('format_version')!r}"
        )
    expected = data.get("expected")
    if not isinstance(expected, dict) or set(expected) != {
        "dense",
        "q8",
        "runtime-f16-flow",
    }:
        raise ValueError(
            "Golden fixture must define dense, q8, and runtime-f16-flow expectations"
        )
    return data


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _golden_values(name: str, shape: tuple[int, ...], dtype: mx.Dtype) -> mx.array:
    """Create cross-device-stable values from a tensor's qualified name."""

    size = math.prod(shape)
    phase = int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:4], "little"
    ) % 29
    indices = mx.arange(size, dtype=mx.int32)
    values = (((indices + phase) % 29).astype(mx.float32) - 14.0) / 256.0
    if name.endswith(("norm.weight", "layernorm.weight")) or ".snake" in name:
        values = 0.875 + (((indices + phase) % 7).astype(mx.float32) / 128.0)
    elif "embed" in name and name.endswith("weight"):
        values = 0.125 + (((indices + phase) % 11).astype(mx.float32) / 256.0)
    return values.reshape(shape).astype(dtype)


def _save_model(
    root: Path,
    component: str,
    model,
    *,
    dtype: mx.Dtype,
    revision: str,
) -> Path:
    path = root / component / "model.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {
        name: _golden_values(f"{component}.{name}", tuple(value.shape), dtype)
        for name, value in tree_flatten(model.parameters())
    }
    mx.eval(list(weights.values()))
    mx.save_safetensors(
        str(path),
        weights,
        metadata={"fixture": revision},
    )
    del model, weights
    mx.clear_cache()
    return path


def _record(root: Path, path: Path) -> ManifestFile:
    relative = path.relative_to(root).as_posix()
    digest = sha256_file(path)
    tensor_count = 0
    dtypes: tuple[str, ...] = ()
    if path.suffix == ".safetensors":
        info = inspect_safetensors(path)
        tensor_count = len(info.tensors)
        dtypes = tuple(sorted({tensor.dtype for tensor in info.tensors}))
    return ManifestFile(
        path=relative,
        size=path.stat().st_size,
        sha256=digest,
        tensor_count=tensor_count,
        dtypes=dtypes,
        source_path=relative,
        source_sha256=digest,
    )


def _write_dense_manifest(
    root: Path,
    files: tuple[Path, ...],
    *,
    source_repository: str,
    source_revision: str,
) -> None:
    grouped: dict[str, list[ManifestFile]] = defaultdict(list)
    for path in files:
        relative = path.relative_to(root)
        grouped[relative.parts[0]].append(_record(root, path))
    manifest = CheckpointManifest(
        profile="dense",
        source_repository=source_repository,
        source_revision=source_revision,
        components=tuple(
            ComponentManifest(name, tuple(sorted(records, key=lambda item: item.path)))
            for name, records in sorted(grouped.items())
        ),
    )
    manifest.write(root / "manifest.json")


def build_golden_checkpoints(root: Path) -> GoldenCheckpoints:
    """Generate a tiny dense checkpoint and its real selective-q8 derivative."""

    contract = load_golden_contract()
    source = cast(dict[str, str], contract["source"])
    components = cast(dict[str, dict[str, Any]], contract["components"])
    assets = cast(dict[str, object], contract["assets"])
    dense = root / "dense"
    q8 = root / "q8"
    files: list[Path] = []
    for descriptor in components.values():
        path = dense / descriptor["config_path"]
        _write_json(path, descriptor["config"])
        files.append(path)
    for relative, value in assets.items():
        path = dense / relative
        _write_json(path, value)
        files.append(path)

    language = components["language_model"]
    depth = components["rvq_depth_decoder"]
    condition = components["condition_encoder"]
    transformer = components["transformer"]
    vocoder = components["vocoder"]
    revision = source["revision"]
    files.extend(
        (
            _save_model(
                dense,
                "language_model",
                Qwen3ForCausalLM(Qwen3Config.from_dict(language["config"])),
                dtype=_DTYPES[language["weights_dtype"]],
                revision=revision,
            ),
            _save_model(
                dense,
                "rvq_depth_decoder",
                RVQDepthDecoder(RVQDepthDecoderConfig.from_dict(depth["config"])),
                dtype=_DTYPES[depth["weights_dtype"]],
                revision=revision,
            ),
            _save_model(
                dense,
                "condition_encoder",
                ConditionEncoder(
                    ConditionEncoderConfig.from_dict(condition["config"])
                ),
                dtype=_DTYPES[condition["weights_dtype"]],
                revision=revision,
            ),
            _save_model(
                dense,
                "transformer",
                FlowTransformer(
                    FlowTransformerConfig.from_dict(transformer["config"])
                ),
                dtype=_DTYPES[transformer["weights_dtype"]],
                revision=revision,
            ),
            _save_model(
                dense,
                "vocoder",
                Vocoder(VocoderConfig.from_dict(vocoder["config"])),
                dtype=_DTYPES[vocoder["weights_dtype"]],
                revision=revision,
            ),
        )
    )
    _write_dense_manifest(
        dense,
        tuple(files),
        source_repository=source["repository"],
        source_revision=revision,
    )
    quantize_checkpoint(dense, q8, verify_source_digests=True)
    return GoldenCheckpoints(dense=dense, q8=q8)


__all__ = [
    "GoldenCheckpoints",
    "build_golden_checkpoints",
    "load_golden_contract",
]
