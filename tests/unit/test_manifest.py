from __future__ import annotations

from pathlib import Path

import pytest

from mlx_minimax_music3.manifest import (
    CheckpointManifest,
    ComponentManifest,
    ManifestError,
    ManifestFile,
    sha256_file,
)


def test_manifest_round_trip_and_integrity(tmp_path: Path) -> None:
    weight = tmp_path / "language_model/model.safetensors"
    weight.parent.mkdir()
    weight.write_bytes(b"checkpoint")
    record = ManifestFile(
        path="language_model/model.safetensors",
        size=weight.stat().st_size,
        sha256=sha256_file(weight),
        tensor_count=1,
        dtypes=("BF16",),
    )
    manifest = CheckpointManifest(
        profile="dense",
        source_repository="MiniMaxAI/MiniMax-Music3",
        source_revision="a" * 40,
        components=(
            ComponentManifest(name="language_model", files=(record,)),
        ),
    )
    path = tmp_path / "manifest.json"

    manifest.write(path)
    restored = CheckpointManifest.read(path)
    restored.verify(tmp_path)

    assert restored == manifest


def test_manifest_rejects_parent_traversal() -> None:
    with pytest.raises(ManifestError, match="inside its root"):
        ManifestFile(
            path="../model.safetensors",
            size=0,
            sha256="0" * 64,
        )


def test_dense_manifest_rejects_quantized_modules() -> None:
    with pytest.raises(ManifestError, match="cannot declare quantization"):
        CheckpointManifest(
            profile="dense",
            source_repository="owner/model",
            source_revision="revision",
            components=(),
            quantized_modules=("model.layers.0.mlp.gate_proj",),
        )


def test_q8_manifest_rejects_duplicate_module_declarations() -> None:
    with pytest.raises(ManifestError, match="unique"):
        CheckpointManifest(
            profile="q8",
            source_repository="owner/model",
            source_revision="revision",
            components=(),
            quantized_modules=("language_model.model",) * 2,
            quantization_mode="affine",
            quantization_bits=8,
            quantization_group_size=64,
        )
