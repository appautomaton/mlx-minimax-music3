"""Tests for dependency-free SafeTensors checkpoint inspection."""

import json
from pathlib import Path

import pytest

from dev.inspect_checkpoint import (
    CheckpointInspectionError,
    inspect_checkpoint,
    inspect_safetensors,
)


def write_safetensors(path: Path, header: dict[str, object], data: bytes) -> None:
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(encoded_header).to_bytes(8, "little") + encoded_header + data)


def test_inspect_checkpoint_reports_components_and_dtypes(tmp_path: Path) -> None:
    write_safetensors(
        tmp_path / "language_model" / "model.safetensors",
        {
            "bf16_weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]},
            "fp32_weight": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
        },
        bytes(12),
    )

    report = inspect_checkpoint(tmp_path)

    assert report["summary"]["files"] == 1
    assert report["summary"]["tensors"] == 2
    assert report["summary"]["tensor_bytes"] == 12
    assert report["summary"]["components"]["language_model"]["dtypes"] == {
        "BF16": 8,
        "F32": 4,
    }


def test_inspect_safetensors_rejects_incorrect_tensor_size(tmp_path: Path) -> None:
    checkpoint = tmp_path / "broken.safetensors"
    write_safetensors(
        checkpoint,
        {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
        bytes(4),
    )

    with pytest.raises(CheckpointInspectionError, match="occupies 4 bytes, expected 8"):
        inspect_safetensors(checkpoint)
