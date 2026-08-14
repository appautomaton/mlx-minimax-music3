"""Dependency-free SafeTensors structure validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class CheckpointInspectionError(ValueError):
    """Raised when a SafeTensors file violates its structural contract."""


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    numel: int
    byte_size: int


@dataclass(frozen=True, slots=True)
class SafeTensorsFileInfo:
    path: str
    file_size: int
    header_size: int
    tensor_bytes: int
    metadata: dict[str, str]
    tensors: tuple[TensorInfo, ...]


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting ambiguous duplicate keys."""

    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CheckpointInspectionError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON object with duplicate-key detection."""

    path = Path(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_json_keys,
        )
    except FileNotFoundError as error:
        raise CheckpointInspectionError(f"Missing JSON file: {path}") from error
    except json.JSONDecodeError as error:
        raise CheckpointInspectionError(f"Invalid JSON in {path}") from error
    if not isinstance(value, dict):
        raise CheckpointInspectionError(f"JSON root must be an object: {path}")
    return value


def inspect_safetensors(path: str | Path) -> SafeTensorsFileInfo:
    """Read and validate one SafeTensors header without touching its payload."""

    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as checkpoint:
        encoded_header_size = checkpoint.read(8)
        if len(encoded_header_size) != 8:
            raise CheckpointInspectionError(f"{path}: missing 8-byte header length")
        header_size = int.from_bytes(encoded_header_size, "little", signed=False)
        if header_size <= 0 or header_size > file_size - 8:
            raise CheckpointInspectionError(
                f"{path}: invalid header size {header_size} for {file_size}-byte file"
            )
        encoded_header = checkpoint.read(header_size)

    try:
        header = json.loads(
            encoded_header,
            object_pairs_hook=reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CheckpointInspectionError(f"{path}: invalid JSON header") from error
    if not isinstance(header, dict):
        raise CheckpointInspectionError(f"{path}: header must be a JSON object")

    raw_metadata = header.pop("__metadata__", {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_metadata.items()
    ):
        raise CheckpointInspectionError(
            f"{path}: __metadata__ must map strings to strings"
        )

    tensors = []
    ranges = []
    for name, raw_info in header.items():
        if not isinstance(raw_info, dict):
            raise CheckpointInspectionError(
                f"{path}: tensor {name!r} metadata must be an object"
            )

        dtype = raw_info.get("dtype")
        shape = raw_info.get("shape")
        offsets = raw_info.get("data_offsets")
        if dtype not in _DTYPE_BYTES:
            raise CheckpointInspectionError(
                f"{path}: tensor {name!r} has unsupported dtype {dtype!r}"
            )
        if not isinstance(shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in shape
        ):
            raise CheckpointInspectionError(
                f"{path}: tensor {name!r} has invalid shape {shape!r}"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(offset, bool) or not isinstance(offset, int)
                for offset in offsets
            )
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise CheckpointInspectionError(
                f"{path}: tensor {name!r} has invalid offsets {offsets!r}"
            )

        numel = prod(shape)
        byte_size = offsets[1] - offsets[0]
        expected_bytes = numel * _DTYPE_BYTES[dtype]
        if byte_size != expected_bytes:
            raise CheckpointInspectionError(
                f"{path}: tensor {name!r} occupies {byte_size} bytes, "
                f"expected {expected_bytes}"
            )
        data_offsets = (offsets[0], offsets[1])
        ranges.append((*data_offsets, name))
        tensors.append(
            TensorInfo(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                data_offsets=data_offsets,
                numel=numel,
                byte_size=byte_size,
            )
        )

    previous_end = 0
    for start, end, name in sorted(ranges):
        if start != previous_end:
            raise CheckpointInspectionError(
                f"{path}: tensor {name!r} starts at {start}, "
                f"expected contiguous offset {previous_end}"
            )
        previous_end = end

    expected_file_size = 8 + header_size + previous_end
    if file_size != expected_file_size:
        raise CheckpointInspectionError(
            f"{path}: file size is {file_size}, expected {expected_file_size} "
            "from header"
        )

    return SafeTensorsFileInfo(
        path=str(path),
        file_size=file_size,
        header_size=header_size,
        tensor_bytes=previous_end,
        metadata=dict(raw_metadata),
        tensors=tuple(sorted(tensors, key=lambda tensor: tensor.name)),
    )
