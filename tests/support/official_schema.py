"""Load the metadata-only Official Music 3 schema contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "music3_official_schema_v1.json"
)
_FIXTURE_FORMAT = "mlx-minimax-music3-official-schema"
_FIXTURE_FORMAT_VERSION = 1


def load_official_schema_contract() -> dict[str, Any]:
    """Load and minimally validate the Official Music 3 schema fixture."""

    data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Official schema fixture must contain a JSON object")
    if data.get("format") != _FIXTURE_FORMAT:
        raise ValueError(
            f"Unsupported Official schema format: {data.get('format')!r}"
        )
    if data.get("format_version") != _FIXTURE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported Official schema version: "
            f"{data.get('format_version')!r}"
        )
    components = data.get("components")
    if not isinstance(components, dict) or set(components) != {
        "language_model",
        "rvq_depth_decoder",
        "condition_encoder",
        "transformer",
        "vocoder",
    }:
        raise ValueError("Official schema fixture has an invalid component set")
    return data


__all__ = ["load_official_schema_contract"]
