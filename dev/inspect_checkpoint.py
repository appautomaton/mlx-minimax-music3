"""Inspect SafeTensors checkpoint structure without loading model arrays."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mlx_minimax_music3.checkpoint import (
    CheckpointInspectionError,
    SafeTensorsFileInfo,
    TensorInfo,
    inspect_safetensors,
)

__all__ = [
    "CheckpointInspectionError",
    "SafeTensorsFileInfo",
    "TensorInfo",
    "inspect_checkpoint",
    "inspect_safetensors",
]


def inspect_checkpoint(root: Path) -> dict[str, Any]:
    """Inspect every SafeTensors file below a componentized checkpoint root."""

    root = root.resolve()
    checkpoint_paths = sorted(
        path for path in root.rglob("*.safetensors") if ".cache" not in path.parts
    )
    if not checkpoint_paths:
        raise CheckpointInspectionError(f"no SafeTensors files found under {root}")

    component_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "files": 0,
            "tensors": 0,
            "tensor_bytes": 0,
            "dtypes": defaultdict(int),
        }
    )
    files = []
    for path in checkpoint_paths:
        info = inspect_safetensors(path)
        relative_path = path.relative_to(root)
        component = relative_path.parts[0]
        stats = component_stats[component]
        stats["files"] += 1
        stats["tensors"] += len(info.tensors)
        stats["tensor_bytes"] += info.tensor_bytes
        for tensor in info.tensors:
            stats["dtypes"][tensor.dtype] += tensor.byte_size

        serialized = asdict(info)
        serialized["path"] = str(relative_path)
        files.append(serialized)

    components = {}
    for component, raw_stats in sorted(component_stats.items()):
        components[component] = {
            "files": raw_stats["files"],
            "tensors": raw_stats["tensors"],
            "tensor_bytes": raw_stats["tensor_bytes"],
            "dtypes": dict(sorted(raw_stats["dtypes"].items())),
        }

    return {
        "root": str(root),
        "files": files,
        "summary": {
            "files": len(files),
            "tensors": sum(
                component["tensors"] for component in components.values()
            ),
            "tensor_bytes": sum(
                component["tensor_bytes"] for component in components.values()
            ),
            "components": components,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="Write the JSON report here.")
    args = parser.parse_args()

    report = inspect_checkpoint(args.checkpoint)
    encoded_report = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded_report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded_report, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
