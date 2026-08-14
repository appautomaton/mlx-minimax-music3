#!/usr/bin/env python3
"""Publish the MLX-dense MiniMax Music 3 checkpoint to Hugging Face.

python scripts/hugging_face/upload.py --dry-run
python scripts/hugging_face/upload.py --card-only
python scripts/hugging_face/upload.py

The large-folder upload is resumable. The tracked model card is uploaded last
as the repository README instead of trusting checkpoint-local documentation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = "appautomaton/MiniMax-Music3-MLX"
SOURCE = Path("weights/mlx-dense/MiniMax-Music3")
CARD = Path(
    "scripts/hugging_face/model_cards/appautomaton/minimax-music3-mlx.md"
)
REQUIRED_CHECKPOINT_FILES = ("LICENSE", "manifest.json")
EXCLUDED_CHECKPOINT_FILES = frozenset({"README.md", "modular_model_index.json"})


def run(command: list[str], env: dict[str, str]) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    if subprocess.run(command, check=False, env=env).returncode != 0:
        sys.exit(1)


def _publishable_files(source: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(source.rglob("*"))
        if path.is_file()
        and ".cache" not in path.parts
        and path.relative_to(source).as_posix() not in EXCLUDED_CHECKPOINT_FILES
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    card = root / CARD
    source = root / SOURCE
    if not card.is_file():
        raise FileNotFoundError(f"missing model card: {card}")
    if not args.card_only:
        missing = [
            name for name in REQUIRED_CHECKPOINT_FILES if not (source / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"missing under {source}: {', '.join(missing)}"
            )

    files = () if args.card_only else _publishable_files(source)
    if args.dry_run:
        print(f"repo: {REPO_ID}")
        print(f"checkpoint files: {len(files)}")
        print(f"checkpoint bytes: {sum(path.stat().st_size for path in files)}")
        print(f"README.md <- {CARD}")
        return 0

    hf = shutil.which("hf")
    if hf is None:
        raise FileNotFoundError("no hf CLI on PATH; install huggingface_hub")

    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"
    if not args.card_only:
        run(
            [
                hf,
                "upload-large-folder",
                "--repo-type",
                "model",
                "--num-workers",
                "1",
                "--exclude",
                "README.md",
                "--exclude",
                "modular_model_index.json",
                "--exclude",
                ".cache/**",
                REPO_ID,
                str(source),
            ],
            env,
        )
    run([hf, "upload", "--repo-type", "model", REPO_ID, str(card), "README.md"], env)
    print(f"Done. https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
