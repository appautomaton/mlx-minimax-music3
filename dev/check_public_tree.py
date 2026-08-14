"""Reject local, generated, or private assets from the public source tree."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 10 * 1024 * 1024

FORBIDDEN_ROOTS = {
    ".references",
    "artifacts",
    "checkpoints",
    "inputs",
    "logs",
    "models",
    "outputs",
    "prompts",
    "runs",
    "samples",
    "tokenizer_cache",
    "weights",
}

ALLOWED_HIDDEN_DIRECTORIES = {".github", ".githooks"}
ALLOWED_HIDDEN_FILES = {".gitignore", ".python-version"}

FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".flac",
    ".gguf",
    ".log",
    ".m4a",
    ".mp3",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
}


def git_paths(*, cached: bool) -> list[Path]:
    command = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
        if cached
        else ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    result = subprocess.run(command, check=True, capture_output=True)
    return [Path(item) for item in result.stdout.decode().split("\0") if item]


def check_path(path: Path) -> list[str]:
    failures: list[str] = []
    posix = PurePosixPath(path.as_posix())
    hidden_directories = {
        part.lower()
        for part in posix.parts[:-1]
        if part.startswith(".") and part.lower() not in ALLOWED_HIDDEN_DIRECTORIES
    }
    if hidden_directories:
        failures.append("hidden local directory")
    root = posix.parts[0].lower() if posix.parts else ""
    if root in FORBIDDEN_ROOTS:
        failures.append(f"forbidden root directory: {root}")

    lower_name = path.name.lower()
    if lower_name.startswith(".") and lower_name not in ALLOWED_HIDDEN_FILES:
        failures.append("hidden local file")
    if lower_name == ".ds_store":
        failures.append("forbidden system metadata file")
    if any(lower_name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        failures.append("forbidden model or generated asset")
    if path.is_symlink():
        failures.append("symbolic links are not allowed in the public tree")
        return failures
    if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
        failures.append(f"file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MiB")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Check only paths staged for commit.",
    )
    args = parser.parse_args()

    failures = {
        path: reasons
        for path in git_paths(cached=args.cached)
        if (reasons := check_path(path))
    }
    if not failures:
        print("Public tree check passed.")
        return 0

    print("Public tree check failed:")
    for path, reasons in failures.items():
        print(f"  {path}: {'; '.join(reasons)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
