"""Verify that package and project versions agree."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path


def package_version() -> str:
    tree = ast.parse(
        Path("src/mlx_minimax_music3/_version.py").read_text(encoding="utf-8")
    )
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "__version__" in targets and isinstance(node.value, ast.Constant):
                return str(node.value.value)
    raise RuntimeError("package __version__ assignment is missing")


def main() -> int:
    with Path("pyproject.toml").open("rb") as file:
        project_version = tomllib.load(file)["project"]["version"]
    source_version = package_version()
    if project_version != source_version:
        raise RuntimeError(
            f"pyproject version {project_version!r} does not match "
            f"package version {source_version!r}"
        )
    print(f"Version is consistent: {project_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

