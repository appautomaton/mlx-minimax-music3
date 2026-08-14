"""Public package and runtime dependency boundary tests."""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

from mlx_minimax_music3 import (
    ExperimentalQuantizationWarning,
    GenerationRequest,
    GenerationResult,
    Music3Pipeline,
    PromptQualityWarning,
    __version__,
    instrumental_lyrics,
)


def test_package_version() -> None:
    assert __version__ == "0.0.1a0"


def test_generation_api_is_public() -> None:
    assert ExperimentalQuantizationWarning.__module__ == "mlx_minimax_music3.pipeline"
    assert GenerationRequest.__module__ == "mlx_minimax_music3.pipeline"
    assert GenerationResult.__module__ == "mlx_minimax_music3.pipeline"
    assert Music3Pipeline.__module__ == "mlx_minimax_music3.pipeline"
    assert PromptQualityWarning.__module__ == "mlx_minimax_music3.prompting"
    assert instrumental_lyrics.__module__ == "mlx_minimax_music3.prompting"


def test_runtime_imports_are_stdlib_or_mlx_only() -> None:
    source_root = Path("src/mlx_minimax_music3")
    allowed_roots = set(sys.stdlib_module_names) | {"mlx", "mlx_minimax_music3"}
    unexpected: dict[str, set[str]] = {}

    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.partition(".")[0])
        if forbidden := roots - allowed_roots:
            unexpected[path.as_posix()] = forbidden

    assert not unexpected


def test_mlx_is_the_only_runtime_dependency() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["dependencies"] == ["mlx>=0.32"]


def test_distribution_includes_all_license_material() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert set(project["project"]["license-files"]) == {
        "LICENSE",
        "LICENSES/Apache-2.0.txt",
        "THIRD_PARTY_NOTICES.md",
    }
