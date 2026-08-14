from __future__ import annotations

from pathlib import Path

from dev.check_public_tree import check_path


def test_package_models_directory_is_public_source() -> None:
    assert check_path(Path("src/mlx_minimax_music3/models/qwen3.py")) == []


def test_root_models_directory_is_private() -> None:
    assert check_path(Path("models/local-config.json")) == [
        "forbidden root directory: models"
    ]
