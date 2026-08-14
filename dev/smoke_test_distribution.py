"""Smoke-test an installed release distribution."""

from __future__ import annotations

from importlib.metadata import version

import mlx

import mlx_minimax_music3


def main() -> int:
    distribution_version = version("mlx-minimax-music3")
    if mlx_minimax_music3.__version__ != distribution_version:
        raise RuntimeError(
            "installed package version does not match distribution metadata: "
            f"{mlx_minimax_music3.__version__!r} != {distribution_version!r}"
        )
    if not hasattr(mlx, "core"):
        raise RuntimeError("installed MLX package does not expose mlx.core")
    if not hasattr(mlx_minimax_music3, "Music3Pipeline"):
        raise RuntimeError("installed package does not expose Music3Pipeline")

    print(f"Distribution smoke test passed: {distribution_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
