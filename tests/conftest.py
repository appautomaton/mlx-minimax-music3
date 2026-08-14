"""Shared fixtures for weightless, platform-independent tests."""

from __future__ import annotations

import time

import pytest

from mlx_minimax_music3 import stages
from mlx_minimax_music3.memory import (
    MemorySnapshot,
    SystemMemory,
    capture_mlx_memory,
)


@pytest.fixture
def isolated_stage_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real MLX telemetry while isolating macOS-only system commands."""

    def capture(label: str, *, include_footprint: bool = False) -> MemorySnapshot:
        del include_footprint
        return MemorySnapshot(
            label=label,
            monotonic_ns=time.monotonic_ns(),
            mlx=capture_mlx_memory(),
            system=SystemMemory(
                process_rss_bytes=0,
                process_footprint_bytes=None,
                process_footprint_peak_bytes=None,
                swap_used_bytes=0,
                swapins_bytes=0,
                swapouts_bytes=0,
                free_percent=100,
            ),
        )

    monkeypatch.setattr(stages, "capture_memory", capture)
