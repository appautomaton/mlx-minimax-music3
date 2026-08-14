from __future__ import annotations

import mlx.core as mx
import pytest
from mlx import nn

from mlx_minimax_music3.stages import StageMemoryPolicy, StageSession


def test_stage_materializes_handoff_and_releases_model() -> None:
    baseline = int(mx.get_active_memory())
    session = StageSession(
        "tiny",
        lambda: nn.Linear(16, 16),
        policy=StageMemoryPolicy(settle_timeout=1.0),
        include_footprint=False,
    )
    with session:
        output = session.require_model()(mx.ones((1, 16)))
        session.handoff(output)
        assert mx.isfinite(output).all().item()

    assert session.model is None
    assert session.report is not None
    assert session.report.handoff_bytes == output.nbytes
    assert int(mx.get_active_memory()) <= baseline + 1024 * 1024
    with pytest.raises(RuntimeError, match="not active"):
        session.require_model()


def test_stage_policy_rejects_invalid_timings() -> None:
    with pytest.raises(ValueError, match="timings"):
        StageMemoryPolicy(settle_timeout=-1)


def test_stage_policy_rejects_invalid_memory_thresholds() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        StageMemoryPolicy(max_swap_growth_bytes=1.5)
    with pytest.raises(ValueError, match="exceed 100"):
        StageMemoryPolicy(min_free_percent=101)


def test_stage_allows_explicit_large_handoff() -> None:
    session = StageSession(
        "handoff",
        lambda: nn.Identity(),
        include_footprint=False,
    )

    with session:
        output = session.require_model()(mx.ones((512, 1024), dtype=mx.float32))
        session.handoff(output)

    assert session.report is not None
    assert session.report.handoff_bytes == 2 * 1024 * 1024
