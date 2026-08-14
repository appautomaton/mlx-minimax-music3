"""Explicit model residency sessions with OS-verified teardown."""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from .memory import MemorySnapshot, capture_memory, memory_delta


class StageMemoryError(RuntimeError):
    """Raised when a completed stage violates its memory-safety policy."""


@dataclass(frozen=True, slots=True)
class StageMemoryPolicy:
    """Delta-based memory limits that avoid false positives from old swap."""

    settle_timeout: float = 3.0
    settle_poll_interval: float = 0.25
    footprint_slack_bytes: int = 256 * 1024 * 1024
    allocator_slack_bytes: int = 1024 * 1024
    max_swap_growth_bytes: int = 256 * 1024 * 1024
    max_swapout_growth_bytes: int = 256 * 1024 * 1024
    min_free_percent: int = 5

    def __post_init__(self) -> None:
        if (
            isinstance(self.settle_timeout, bool)
            or not isinstance(self.settle_timeout, int | float)
            or not math.isfinite(self.settle_timeout)
            or isinstance(self.settle_poll_interval, bool)
            or not isinstance(self.settle_poll_interval, int | float)
            or not math.isfinite(self.settle_poll_interval)
            or self.settle_timeout < 0
            or self.settle_poll_interval <= 0
        ):
            raise ValueError("Memory settle timings must be positive")
        for name in (
            "footprint_slack_bytes",
            "allocator_slack_bytes",
            "max_swap_growth_bytes",
            "max_swapout_growth_bytes",
            "min_free_percent",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.min_free_percent > 100:
            raise ValueError("min_free_percent cannot exceed 100")


DEFAULT_STAGE_MEMORY_POLICY = StageMemoryPolicy()


@dataclass(frozen=True, slots=True)
class StageMemoryReport:
    label: str
    before: MemorySnapshot
    loaded: MemorySnapshot
    released: MemorySnapshot
    handoff_bytes: int

    @property
    def peak_active_bytes(self) -> int:
        return self.released.mlx.peak_bytes


class StageSession[ModelT]:
    """Own one model and verify that its residency ends at context exit.

    The context returns the session, not the model, so clearing ``session.model``
    removes the normal caller reference during teardown. Cross-stage arrays must
    be passed through :meth:`handoff` before leaving the context.
    """

    def __init__(
        self,
        label: str,
        loader: Callable[[], ModelT],
        *,
        policy: StageMemoryPolicy = DEFAULT_STAGE_MEMORY_POLICY,
        include_footprint: bool = True,
    ) -> None:
        if not label:
            raise ValueError("Stage label cannot be empty")
        self.label = label
        self._loader = loader
        self.policy = policy
        self.include_footprint = include_footprint
        self.model: ModelT | None = None
        self.report: StageMemoryReport | None = None
        self._before: MemorySnapshot | None = None
        self._loaded: MemorySnapshot | None = None
        self._handoff_ids: set[int] = set()
        self._handoff_bytes = 0

    def __enter__(self) -> StageSession[ModelT]:
        import mlx.core as mx

        if self.model is not None:
            raise RuntimeError("Stage session is already active")
        self._handoff_ids.clear()
        self._handoff_bytes = 0
        self._before = capture_memory(
            f"{self.label}:before",
            include_footprint=self.include_footprint,
        )
        if self._before.system.free_percent < self.policy.min_free_percent:
            raise StageMemoryError(
                f"Stage {self.label!r}: system free memory is "
                f"{self._before.system.free_percent}% before model loading"
            )
        mx.reset_peak_memory()
        try:
            self.model = self._loader()
            self._loaded = capture_memory(
                f"{self.label}:loaded",
                include_footprint=self.include_footprint,
            )
        except BaseException:
            self._release()
            raise
        return self

    def require_model(self) -> ModelT:
        if self.model is None:
            raise RuntimeError(f"Stage {self.label!r} is not active")
        return self.model

    def handoff(self, *arrays: object) -> None:
        """Materialize durable outputs before model teardown."""

        if self.model is None:
            raise RuntimeError(f"Stage {self.label!r} is not active")
        if not arrays:
            raise ValueError("handoff requires at least one array or array tree")
        import mlx.core as mx

        mx.eval(arrays)
        self._record_handoff(arrays)

    def _record_handoff(self, value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                self._record_handoff(child)
            return
        if isinstance(value, tuple | list):
            for child in value:
                self._record_handoff(child)
            return
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int) and id(value) not in self._handoff_ids:
            self._handoff_ids.add(id(value))
            self._handoff_bytes += nbytes

    def _release(self) -> MemorySnapshot:
        import mlx.core as mx

        self.model = None
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        released = capture_memory(
            f"{self.label}:released",
            include_footprint=self.include_footprint,
        )

        before = self._before
        if (
            self.include_footprint
            and before is not None
            and before.system.process_footprint_bytes is not None
            and released.system.process_footprint_bytes is not None
        ):
            target = (
                before.system.process_footprint_bytes
                + self._handoff_bytes
                + self.policy.footprint_slack_bytes
            )
            deadline = time.monotonic() + self.policy.settle_timeout
            while (
                released.system.process_footprint_bytes > target
                and time.monotonic() < deadline
            ):
                time.sleep(
                    min(
                        self.policy.settle_poll_interval,
                        max(0.0, deadline - time.monotonic()),
                    )
                )
                released = capture_memory(
                    f"{self.label}:settling",
                    include_footprint=True,
                )
        return released

    def _validate_release(
        self, before: MemorySnapshot, released: MemorySnapshot
    ) -> None:
        delta = memory_delta(before, released)
        failures = []
        if released.mlx.active_bytes > (
            before.mlx.active_bytes
            + self._handoff_bytes
            + self.policy.allocator_slack_bytes
        ):
            failures.append(
                f"MLX active memory retained {delta.active_bytes} bytes"
            )
        if released.mlx.cache_bytes > (
            before.mlx.cache_bytes + self.policy.allocator_slack_bytes
        ):
            failures.append(f"MLX cache retained {delta.cache_bytes} bytes")
        if (
            delta.process_footprint_bytes is not None
            and delta.process_footprint_bytes
            > self._handoff_bytes + self.policy.footprint_slack_bytes
        ):
            failures.append(
                "process footprint did not settle: "
                f"+{delta.process_footprint_bytes} bytes"
            )
        if delta.swap_used_bytes > self.policy.max_swap_growth_bytes:
            failures.append(f"swap grew by {delta.swap_used_bytes} bytes")
        if delta.swapouts_bytes > self.policy.max_swapout_growth_bytes:
            failures.append(f"swap-outs grew by {delta.swapouts_bytes} bytes")
        if released.system.free_percent < self.policy.min_free_percent:
            failures.append(
                f"system free memory is {released.system.free_percent}%"
            )
        if failures:
            raise StageMemoryError(f"Stage {self.label!r}: " + "; ".join(failures))

    def __exit__(self, exception_type, exception, traceback) -> bool:
        del exception_type, traceback
        released = self._release()
        if self._before is not None and self._loaded is not None:
            self.report = StageMemoryReport(
                label=self.label,
                before=self._before,
                loaded=self._loaded,
                released=released,
                handoff_bytes=self._handoff_bytes,
            )
            if exception is None:
                self._validate_release(self._before, released)
        return False
