"""Memory-conscious key-value caches for autoregressive attention."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx


class CacheCapacityError(ValueError):
    """Raised when an append would exceed a cache's planned capacity."""


def causal_mask(length: int, offset: int = 0) -> mx.array | None:
    """Return a causal boolean mask for queries starting at ``offset``."""

    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if offset < 0:
        raise ValueError(f"offset cannot be negative, got {offset}")
    if length == 1:
        return None
    key_positions = mx.arange(offset + length)[None, :]
    query_positions = mx.arange(offset, offset + length)[:, None]
    return query_positions >= key_positions


class KVCache:
    """Append-only attention cache with optional one-shot preallocation.

    Passing ``capacity`` reserves the known request length on the first append.
    This avoids repeated copies in long generation loops and turns an accidental
    overrun into a clear error. Without it, storage grows in ``step_size`` blocks.
    """

    def __init__(self, *, capacity: int | None = None, step_size: int = 256) -> None:
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self._planned_capacity = capacity
        self._step_size = step_size
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    @property
    def capacity(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    @property
    def nbytes(self) -> int:
        if self.keys is None or self.values is None:
            return 0
        return self.keys.nbytes + self.values.nbytes

    def _allocation_size(self, required: int) -> int:
        if self._planned_capacity is not None:
            if required > self._planned_capacity:
                raise CacheCapacityError(
                    f"KV cache requires {required} positions but was planned for "
                    f"{self._planned_capacity}"
                )
            return self._planned_capacity
        return ((required + self._step_size - 1) // self._step_size) * self._step_size

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append ``[batch, heads, length, dim]`` keys and values."""

        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must have rank 4")
        if keys.shape[:3] != values.shape[:3]:
            raise ValueError("keys and values must share batch, head, and length axes")
        if self.keys is not None:
            if keys.shape[:2] != self.keys.shape[:2]:
                raise ValueError("key batch or head count changed within a cache")
            if values.shape[:2] != self.values.shape[:2]:
                raise ValueError("value batch or head count changed within a cache")
            if keys.shape[3] != self.keys.shape[3]:
                raise ValueError("key head dimension changed within a cache")
            if values.shape[3] != self.values.shape[3]:
                raise ValueError("value head dimension changed within a cache")

        start = self.offset
        required = start + keys.shape[2]
        if required > self.capacity:
            new_capacity = self._allocation_size(required)
            key_storage = mx.zeros(
                (*keys.shape[:2], new_capacity, keys.shape[3]), dtype=keys.dtype
            )
            value_storage = mx.zeros(
                (*values.shape[:2], new_capacity, values.shape[3]), dtype=values.dtype
            )
            if self.keys is not None and self.values is not None:
                key_storage[..., :start, :] = self.keys[..., :start, :]
                value_storage[..., :start, :] = self.values[..., :start, :]
            self.keys = key_storage
            self.values = value_storage

        self.offset = required
        self.keys[..., start:required, :] = keys
        self.values[..., start:required, :] = values
        return (
            self.keys[..., :required, :],
            self.values[..., :required, :],
        )

    def reset(self) -> None:
        """Release all cache storage."""

        self.keys = None
        self.values = None
        self.offset = 0


def make_kv_caches(
    num_layers: int, *, capacity: int | None = None, step_size: int = 256
) -> list[KVCache]:
    """Create one independent cache per transformer layer."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    return [
        KVCache(capacity=capacity, step_size=step_size) for _ in range(num_layers)
    ]


def validate_cache_sequence(
    caches: Sequence[KVCache] | None, *, num_layers: int
) -> int:
    """Validate a cache list and return its common sequence offset."""

    if caches is None:
        return 0
    if len(caches) != num_layers:
        raise ValueError(f"Expected {num_layers} layer caches, got {len(caches)}")
    offsets = {cache.offset for cache in caches}
    if len(offsets) != 1:
        raise ValueError("All layer caches must have the same sequence offset")
    return next(iter(offsets))
