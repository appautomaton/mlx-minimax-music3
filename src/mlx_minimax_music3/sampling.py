# Portions adapted from SGLang under the Apache License 2.0.
"""Numerically stable, reference-compatible sampling utilities."""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass

import mlx.core as mx

_UINT32_MASK = 0xFFFF_FFFF
_SAMPLING_SEED_MASK = 0x7FFF_FFFF
_SAMPLING_NAMESPACE = "minimax-ttm-ar"


def derive_sampling_seed(
    namespace: str, public_seed: object, label: str | None = None
) -> int:
    """Derive the positive-int32 request seed used by SGLang sampling."""

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("sampling namespace must be a non-empty string")
    parts = [namespace, str(public_seed)]
    if label is not None:
        parts.append(label)
    digest = hashlib.blake2b(":".join(parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "little") & _SAMPLING_SEED_MASK


def derive_acoustic_seed(seed: int, *parts: object) -> int:
    """Derive the per-chunk 63-bit noise seed used by SGLang Music 3."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be a non-negative 64-bit integer")
    digest = hashlib.blake2b(digest_size=8, person=b"minimax-ttm")
    digest.update(seed.to_bytes(8, "little", signed=False))
    for part in parts:
        raw = str(part).encode("utf-8")
        digest.update(len(raw).to_bytes(4, "little"))
        digest.update(raw)
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def _rotate_left_32(value: int, amount: int) -> int:
    value &= _UINT32_MASK
    return ((value << amount) | (value >> (32 - amount))) & _UINT32_MASK


def _murmur3_mix(hash_value: int, key: int) -> int:
    key = (key * 0xCC9E2D51) & _UINT32_MASK
    key = _rotate_left_32(key, 15)
    key = (key * 0x1B873593) & _UINT32_MASK
    hash_value ^= key
    hash_value = _rotate_left_32(hash_value, 13)
    return (hash_value * 5 + 0xE6546B64) & _UINT32_MASK


def murmur_hash32(seed: int, position: int, column: int) -> int:
    """Hash one `(seed, position, vocabulary column)` reference tuple."""

    if not 0 <= seed < 2**64:
        raise ValueError("sampling seed must fit uint64")
    if not 0 <= position <= _UINT32_MASK:
        raise ValueError("sampling position must fit uint32")
    if not 0 <= column <= _UINT32_MASK:
        raise ValueError("sampling column must fit uint32")

    hash_value = 0
    for key in (
        seed & _UINT32_MASK,
        (seed >> 32) & _UINT32_MASK,
        position,
        column,
    ):
        hash_value = _murmur3_mix(hash_value, key)
    hash_value ^= 16
    hash_value ^= hash_value >> 16
    hash_value = (hash_value * 0x85EBCA6B) & _UINT32_MASK
    hash_value ^= hash_value >> 13
    hash_value = (hash_value * 0xC2B2AE35) & _UINT32_MASK
    hash_value ^= hash_value >> 16
    return hash_value & _UINT32_MASK


@dataclass(frozen=True, slots=True)
class SeedSchedule:
    """Map a public Music 3 seed onto reference sampling positions."""

    seed: int
    num_codebooks: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must be a non-negative 64-bit integer")
        if isinstance(self.num_codebooks, bool) or not isinstance(
            self.num_codebooks, int
        ):
            raise TypeError("num_codebooks must be an integer")
        if self.num_codebooks <= 0:
            raise ValueError("num_codebooks must be positive")

    @property
    def sampling_seed(self) -> int:
        return derive_sampling_seed(_SAMPLING_NAMESPACE, self.seed)

    def position(self, *, frame_index: int, codebook_index: int) -> int:
        if frame_index < 0:
            raise ValueError("frame_index cannot be negative")
        if not 0 <= codebook_index < self.num_codebooks:
            raise ValueError(
                f"codebook_index must be in [0, {self.num_codebooks - 1}]"
            )
        return frame_index * self.num_codebooks + codebook_index


def sanitize_logits(logits: mx.array) -> mx.array:
    """Convert logits to FP32 and replace non-finite source values."""

    values = logits.astype(mx.float32)
    values = mx.where(mx.isnan(values), mx.array(-1e9, mx.float32), values)
    values = mx.where(values == mx.inf, mx.array(1e9, mx.float32), values)
    return mx.where(values == -mx.inf, mx.array(-1e9, mx.float32), values)


def restrict_top_k(logits: mx.array, top_k: int) -> mx.array:
    """Mask every value below the kth largest logit."""

    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    if not 0 < top_k <= logits.shape[-1]:
        raise ValueError(f"top_k must be in [1, {logits.shape[-1]}]")
    if top_k == logits.shape[-1]:
        return logits
    threshold = mx.topk(logits, k=top_k, axis=-1)[..., -1:]
    return mx.where(logits < threshold, -mx.inf, logits)


def _gumbel_noise(seed: int, position: int, column: int) -> float:
    hashed = murmur_hash32(seed, position, column)
    uniform = hashed / _UINT32_MASK
    negative_log = -math.log(uniform) if uniform else sys.float_info.max
    return -math.log(max(negative_log, 2.0**-32))


def sample_top_k(
    logits: mx.array,
    *,
    top_k: int,
    seed: int,
    position: int,
) -> mx.array:
    """Sample one row with SGLang's deterministic top-k Gumbel-max rule."""

    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("reference sampling expects logits with shape [1, vocab]")
    if not 0 < top_k <= logits.shape[-1]:
        raise ValueError(f"top_k must be in [1, {logits.shape[-1]}]")
    values = sanitize_logits(logits)
    if top_k == values.shape[-1]:
        candidate_indices = mx.arange(values.shape[-1], dtype=mx.int32)
    else:
        candidate_indices = mx.argpartition(values, kth=-top_k, axis=-1)[
            0, -top_k:
        ].astype(mx.int32)
    candidate_logits = values[0, candidate_indices]
    mx.eval(candidate_indices, candidate_logits)

    candidates = sorted(
        zip(candidate_indices.tolist(), candidate_logits.tolist(), strict=True)
    )
    best_column = max(
        candidates,
        key=lambda candidate: candidate[1]
        + _gumbel_noise(seed, position, candidate[0]),
    )[0]
    return mx.array([best_column], dtype=mx.int32)


def classifier_free_guidance(logits: mx.array, *, scale: float) -> mx.array:
    """Combine `[conditional, unconditional]` rows in FP32."""

    if logits.ndim != 2 or logits.shape[0] != 2:
        raise ValueError("CFG logits must have shape [2, vocabulary]")
    if scale < 0:
        raise ValueError("CFG scale cannot be negative")
    values = logits.astype(mx.float32)
    conditional = values[:1]
    unconditional = values[1:2]
    return unconditional + (conditional - unconditional) * scale
