from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_minimax_music3.sampling import (
    SeedSchedule,
    classifier_free_guidance,
    derive_acoustic_seed,
    derive_sampling_seed,
    murmur_hash32,
    sample_top_k,
)


def test_seed_schedule_is_position_stable() -> None:
    schedule = SeedSchedule(seed=42, num_codebooks=8)

    assert schedule.sampling_seed == 1_153_769_322
    assert schedule.position(frame_index=3, codebook_index=4) == 28
    assert schedule.position(frame_index=3, codebook_index=5) == 29


def test_seed_derivation_matches_sglang_reference_vectors() -> None:
    assert derive_sampling_seed("minimax-ttm-ar", 0) == 411_363_039
    assert derive_sampling_seed("minimax-ttm-ar", 3) == 122_074_423
    assert derive_acoustic_seed(3, "dit", 0) == 1_648_571_301_556_962_109
    assert derive_acoustic_seed(3, "dit", 1) == 5_080_984_550_091_710_395


def test_murmur_hash_matches_reference_vectors() -> None:
    seed = 1_153_769_322

    assert murmur_hash32(seed, 0, 0) == 1_795_620_021
    assert murmur_hash32(seed, 0, 1) == 1_827_116_861
    assert murmur_hash32(seed, 7, 1_023) == 390_523_399
    assert murmur_hash32(seed, 24, 16_384) == 3_048_668_828


def test_top_k_sampling_never_draws_a_masked_token() -> None:
    logits = mx.array([[10.0, 9.0, 8.0, 7.0]])
    samples = [
        int(
            sample_top_k(
                logits,
                top_k=2,
                seed=derive_sampling_seed("minimax-ttm-ar", seed),
                position=0,
            ).item()
        )
        for seed in range(32)
    ]

    assert set(samples) <= {0, 1}


def test_top_k_sampling_matches_reference_gumbel_vector() -> None:
    sampled = sample_top_k(
        mx.array([[1.0, 2.0, 3.0, 4.0]]),
        top_k=4,
        seed=1_809_552_049,
        position=0,
    )

    assert sampled.item() == 3


def test_classifier_free_guidance_uses_fp32() -> None:
    logits = mx.array([[3.0, 5.0], [1.0, 2.0]], dtype=mx.bfloat16)

    guided = classifier_free_guidance(logits, scale=1.5)
    mx.eval(guided)

    assert guided.dtype == mx.float32
    assert mx.allclose(guided, mx.array([[4.0, 6.5]])).item()


def test_seed_schedule_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="64-bit"):
        SeedSchedule(seed=-1, num_codebooks=8)
