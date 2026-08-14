from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_minimax_music3.acoustic import FlowGenerationConfig, solve_flow_chunk


class ZeroTransformer:
    config = SimpleNamespace(in_channels=4, condition_dim=8)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        condition: mx.array,
    ) -> mx.array:
        del timestep, condition
        return mx.zeros_like(hidden_states)


class UnitTransformer:
    config = SimpleNamespace(in_channels=4, condition_dim=8)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        condition: mx.array,
    ) -> mx.array:
        del timestep, condition
        output = mx.zeros_like(hidden_states)
        output[0] = 1.0
        return output


def test_zero_velocity_preserves_seeded_noise() -> None:
    condition = mx.zeros((1, 12, 8))
    key = mx.random.key(99)
    expected = mx.random.normal((1, 12, 4), key=key)

    latents, _, _ = solve_flow_chunk(
        ZeroTransformer(),
        condition,
        key=key,
        config=FlowGenerationConfig(num_steps=3),
    )
    mx.eval(expected, latents)

    assert mx.array_equal(latents, expected).item()


def test_cfg_euler_update_and_overlap_restore() -> None:
    condition = mx.zeros((1, 400, 8))
    previous_latent = mx.full((1, 20, 4), 7.0)
    previous_condition = mx.full((1, 20, 8), 3.0)
    key = mx.random.key(7)
    initial_noise = mx.random.normal((1, 400, 4), key=key)

    latents, carry_latent, carry_condition = solve_flow_chunk(
        UnitTransformer(),
        condition,
        key=key,
        config=FlowGenerationConfig(num_steps=4, cfg_scale=1.5),
        previous_latent=previous_latent,
        previous_condition=previous_condition,
    )
    mx.eval(latents, carry_latent, carry_condition)

    assert mx.array_equal(latents[:, :20], previous_latent).item()
    assert mx.allclose(latents[:, 20:], initial_noise[:, 20:] + 1.5).item()
    assert carry_latent.shape == (1, 172, 4)
    assert carry_condition.shape == (1, 172, 8)
    assert mx.array_equal(carry_latent, latents[:, 56:228]).item()


def test_flow_config_rejects_non_finite_guidance() -> None:
    with pytest.raises(ValueError, match="finite"):
        FlowGenerationConfig(cfg_scale=float("nan"))
