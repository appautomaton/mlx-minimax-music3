from __future__ import annotations

import mlx.core as mx

from mlx_minimax_music3.config import (
    ConditionEncoderConfig,
    FlowTransformerConfig,
)
from mlx_minimax_music3.models.condition_encoder import ConditionEncoder
from mlx_minimax_music3.models.flow_transformer import FlowTransformer


def test_condition_encoder_mixes_layers_and_resamples_nearest() -> None:
    config = ConditionEncoderConfig(
        condition_hidden_dim=2,
        num_condition_layers=2,
        out_dim=2,
        input_sampling_rate=1,
        input_hop_length=1,
        output_sampling_rate=2,
        output_hop_length=1,
    )
    model = ConditionEncoder(config)
    model.layer_weight_logits = mx.array([0.0, 0.0])
    model.layer_scale = mx.array([2.0])
    model.proj.weight = mx.zeros_like(model.proj.weight)
    model.proj.weight[:, 1, :] = mx.eye(2)
    model.proj.bias = mx.zeros_like(model.proj.bias)
    frames = mx.array([[[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]]])

    encoded = model(frames)
    mx.eval(encoded)

    assert encoded.shape == (1, 4, 2)
    assert mx.array_equal(
        encoded,
        mx.array([[[6.0, 10.0], [6.0, 10.0], [8.0, 12.0], [8.0, 12.0]]]),
    ).item()


def test_tiny_flow_transformer_preserves_latent_shape() -> None:
    config = FlowTransformerConfig(
        in_channels=4,
        condition_dim=8,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=4,
        ff_inner_dim=16,
        fourier_embedding_dim=8,
        rotary_dim=4,
    )
    model = FlowTransformer(config)
    latents = mx.random.normal((2, 7, 4))
    condition = mx.random.normal((2, 7, 8))

    velocity = model(latents, mx.array([0.25, 0.75]), condition)
    mx.eval(velocity)

    assert velocity.shape == latents.shape
    assert mx.isfinite(velocity).all().item()


def test_flow_transformer_rejects_misaligned_condition() -> None:
    config = FlowTransformerConfig(
        in_channels=4,
        condition_dim=8,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=4,
        ff_inner_dim=16,
        fourier_embedding_dim=8,
        rotary_dim=4,
    )
    model = FlowTransformer(config)

    try:
        model(
            mx.zeros((1, 4, 4)),
            mx.array(0.5),
            mx.zeros((1, 5, 8)),
        )
    except ValueError as error:
        assert "share batch and length" in str(error)
    else:
        raise AssertionError("Expected misaligned condition to be rejected")
