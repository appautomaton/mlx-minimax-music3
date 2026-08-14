# Copyright 2026 The MiniMax Team and The HuggingFace Team.
# Adapted from Hugging Face Diffusers under the Apache License 2.0.
"""MLX flow-matching diffusion transformer for MiniMax Music 3."""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from ..config import FlowTransformerConfig


class FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim % 2:
            raise ValueError("Fourier embedding dimension must be even")
        self.weight = mx.random.normal((embedding_dim // 2, 1))

    def __call__(self, timestep: mx.array) -> mx.array:
        if timestep.ndim == 0:
            timestep = timestep[None]
        if timestep.ndim != 1:
            raise ValueError("timestep must be a scalar or rank-1 array")
        angles = 2.0 * math.pi * (timestep[:, None] @ self.weight.T)
        return mx.concatenate((mx.cos(angles), mx.sin(angles)), axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, embedding: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(embedding)))


class FlowAttention(nn.Module):
    def __init__(self, config: FlowTransformerConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = [nn.Linear(hidden_size, hidden_size, bias=False)]
        self._rope = nn.RoPE(
            config.rotary_dim,
            traditional=False,
            base=10_000.0,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        head_shape = (
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        queries = self.to_q(hidden_states).reshape(head_shape).transpose(0, 2, 1, 3)
        keys = self.to_k(hidden_states).reshape(head_shape).transpose(0, 2, 1, 3)
        values = self.to_v(hidden_states).reshape(head_shape).transpose(0, 2, 1, 3)
        queries = self._rope(queries)
        keys = self._rope(keys)
        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, hidden_size
        )
        return self.to_out[0](attended)


class FlowTransformerBlock(nn.Module):
    def __init__(self, config: FlowTransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = FlowAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ff_in = nn.Linear(config.hidden_size, config.ff_inner_dim * 2)
        self.ff_out = nn.Linear(config.ff_inner_dim, config.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        value, gate = mx.split(self.ff_in(self.norm2(hidden_states)), 2, axis=-1)
        return hidden_states + self.ff_out(value * nn.silu(gate))


class FlowTransformer(nn.Module):
    """Predict flow velocity for channels-last audio latent sequences."""

    def __init__(self, config: FlowTransformerConfig) -> None:
        super().__init__()
        self.config = config
        concat_channels = 2 * config.in_channels + config.condition_dim
        self.time_proj = FourierEmbedding(config.fourier_embedding_dim)
        self.time_embed = TimestepEmbedding(
            config.fourier_embedding_dim, config.hidden_size
        )
        self.preprocess_conv = nn.Conv1d(
            concat_channels,
            concat_channels,
            kernel_size=1,
            bias=False,
        )
        self.proj_in = nn.Linear(concat_channels, config.hidden_size, bias=False)
        self.transformer_blocks = [
            FlowTransformerBlock(config) for _ in range(config.num_layers)
        ]
        self.proj_out = nn.Linear(config.hidden_size, config.in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(
            config.in_channels,
            config.in_channels,
            kernel_size=1,
            bias=False,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        condition: mx.array,
    ) -> mx.array:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, length, channels]")
        if condition.ndim != 3:
            raise ValueError("condition must have shape [batch, length, features]")
        if hidden_states.shape[:2] != condition.shape[:2]:
            raise ValueError("Latents and condition must share batch and length")
        if hidden_states.shape[-1] != self.config.in_channels:
            raise ValueError("Latent channel count does not match the transformer")
        if condition.shape[-1] != self.config.condition_dim:
            raise ValueError("Condition feature count does not match the transformer")
        if timestep.ndim == 0:
            timestep = mx.broadcast_to(timestep, (hidden_states.shape[0],))
        elif timestep.shape != (hidden_states.shape[0],):
            raise ValueError("timestep must be scalar or have one value per batch row")

        combined = mx.concatenate(
            (hidden_states, mx.zeros_like(hidden_states), condition), axis=-1
        )
        combined = self.preprocess_conv(combined) + combined
        hidden_states = self.proj_in(combined)
        time_embedding = self.time_embed(self.time_proj(timestep))
        hidden_states = mx.concatenate(
            (time_embedding[:, None, :], hidden_states), axis=1
        )
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.proj_out(hidden_states[:, 1:])
        return self.postprocess_conv(hidden_states) + hidden_states
