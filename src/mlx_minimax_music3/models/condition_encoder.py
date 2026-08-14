# Copyright 2026 The MiniMax Team and The HuggingFace Team.
# Adapted from Hugging Face Diffusers under the Apache License 2.0.
"""MLX acoustic condition encoder for MiniMax Music 3."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from ..config import ConditionEncoderConfig


class ConditionEncoder(nn.Module):
    """Mix autoregressive states and align them to the latent timeline."""

    def __init__(self, config: ConditionEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.layer_weight_logits = mx.zeros((config.num_condition_layers,))
        self.layer_scale = mx.ones((1,))
        self.proj = nn.Conv1d(
            config.condition_hidden_dim,
            config.out_dim,
            kernel_size=3,
            padding=1,
        )

    def latent_length(self, num_frames: int) -> int:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        return max(
            1,
            int(
                num_frames
                * self.config.output_sampling_rate
                / self.config.input_sampling_rate
                * self.config.input_hop_length
                / self.config.output_hop_length
            ),
        )

    def __call__(self, frame_hiddens: mx.array) -> mx.array:
        if frame_hiddens.ndim != 3:
            raise ValueError(
                "frame_hiddens must have shape [batch, frames, layers * hidden]"
            )
        batch_size, num_frames, features = frame_hiddens.shape
        expected = (
            self.config.num_condition_layers * self.config.condition_hidden_dim
        )
        if features != expected:
            raise ValueError(
                f"Expected {expected} conditioning features, got {features}"
            )

        hidden_states = frame_hiddens.astype(self.proj.weight.dtype).reshape(
            batch_size,
            num_frames,
            self.config.num_condition_layers,
            self.config.condition_hidden_dim,
        )
        layer_weights = mx.softmax(
            self.layer_weight_logits.astype(mx.float32), axis=0
        ).astype(hidden_states.dtype)
        hidden_states = mx.sum(
            hidden_states * layer_weights[None, None, :, None], axis=2
        )
        hidden_states = self.layer_scale.astype(hidden_states.dtype) * hidden_states
        hidden_states = self.proj(hidden_states)

        output_length = self.latent_length(num_frames)
        indices = mx.floor(
            mx.arange(output_length, dtype=mx.float32)
            * (num_frames / output_length)
        ).astype(mx.int32)
        return hidden_states[:, indices, :]
