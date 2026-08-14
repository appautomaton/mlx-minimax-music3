# Copyright 2026 The MiniMax Team and The HuggingFace Team.
# Adapted from Hugging Face Diffusers under the Apache License 2.0.
"""MLX Flow-VAE/DAC-style waveform decoder for MiniMax Music 3."""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from ..config import VocoderConfig


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # Keep the published [1, channels, 1] shape in the module tree.
        self.alpha = mx.ones((1, channels, 1))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        alpha = self.alpha.transpose(0, 2, 1).astype(hidden_states.dtype)
        return hidden_states + mx.square(mx.sin(alpha * hidden_states)) / (
            alpha + 1e-9
        )


class ConvTranspose1d(nn.Module):
    """Small channels-last transposed convolution module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        scale = math.sqrt(1 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels),
        )
        self.bias = mx.zeros((out_channels,))
        self.stride = stride
        self.padding = padding

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return (
            mx.conv_transpose1d(
                hidden_states,
                self.weight,
                stride=self.stride,
                padding=self.padding,
            )
            + self.bias
        )


class VocoderResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.snake1 = Snake1d(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            dilation=dilation,
            padding=3 * dilation,
        )
        self.snake2 = Snake1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = self.conv1(self.snake1(hidden_states))
        residual = self.conv2(self.snake2(residual))
        return hidden_states + residual


class VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int) -> None:
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.res_unit1 = VocoderResidualUnit(output_dim, dilation=1)
        self.res_unit2 = VocoderResidualUnit(output_dim, dilation=3)
        self.res_unit3 = VocoderResidualUnit(output_dim, dilation=9)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.conv_t1(self.snake1(hidden_states))
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        return self.res_unit3(hidden_states)


class Vocoder(nn.Module):
    """Decode channels-last Flow-VAE latents to stereo 44.1 kHz audio."""

    def __init__(self, config: VocoderConfig) -> None:
        super().__init__()
        if config.latent_channels % 2:
            raise ValueError("latent_channels must split evenly into stereo streams")
        self.config = config
        self.dec_in_proj = nn.Conv1d(
            config.latent_channels // 2,
            config.decoder_input_dim,
            kernel_size=1,
        )
        self.conv_in = nn.Conv1d(
            config.decoder_input_dim,
            config.decoder_hidden_dim,
            kernel_size=7,
            padding=3,
        )
        self.blocks = []
        output_dim = config.decoder_hidden_dim
        for index, stride in enumerate(config.upsampling_ratios):
            input_dim = config.decoder_hidden_dim // (2**index)
            output_dim = config.decoder_hidden_dim // (2 ** (index + 1))
            self.blocks.append(VocoderBlock(input_dim, output_dim, stride))
        self.snake_out = Snake1d(output_dim)
        self.conv_out = nn.Conv1d(output_dim, 1, kernel_size=7, padding=3)

    def __call__(self, latents: mx.array) -> mx.array:
        if latents.ndim != 3:
            raise ValueError("latents must have shape [batch, length, channels]")
        if latents.shape[-1] != self.config.latent_channels:
            raise ValueError("Latent channel count does not match the vocoder")
        batch_size, length, _ = latents.shape
        hidden_states = latents.reshape(
            batch_size, length, 2, self.config.latent_channels // 2
        )
        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(
            batch_size * 2,
            length,
            self.config.latent_channels // 2,
        )
        hidden_states = self.conv_in(self.dec_in_proj(hidden_states))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        waveform = mx.tanh(self.conv_out(self.snake_out(hidden_states)))[..., 0]
        return waveform.reshape(batch_size, 2, -1)
