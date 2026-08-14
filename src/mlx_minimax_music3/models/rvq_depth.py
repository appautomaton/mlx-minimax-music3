# Copyright 2026 The MiniMax Team and The HuggingFace Team.
# Portions adapted from Hugging Face Diffusers under the Apache License 2.0.
"""MLX residual vector quantization depth decoder for MiniMax Music 3."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from ..config import RVQDepthDecoderConfig
from .cache import KVCache, causal_mask, make_kv_caches, validate_cache_sequence


class RVQAttention(nn.Module):
    def __init__(self, config: RVQDepthDecoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | None,
        cache: KVCache | None,
    ) -> mx.array:
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
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, hidden_size
        )
        return self.to_out(attended)


class RVQDecoderBlock(nn.Module):
    def __init__(self, config: RVQDepthDecoderConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn = RVQAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | None,
        cache: KVCache | None,
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.input_layernorm(hidden_states), mask=mask, cache=cache
        )
        normalized = self.post_attention_layernorm(hidden_states)
        gated = nn.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        return hidden_states + self.down_proj(gated)


class RVQDepthDecoder(nn.Module):
    """Predict the seven residual codebooks within one generated audio frame."""

    def __init__(self, config: RVQDepthDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.num_residual_codebooks,
            config.hidden_size,
        )
        self.projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layers = [RVQDecoderBlock(config) for _ in range(config.num_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.audio_heads = [
            nn.Linear(config.hidden_size, config.audio_vocab_size, bias=False)
            for _ in range(config.num_residual_codebooks)
        ]

    def __call__(
        self,
        inputs_embeds: mx.array,
        *,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        if inputs_embeds.ndim != 3:
            raise ValueError("RVQ inputs must have shape [batch, length, hidden]")
        if inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError("RVQ input hidden dimension does not match its config")
        offset = validate_cache_sequence(cache, num_layers=len(self.layers))
        sequence_length = inputs_embeds.shape[1]
        end = offset + sequence_length
        if end > self.config.max_position_embeddings:
            raise ValueError(
                "RVQ sequence exceeds max_position_embeddings: "
                f"{end} > {self.config.max_position_embeddings}"
            )

        positions = mx.arange(offset, end)
        hidden_states = inputs_embeds + self.pos_embedding(positions)[None, :, :]
        mask = causal_mask(sequence_length, offset)
        layer_caches = cache if cache is not None else [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, layer_caches, strict=True):
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache=layer_cache,
            )
        return self.norm(hidden_states)

    def embed_residual_code(
        self, code: mx.array, *, codebook_index: int
    ) -> mx.array:
        """Embed c1..c7 using the checkpoint's concatenated embedding table."""

        if not 1 <= codebook_index < self.config.num_codebooks:
            raise ValueError(
                f"codebook_index must be in [1, {self.config.num_codebooks - 1}]"
            )
        offset = (codebook_index - 1) * self.config.audio_vocab_size
        return self.audio_embeddings(code + offset)

    def logits(self, hidden_states: mx.array, *, codebook_index: int) -> mx.array:
        """Project a decoder state onto one residual codebook vocabulary."""

        if not 1 <= codebook_index < self.config.num_codebooks:
            raise ValueError(
                f"codebook_index must be in [1, {self.config.num_codebooks - 1}]"
            )
        return self.audio_heads[codebook_index - 1](hidden_states)

    def make_cache(self) -> list[KVCache]:
        return make_kv_caches(
            self.config.num_layers,
            capacity=self.config.num_codebooks,
        )
