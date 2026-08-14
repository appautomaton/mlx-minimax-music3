"""MLX implementation of the Qwen3 backbone used by MiniMax Music 3.

The module layout follows the released Transformers checkpoint so dense weights
load without name rewrites. The attention and cache structure is informed by the
MIT-licensed MLX-VLM Qwen3 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from ..config import Qwen3Config
from .cache import KVCache, causal_mask, make_kv_caches, validate_cache_sequence


@dataclass(frozen=True, slots=True)
class CausalLMOutput:
    """Language-model logits and the backbone states that produced them."""

    logits: mx.array
    last_hidden_state: mx.array


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(
            config.head_dim, traditional=False, base=config.rope_theta
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | None,
        cache: KVCache | None,
        offset: int,
    ) -> mx.array:
        batch_size, sequence_length, _ = hidden_states.shape

        queries = self.q_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )
        keys = self.k_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        values = self.v_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

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
            batch_size, sequence_length, -1
        )
        return self.o_proj(attended)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | None,
        cache: KVCache | None,
        offset: int,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            mask=mask,
            cache=cache,
            offset=offset,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        if hidden_states.ndim != 3:
            raise ValueError("Qwen3 inputs must have shape [batch, length, hidden]")

        offset = validate_cache_sequence(cache, num_layers=len(self.layers))
        sequence_length = hidden_states.shape[1]
        mask = causal_mask(sequence_length, offset)
        layer_caches = cache if cache is not None else [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, layer_caches, strict=True):
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache=layer_cache,
                offset=offset,
            )
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 backbone and language-model head with checkpoint-compatible names."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        cache: list[KVCache] | None = None,
    ) -> CausalLMOutput:
        hidden_states = self.model(
            input_ids, inputs_embeds=inputs_embeds, cache=cache
        )
        logits = (
            self.model.embed_tokens.as_linear(hidden_states)
            if self.config.tie_word_embeddings
            else self.lm_head(hidden_states)
        )
        return CausalLMOutput(
            logits=logits,
            last_hidden_state=hidden_states,
        )

    def make_cache(self, *, capacity: int | None = None) -> list[KVCache]:
        return make_kv_caches(
            self.config.num_hidden_layers,
            capacity=capacity,
        )
