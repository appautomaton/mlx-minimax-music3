from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_minimax_music3.config import Qwen3Config, RVQDepthDecoderConfig
from mlx_minimax_music3.models.cache import CacheCapacityError, KVCache
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM
from mlx_minimax_music3.models.rvq_depth import RVQDepthDecoder


def _qwen_config() -> Qwen3Config:
    return Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        max_position_embeddings=32,
        rope_theta=10_000,
        published_dtype="float32",
    )


def _rvq_config() -> RVQDepthDecoderConfig:
    return RVQDepthDecoderConfig(
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_attention_heads=4,
        audio_vocab_size=32,
        num_codebooks=4,
        max_position_embeddings=8,
    )


def test_kv_cache_preallocates_and_enforces_capacity() -> None:
    cache = KVCache(capacity=3)
    keys = mx.ones((1, 2, 2, 4))
    values = mx.zeros((1, 2, 2, 4))

    cached_keys, cached_values = cache.update_and_fetch(keys, values)
    cache.update_and_fetch(keys[:, :, :1], values[:, :, :1])

    assert cached_keys.shape == (1, 2, 2, 4)
    assert cached_values.shape == (1, 2, 2, 4)
    assert cache.capacity == 3
    assert cache.offset == 3
    with pytest.raises(CacheCapacityError, match="planned for 3"):
        cache.update_and_fetch(keys[:, :, :1], values[:, :, :1])


def test_qwen_cached_decode_matches_full_sequence() -> None:
    mx.random.seed(7)
    model = Qwen3ForCausalLM(_qwen_config())
    tokens = mx.array([[1, 2, 3, 4]])

    full = model(tokens).logits[:, -1]
    cache = model.make_cache(capacity=tokens.shape[1])
    prefill = model(tokens[:, :3], cache=cache)
    mx.eval(prefill.logits)
    cached = model(tokens[:, 3:], cache=cache).logits[:, -1]
    mx.eval(full, cached)

    # Prefill and single-token decode use different fused attention kernels.
    assert mx.allclose(full, cached, rtol=2e-3, atol=2e-3).item()
    assert all(layer_cache.offset == 4 for layer_cache in cache)


def test_qwen_rejects_misaligned_layer_caches() -> None:
    model = Qwen3ForCausalLM(_qwen_config())
    caches = model.make_cache(capacity=4)
    keys = mx.zeros((1, 2, 1, 4))
    caches[0].update_and_fetch(keys, keys)

    with pytest.raises(ValueError, match="same sequence offset"):
        model(mx.array([[1]]), cache=caches)


def test_rvq_cached_decode_matches_full_sequence() -> None:
    mx.random.seed(11)
    model = RVQDepthDecoder(_rvq_config())
    inputs = mx.random.normal((2, 4, 16))

    full = model(inputs)[:, -1]
    cache = model.make_cache()
    prefill = model(inputs[:, :3], cache=cache)
    mx.eval(prefill)
    cached = model(inputs[:, 3:], cache=cache)[:, -1]
    mx.eval(full, cached)

    assert mx.allclose(full, cached, rtol=1e-3, atol=1e-3).item()


def test_rvq_codebook_offsets_do_not_alias() -> None:
    model = RVQDepthDecoder(_rvq_config())
    code = mx.array([3])

    first = model.embed_residual_code(code, codebook_index=1)
    second = model.embed_residual_code(code, codebook_index=2)
    mx.eval(first, second)

    assert not mx.array_equal(first, second).item()
