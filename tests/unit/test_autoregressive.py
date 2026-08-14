from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_minimax_music3.autoregressive import (
    AutoregressiveConfig,
    _restricted_semantic_logits,
    generate_autoregressive,
)
from mlx_minimax_music3.config import Qwen3Config, RVQDepthDecoderConfig
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM
from mlx_minimax_music3.models.rvq_depth import RVQDepthDecoder
from mlx_minimax_music3.prompting import (
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    SEMANTIC_VOCAB_SIZE,
)
from mlx_minimax_music3.tokenizer import TokenizedPrompt

_HIDDEN_SIZE = 16


def _argmax_sampler(
    logits: mx.array, *, top_k: int, seed: int, position: int
) -> mx.array:
    del top_k, seed, position
    # c0 column zero is the stop token in the narrowed reference vocabulary.
    return mx.array([1 if logits.shape[-1] == 16_385 else 0], dtype=mx.int32)


def _literal_argmax_sampler(
    logits: mx.array, *, top_k: int, seed: int, position: int
) -> mx.array:
    del top_k, seed, position
    return mx.argmax(logits, axis=-1).astype(mx.int32)


def _tiny_models() -> tuple[Qwen3ForCausalLM, RVQDepthDecoder]:
    language_model = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=170_000,
            max_position_embeddings=16,
            published_dtype="float32",
        )
    )
    decoder = RVQDepthDecoder(
        RVQDepthDecoderConfig(
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=32,
            num_layers=1,
            num_attention_heads=4,
            audio_vocab_size=32,
            num_codebooks=4,
            max_position_embeddings=8,
        )
    )
    language_model.lm_head.weight = mx.zeros_like(language_model.lm_head.weight)
    for head in decoder.audio_heads:
        head.weight = mx.zeros_like(head.weight)
    return language_model, decoder


def test_tiny_autoregressive_loop_produces_aligned_frames() -> None:
    language_model, decoder = _tiny_models()
    prompt = TokenizedPrompt(
        conditional=(1, 2, 3),
        unconditional=(1, 4, 3),
    )

    result = generate_autoregressive(
        language_model,
        decoder,
        prompt,
        AutoregressiveConfig(audio_duration=2 / 25, buffer_flush_interval=1),
        sampler=_argmax_sampler,
    )
    mx.eval(result.codes, result.frame_hiddens)

    assert result.codes.shape == (1, 2, 4)
    assert result.frame_hiddens.shape == (1, 2, 4 * _HIDDEN_SIZE)
    assert not result.stopped_on_audio_end
    assert mx.array_equal(result.codes, mx.zeros_like(result.codes)).item()


def test_minimum_duration_masks_early_stop_until_required_frames() -> None:
    language_model, decoder = _tiny_models()
    prompt = TokenizedPrompt(
        conditional=(1, 2, 3),
        unconditional=(1, 4, 3),
    )

    result = generate_autoregressive(
        language_model,
        decoder,
        prompt,
        AutoregressiveConfig(
            audio_duration=4 / 25,
            min_audio_duration=2 / 25,
            buffer_flush_interval=1,
        ),
        sampler=_literal_argmax_sampler,
    )
    mx.eval(result.codes, result.frame_hiddens)

    assert result.codes.shape == (1, 2, 4)
    assert result.stopped_on_audio_end


def test_restricted_semantic_head_matches_full_projection() -> None:
    language_model, _ = _tiny_models()
    weight = language_model.lm_head.weight
    rows = mx.arange(weight.shape[0], dtype=mx.float32)[:, None]
    columns = mx.arange(weight.shape[1], dtype=mx.float32)[None, :]
    language_model.lm_head.weight = mx.sin(rows * 0.001 + columns * 0.01).astype(
        mx.bfloat16
    )
    hidden = (mx.arange(32, dtype=mx.float32).reshape(2, 16) / 32).astype(mx.bfloat16)
    full = language_model.lm_head(hidden).astype(mx.float32)
    ids = mx.concatenate(
        (
            mx.array([AUDIO_END_TOKEN_ID], dtype=mx.int32),
            mx.arange(
                AUDIO_CODE_OFFSET,
                AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE,
                dtype=mx.int32,
            ),
        )
    )
    expected = full[:, ids]
    actual = _restricted_semantic_logits(
        language_model,
        hidden,
        allow_stop=True,
    )
    mx.eval(expected, actual)

    assert mx.allclose(actual, expected, rtol=0, atol=0).item()


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_autoregressive_config_rejects_seed_outside_uint64(seed: int) -> None:
    with pytest.raises(ValueError, match="64-bit"):
        AutoregressiveConfig(seed=seed)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_autoregressive_config_rejects_non_finite_controls(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        AutoregressiveConfig(audio_duration=value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_autoregressive_config_rejects_invalid_minimum_duration(value: float) -> None:
    with pytest.raises(ValueError, match="min_audio_duration"):
        AutoregressiveConfig(min_audio_duration=value)


def test_autoregressive_config_rejects_minimum_above_ceiling() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        AutoregressiveConfig(audio_duration=1.0, min_audio_duration=1.1)
