from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_minimax_music3.autoregressive import (
    AutoregressiveConfig,
    generate_autoregressive,
)
from mlx_minimax_music3.config import Qwen3Config, RVQDepthDecoderConfig
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM
from mlx_minimax_music3.models.rvq_depth import RVQDepthDecoder
from mlx_minimax_music3.tokenizer import TokenizedPrompt


def _argmax_sampler(
    logits: mx.array, *, top_k: int, seed: int, position: int
) -> mx.array:
    del top_k, seed, position
    # c0 column zero is the stop token in the narrowed reference vocabulary.
    return mx.array([1 if logits.shape[-1] == 16_385 else 0], dtype=mx.int32)


def test_tiny_autoregressive_loop_produces_aligned_frames() -> None:
    hidden_size = 16
    language_model = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=hidden_size,
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
            hidden_size=hidden_size,
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
    assert result.frame_hiddens.shape == (1, 2, 4 * hidden_size)
    assert not result.stopped_on_audio_end
    assert mx.array_equal(result.codes, mx.zeros_like(result.codes)).item()


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_autoregressive_config_rejects_seed_outside_uint64(seed: int) -> None:
    with pytest.raises(ValueError, match="64-bit"):
        AutoregressiveConfig(seed=seed)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_autoregressive_config_rejects_non_finite_controls(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        AutoregressiveConfig(audio_duration=value)
