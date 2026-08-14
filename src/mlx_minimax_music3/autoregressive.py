"""Global and depth autoregressive generation for MiniMax Music 3."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from .models.qwen3 import Qwen3ForCausalLM
from .models.rvq_depth import RVQDepthDecoder
from .prompting import (
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    MAX_AUDIO_FRAMES,
    SEMANTIC_VOCAB_SIZE,
)
from .sampling import (
    SeedSchedule,
    classifier_free_guidance,
    sample_top_k,
)
from .tokenizer import TokenizedPrompt

FRAME_RATE = 25
AR_CFG_SCALE = 1.5
AR_TOP_K = 50


class Sampler(Protocol):
    def __call__(
        self,
        logits: mx.array,
        *,
        top_k: int,
        seed: int,
        position: int,
    ) -> mx.array: ...


@dataclass(frozen=True, slots=True)
class AutoregressiveConfig:
    """Request-local controls for the fixed Music 3 generation recipe."""

    audio_duration: float = 60.0
    seed: int = 0
    cfg_scale: float = AR_CFG_SCALE
    top_k: int = AR_TOP_K
    frame_rate: int = FRAME_RATE
    buffer_flush_interval: int = 32

    def __post_init__(self) -> None:
        if not math.isfinite(self.audio_duration) or self.audio_duration <= 0:
            raise ValueError("audio_duration must be finite and positive")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 0:
            raise ValueError("cfg_scale must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must be a non-negative 64-bit integer")
        for name in ("top_k", "frame_rate", "buffer_flush_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def max_frames(self) -> int:
        frames = min(int(self.audio_duration * self.frame_rate), MAX_AUDIO_FRAMES)
        if frames == 0:
            raise ValueError(
                "audio_duration is shorter than one autoregressive frame"
            )
        return frames


@dataclass(frozen=True, slots=True)
class GenerationProgress:
    """Progress emitted only at safe completed-frame boundaries."""

    completed_frames: int
    maximum_frames: int


@dataclass(frozen=True, slots=True)
class AutoregressiveResult:
    """Evaluated codes and acoustic-conditioning hidden states."""

    codes: mx.array
    frame_hiddens: mx.array
    stopped_on_audio_end: bool

    @property
    def num_frames(self) -> int:
        return self.frame_hiddens.shape[1]


class _FrameBuffer:
    """Preallocated, periodically evaluated durable stage output."""

    def __init__(
        self,
        *,
        max_frames: int,
        num_codebooks: int,
        hidden_size: int,
        dtype: mx.Dtype,
        flush_interval: int,
    ) -> None:
        self._hiddens = mx.zeros(
            (1, max_frames, num_codebooks * hidden_size), dtype=dtype
        )
        self._codes = mx.zeros((1, max_frames, num_codebooks), dtype=mx.int32)
        self._max_frames = max_frames
        self._flush_interval = flush_interval
        self.count = 0

    def append(self, codes: mx.array, hidden_states: mx.array) -> None:
        if self.count >= self._max_frames:
            raise RuntimeError("Frame buffer is full")
        self._codes[:, self.count, :] = codes
        self._hiddens[:, self.count, :] = hidden_states
        self.count += 1
        if self.count % self._flush_interval == 0:
            mx.eval(self._codes, self._hiddens)

    def finish(self) -> tuple[mx.array, mx.array]:
        if self.count == 0:
            raise ValueError(
                "MiniMax Music 3 generated zero frames; the prompt ended immediately"
            )
        mx.eval(self._codes, self._hiddens)
        if self.count == self._max_frames:
            return self._codes, self._hiddens
        codes = mx.contiguous(self._codes[:, : self.count])
        hiddens = mx.contiguous(self._hiddens[:, : self.count])
        mx.eval(codes, hiddens)
        return codes, hiddens


def _sample_semantic_code(
    language_model: Qwen3ForCausalLM,
    last_hidden: mx.array,
    *,
    cfg_scale: float,
    top_k: int,
    seed: int,
    position: int,
    sampler: Sampler,
) -> mx.array:
    logits = language_model.lm_head(last_hidden).astype(mx.float32)
    semantic_ids = mx.arange(
        AUDIO_CODE_OFFSET,
        AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE,
        dtype=mx.int32,
    )
    # SGLang hashes the narrowed vocabulary column during sampling. Preserve its
    # exact order: stop first, followed by c0 codes 0 through 16,383.
    allowed_ids = mx.concatenate(
        (mx.array([AUDIO_END_TOKEN_ID], dtype=mx.int32), semantic_ids)
    )
    allowed_logits = logits[:, allowed_ids]
    conditional = allowed_logits[:1]
    guided = classifier_free_guidance(allowed_logits, scale=cfg_scale)
    guided = mx.where(
        conditional
        < mx.topk(
            conditional,
            k=min(top_k, conditional.shape[-1]),
            axis=-1,
        )[..., -1:],
        -mx.inf,
        guided,
    )
    local_index = sampler(
        guided,
        top_k=min(top_k, guided.shape[-1]),
        seed=seed,
        position=position,
    )
    return allowed_ids[local_index]


def _generate_depth_codes(
    language_model: Qwen3ForCausalLM,
    decoder: RVQDepthDecoder,
    last_hidden: mx.array,
    semantic_token: mx.array,
    *,
    frame_index: int,
    cfg_scale: float,
    top_k: int,
    seeds: SeedSchedule,
    sampler: Sampler,
) -> tuple[mx.array, mx.array]:
    semantic_code = semantic_token - AUDIO_CODE_OFFSET
    paired_semantic = mx.repeat(semantic_code, 2, axis=0)
    semantic_embedding = language_model.model.embed_tokens(
        paired_semantic + AUDIO_CODE_OFFSET
    )
    first_inputs = mx.stack(
        (
            decoder.projection(last_hidden),
            decoder.projection(semantic_embedding),
        ),
        axis=1,
    )
    cache = decoder.make_cache()
    hidden = decoder(first_inputs, cache=cache)[:, -1]

    sampled_codes = [semantic_code]
    hidden_parts = []
    for codebook_index in range(1, decoder.config.num_codebooks):
        hidden_parts.append(hidden[:1])
        logits = decoder.logits(hidden, codebook_index=codebook_index)
        guided = classifier_free_guidance(logits, scale=cfg_scale)
        sampled = sampler(
            guided,
            top_k=min(top_k, decoder.config.audio_vocab_size),
            seed=seeds.sampling_seed,
            position=seeds.position(
                frame_index=frame_index,
                codebook_index=codebook_index,
            ),
        )
        sampled_codes.append(sampled)
        if codebook_index < decoder.config.num_codebooks - 1:
            paired = mx.repeat(sampled, 2, axis=0)
            embedding = decoder.embed_residual_code(
                paired, codebook_index=codebook_index
            )
            projected = decoder.projection(embedding)[:, None, :]
            hidden = decoder(projected, cache=cache)[:, -1]

    codes = mx.stack(sampled_codes, axis=-1)
    depth_hiddens = mx.concatenate(hidden_parts, axis=-1)
    return codes, depth_hiddens


def _embed_audio_frame(
    language_model: Qwen3ForCausalLM,
    decoder: RVQDepthDecoder,
    codes: mx.array,
) -> mx.array:
    paired_codes = mx.repeat(codes, 2, axis=0)
    semantic = language_model.model.embed_tokens(
        paired_codes[:, :1] + AUDIO_CODE_OFFSET
    )
    offsets = (
        mx.arange(decoder.config.num_residual_codebooks, dtype=mx.int32)
        * decoder.config.audio_vocab_size
    )[None, :]
    residual = decoder.audio_embeddings(paired_codes[:, 1:] + offsets)
    residual = residual.sum(axis=1, keepdims=True)
    return (semantic + residual.astype(semantic.dtype)) * (
        decoder.config.num_codebooks**-0.5
    )


def generate_autoregressive(
    language_model: Qwen3ForCausalLM,
    decoder: RVQDepthDecoder,
    prompt: TokenizedPrompt,
    config: AutoregressiveConfig,
    *,
    sampler: Sampler = sample_top_k,
    progress: Callable[[GenerationProgress], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AutoregressiveResult:
    """Generate Music 3 frame codes and hidden-state conditioning."""

    if language_model.config.hidden_size != decoder.config.hidden_size:
        raise ValueError("Language model and RVQ decoder hidden sizes must match")
    if language_model.config.vocab_size < AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE:
        raise ValueError("Language-model vocabulary cannot represent Music 3 codes")

    max_frames = config.max_frames
    text_ids = mx.array(prompt.rows(), dtype=mx.int32)
    cache = language_model.make_cache(capacity=prompt.length + max_frames)
    last_hidden = language_model.model(text_ids, cache=cache)[:, -1]
    mx.eval(last_hidden)

    seeds = SeedSchedule(config.seed, decoder.config.num_codebooks)
    frames = _FrameBuffer(
        max_frames=max_frames,
        num_codebooks=decoder.config.num_codebooks,
        hidden_size=decoder.config.hidden_size,
        dtype=last_hidden.dtype,
        flush_interval=config.buffer_flush_interval,
    )
    stopped_on_audio_end = False

    # Frame zero advances past <|audio_start|>; it is feedback, not output.
    for frame_index in range(max_frames + 1):
        if cancelled is not None and cancelled():
            raise InterruptedError("Music 3 autoregressive generation was cancelled")
        semantic_token = _sample_semantic_code(
            language_model,
            last_hidden,
            cfg_scale=config.cfg_scale,
            top_k=config.top_k,
            seed=seeds.sampling_seed,
            position=seeds.position(frame_index=frame_index, codebook_index=0),
            sampler=sampler,
        )
        if int(semantic_token.item()) == AUDIO_END_TOKEN_ID:
            stopped_on_audio_end = True
            break

        codes, depth_hiddens = _generate_depth_codes(
            language_model,
            decoder,
            last_hidden,
            semantic_token,
            frame_index=frame_index,
            cfg_scale=config.cfg_scale,
            top_k=config.top_k,
            seeds=seeds,
            sampler=sampler,
        )
        if frame_index > 0:
            frame_hidden = mx.concatenate((last_hidden[:1], depth_hiddens), axis=-1)
            frames.append(codes, frame_hidden)
            if progress is not None:
                progress(GenerationProgress(frames.count, max_frames))
            if frames.count >= max_frames:
                break

        feedback = _embed_audio_frame(language_model, decoder, codes)
        last_hidden = language_model.model(
            inputs_embeds=feedback,
            cache=cache,
        )[:, -1]

    codes, frame_hiddens = frames.finish()
    return AutoregressiveResult(
        codes=codes,
        frame_hiddens=frame_hiddens,
        stopped_on_audio_end=stopped_on_audio_end,
    )
