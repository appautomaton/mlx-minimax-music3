"""Chunked flow matching over Music 3 acoustic latents."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from .chunking import ChunkWindow, chunk_windows, overlap_latent_length
from .models.condition_encoder import ConditionEncoder
from .models.flow_transformer import FlowTransformer
from .sampling import derive_acoustic_seed

DEFAULT_FLOW_STEPS = 30
DEFAULT_FLOW_CFG_SCALE = 1.7


@dataclass(frozen=True, slots=True)
class FlowGenerationConfig:
    """Numerical controls for the reference-compatible Euler solve."""

    num_steps: int = DEFAULT_FLOW_STEPS
    cfg_scale: float = DEFAULT_FLOW_CFG_SCALE

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_steps, bool)
            or not isinstance(self.num_steps, int)
            or self.num_steps <= 0
        ):
            raise ValueError("num_steps must be a positive integer")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 0:
            raise ValueError("cfg_scale must be finite and non-negative")


DEFAULT_FLOW_CONFIG = FlowGenerationConfig()


@dataclass(frozen=True, slots=True)
class FlowProgress:
    chunk_index: int
    num_chunks: int
    step: int
    num_steps: int


@dataclass(frozen=True, slots=True)
class LatentChunk:
    """One evaluated channels-last latent window and its frame range."""

    window: ChunkWindow
    latents: mx.array


@dataclass(frozen=True, slots=True)
class AcousticLatents:
    """Evaluated latent windows ready for the separately resident vocoder."""

    chunks: tuple[LatentChunk, ...]

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


def solve_flow_chunk(
    transformer: FlowTransformer,
    condition: mx.array,
    *,
    key: mx.array,
    config: FlowGenerationConfig,
    previous_latent: mx.array | None = None,
    previous_condition: mx.array | None = None,
    progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Euler-solve one condition window and return its overlap carry state."""

    if condition.ndim != 3 or condition.shape[0] != 1:
        raise ValueError("condition must have shape [1, length, features]")
    if condition.shape[-1] != transformer.config.condition_dim:
        raise ValueError("Condition feature count does not match the transformer")
    condition = condition.astype(mx.float32)
    latent_shape = (
        1,
        condition.shape[1],
        transformer.config.in_channels,
    )
    latents = mx.random.normal(latent_shape, dtype=condition.dtype, key=key)

    overlap = 0
    latent_prompt = None
    noise_prompt = None
    if previous_latent is not None:
        if previous_latent.ndim != 3 or previous_latent.shape[0] != 1:
            raise ValueError("previous_latent must have shape [1, length, channels]")
        overlap = min(previous_latent.shape[1], condition.shape[1])
        if overlap > 0:
            if previous_condition is None:
                raise ValueError("previous_condition is required with previous_latent")
            if previous_condition.shape != (
                1,
                previous_latent.shape[1],
                condition.shape[-1],
            ):
                raise ValueError("Previous latent and condition carry are misaligned")
            latent_prompt = previous_latent[:, :overlap].astype(latents.dtype)
            noise_prompt = mx.contiguous(latents[:, :overlap])
            condition[:, :overlap] = previous_condition[:, :overlap].astype(
                condition.dtype
            )

    condition_cfg = mx.concatenate((condition, mx.zeros_like(condition)), axis=0)
    compute_dtype = transformer.proj_in.weight.dtype
    condition_cfg_compute = condition_cfg.astype(compute_dtype)
    step_size = 1.0 / config.num_steps
    for step in range(config.num_steps):
        if cancelled is not None and cancelled():
            raise InterruptedError("Music 3 flow generation was cancelled")
        time_value = step / config.num_steps
        if overlap and latent_prompt is not None and noise_prompt is not None:
            latents[:, :overlap] = (
                1.0 - (1.0 - 1e-6) * time_value
            ) * noise_prompt + time_value * latent_prompt
        latent_cfg = mx.broadcast_to(latents, (2, *latents.shape[1:])).astype(
            compute_dtype
        )
        timestep_cfg = mx.full((2,), time_value, dtype=compute_dtype)
        velocity = transformer(
            latent_cfg,
            timestep_cfg,
            condition_cfg_compute,
        ).astype(mx.float32)
        guided = (
            config.cfg_scale * velocity[:1]
            + (1.0 - config.cfg_scale) * velocity[1:2]
        )
        latents = latents + step_size * guided.astype(latents.dtype)
        mx.eval(latents)
        if progress is not None:
            progress(step + 1)

    if overlap and latent_prompt is not None:
        latents[:, :overlap] = latent_prompt
        mx.eval(latents)

    overlap_length = overlap_latent_length()
    carry_start = max(0, latents.shape[1] - 2 * overlap_length)
    carry_end = max(carry_start, latents.shape[1] - overlap_length)
    next_latent = mx.contiguous(latents[:, carry_start:carry_end])
    next_condition = mx.contiguous(condition[:, carry_start:carry_end])
    mx.eval(latents, next_latent, next_condition)
    return latents, next_latent, next_condition


def generate_acoustic_latents(
    transformer: FlowTransformer,
    condition_encoder: ConditionEncoder,
    frame_hiddens: mx.array,
    *,
    seed: int,
    config: FlowGenerationConfig = DEFAULT_FLOW_CONFIG,
    progress: Callable[[FlowProgress], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AcousticLatents:
    """Generate all overlapping latent windows from autoregressive states."""

    if frame_hiddens.ndim != 3 or frame_hiddens.shape[0] != 1:
        raise ValueError("frame_hiddens must have shape [1, frames, features]")
    windows = chunk_windows(frame_hiddens.shape[1])
    if not windows:
        raise ValueError("At least one autoregressive frame is required")

    chunks = []
    previous_latent = None
    previous_condition = None
    for window in windows:
        if cancelled is not None and cancelled():
            raise InterruptedError("Music 3 acoustic generation was cancelled")
        condition = condition_encoder(frame_hiddens[:, window.start : window.end])
        mx.eval(condition)

        def report(step: int, *, window_index: int = window.index) -> None:
            if progress is not None:
                progress(
                    FlowProgress(
                        chunk_index=window_index,
                        num_chunks=len(windows),
                        step=step,
                        num_steps=config.num_steps,
                    )
                )

        latents, previous_latent, previous_condition = solve_flow_chunk(
            transformer,
            condition,
            key=mx.random.key(derive_acoustic_seed(seed, "dit", window.index)),
            config=config,
            previous_latent=previous_latent,
            previous_condition=previous_condition,
            progress=report,
            cancelled=cancelled,
        )
        chunks.append(LatentChunk(window=window, latents=latents))
    return AcousticLatents(chunks=tuple(chunks))
