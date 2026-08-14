"""Waveform decoding and overlap cropping for latent windows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from .acoustic import AcousticLatents
from .chunking import crop_sample_bounds
from .models.vocoder import Vocoder


@dataclass(frozen=True, slots=True)
class Waveform:
    """Evaluated channels-first floating-point audio."""

    samples: mx.array
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError("Waveform samples must have shape [channels, samples]")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def num_channels(self) -> int:
        return self.samples.shape[0]

    @property
    def num_samples(self) -> int:
        return self.samples.shape[1]


def decode_latent_chunks(
    vocoder: Vocoder,
    acoustic: AcousticLatents,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Waveform:
    """Decode, crop, and concatenate all latent windows at native 44.1 kHz."""

    if not acoustic.chunks:
        raise ValueError("At least one latent chunk is required")
    waveform_chunks = []
    total = len(acoustic.chunks)
    for index, chunk in enumerate(acoustic.chunks):
        if cancelled is not None and cancelled():
            raise InterruptedError("Music 3 waveform decoding was cancelled")
        waveform = vocoder(chunk.latents)
        left, right = crop_sample_bounds(chunk.window)
        if left + right >= waveform.shape[-1]:
            raise ValueError(
                f"Waveform crop removes all samples from chunk {chunk.window.index}"
            )
        stop = waveform.shape[-1] - right if right else waveform.shape[-1]
        cropped = mx.contiguous(waveform[0, :, left:stop].astype(mx.float32))
        mx.eval(cropped)
        waveform_chunks.append(cropped)
        if progress is not None:
            progress(index + 1, total)

    samples = (
        waveform_chunks[0]
        if len(waveform_chunks) == 1
        else mx.concatenate(waveform_chunks, axis=-1)
    )
    samples = mx.clip(samples, -1.0, 1.0)
    mx.eval(samples)
    return Waveform(samples=samples, sample_rate=vocoder.config.sampling_rate)
