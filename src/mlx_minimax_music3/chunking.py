"""Chunk boundaries and overlap cropping for MiniMax Music 3."""

from __future__ import annotations

from dataclasses import dataclass

AR_CHUNK_FRAMES = 200
AR_CHUNK_HOP_FRAMES = 100
LATENT_HOP_SAMPLES = 512
AR_HOP_LATENT_FRAMES = 344
BLEND_LATENT_FRAMES = AR_HOP_LATENT_FRAMES // 4


@dataclass(frozen=True, slots=True)
class ChunkWindow:
    """One overlapping window of autoregressive frame hidden states."""

    index: int
    start: int
    end: int
    is_first: bool
    is_last: bool

    @property
    def length(self) -> int:
        return self.end - self.start


def chunk_windows(frames: int) -> tuple[ChunkWindow, ...]:
    """Split frame hidden states into 200-frame windows with a 100-frame hop."""

    if isinstance(frames, bool) or not isinstance(frames, int):
        raise TypeError("frames must be an integer")
    if frames < 0:
        raise ValueError("frames must not be negative")
    if frames == 0:
        return ()
    if frames <= AR_CHUNK_FRAMES:
        return (ChunkWindow(0, 0, frames, True, True),)

    windows = []
    index = 0
    start = 0
    while start < frames:
        end = min(start + AR_CHUNK_FRAMES, frames)
        windows.append(
            ChunkWindow(
                index=index,
                start=start,
                end=end,
                is_first=index == 0,
                is_last=end >= frames,
            )
        )
        if end >= frames:
            break
        index += 1
        start += AR_CHUNK_HOP_FRAMES
    return tuple(windows)


def overlap_latent_length() -> int:
    """Return the latent-frame overlap passed between neighboring windows."""

    return AR_HOP_LATENT_FRAMES // 2


def crop_sample_bounds(window: ChunkWindow) -> tuple[int, int]:
    """Return the left and right waveform samples trimmed from a decoded window."""

    left = 0 if window.is_first else BLEND_LATENT_FRAMES * LATENT_HOP_SAMPLES
    right = (
        0
        if window.is_last
        else (AR_HOP_LATENT_FRAMES - BLEND_LATENT_FRAMES) * LATENT_HOP_SAMPLES
    )
    return left, right


__all__ = [
    "AR_CHUNK_FRAMES",
    "AR_CHUNK_HOP_FRAMES",
    "ChunkWindow",
    "chunk_windows",
    "crop_sample_bounds",
    "overlap_latent_length",
]
