"""Memory-bounded audio output for generated Music 3 waveforms."""

from __future__ import annotations

import os
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .decoding import Waveform


@dataclass(frozen=True, slots=True)
class AudioFile:
    """Metadata for one completely written PCM WAV file."""

    path: Path
    sample_rate: int
    num_channels: int
    num_samples: int


def _pcm16_bytes(samples: mx.array) -> bytes:
    interleaved = mx.round(mx.clip(samples, -1.0, 1.0) * 32_767.0)
    interleaved = interleaved.T.reshape(-1).astype(mx.int16)
    mx.eval(interleaved)
    encoded = array("h", interleaved.tolist())
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def write_pcm16_wav(
    waveform: Waveform,
    path: str | Path,
    *,
    chunk_samples: int = 65_536,
    overwrite: bool = False,
) -> AudioFile:
    """Atomically write channels-first floating-point audio as PCM16 WAV.

    Conversion is deliberately chunked so a long waveform does not require a
    second full-size Python or PCM copy in memory.
    """

    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    if waveform.num_channels > 65_535:
        raise ValueError("WAV cannot represent more than 65,535 channels")
    if not mx.isfinite(waveform.samples).all().item():
        raise ValueError("Waveform contains non-finite samples")

    path = Path(path).resolve()
    if path.suffix.lower() != ".wav":
        raise ValueError("PCM output path must end in .wav")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing audio: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(waveform.num_channels)
            output.setsampwidth(2)
            output.setframerate(waveform.sample_rate)
            for start in range(0, waveform.num_samples, chunk_samples):
                stop = min(start + chunk_samples, waveform.num_samples)
                output.writeframesraw(_pcm16_bytes(waveform.samples[:, start:stop]))
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return AudioFile(
        path=path,
        sample_rate=waveform.sample_rate,
        num_channels=waveform.num_channels,
        num_samples=waveform.num_samples,
    )


__all__ = ["AudioFile", "write_pcm16_wav"]
