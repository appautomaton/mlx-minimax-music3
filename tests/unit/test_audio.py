from __future__ import annotations

import wave
from array import array
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_minimax_music3.audio import write_pcm16_wav
from mlx_minimax_music3.decoding import Waveform


def test_write_pcm16_wav_is_interleaved_and_chunked(tmp_path: Path) -> None:
    waveform = Waveform(
        samples=mx.array(
            [
                [-1.0, 0.0, 1.0],
                [0.5, -0.5, 0.25],
            ]
        ),
        sample_rate=44_100,
    )
    path = tmp_path / "nested/output.wav"

    result = write_pcm16_wav(waveform, path, chunk_samples=2)

    with wave.open(str(path), "rb") as audio:
        assert audio.getnchannels() == 2
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 44_100
        assert audio.getnframes() == 3
        encoded = array("h")
        encoded.frombytes(audio.readframes(3))
    assert encoded.tolist() == [-32767, 16384, 0, -16384, 32767, 8192]
    assert result.path == path.resolve()
    assert result.num_samples == 3


def test_write_pcm16_wav_refuses_implicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "output.wav"
    path.write_bytes(b"existing")
    waveform = Waveform(mx.zeros((2, 1)), sample_rate=44_100)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_pcm16_wav(waveform, path)

    assert path.read_bytes() == b"existing"


def test_write_pcm16_wav_rejects_non_finite_audio(tmp_path: Path) -> None:
    waveform = Waveform(mx.array([[float("nan")]]), sample_rate=44_100)

    with pytest.raises(ValueError, match="non-finite"):
        write_pcm16_wav(waveform, tmp_path / "output.wav")
