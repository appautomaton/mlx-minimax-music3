from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mlx_minimax_music3.acoustic import AcousticLatents, LatentChunk
from mlx_minimax_music3.chunking import chunk_windows
from mlx_minimax_music3.decoding import decode_latent_chunks


class RepeatVocoder:
    config = SimpleNamespace(sampling_rate=44_100)

    def __call__(self, latents: mx.array) -> mx.array:
        samples = mx.repeat(latents[..., :1], 512, axis=1)[..., 0]
        return mx.stack((samples, samples), axis=1)


def test_decode_crops_overlap_to_the_full_timeline() -> None:
    windows = chunk_windows(250)
    latent_lengths = (689, 516)
    chunks = tuple(
        LatentChunk(
            window=window,
            latents=mx.zeros((1, length, 4)),
        )
        for window, length in zip(windows, latent_lengths, strict=True)
    )

    waveform = decode_latent_chunks(
        RepeatVocoder(), AcousticLatents(chunks=chunks)
    )

    assert waveform.samples.shape == (2, 861 * 512)
    assert waveform.sample_rate == 44_100
