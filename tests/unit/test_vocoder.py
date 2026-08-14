from __future__ import annotations

import mlx.core as mx

from mlx_minimax_music3.config import VocoderConfig
from mlx_minimax_music3.models.vocoder import ConvTranspose1d, Vocoder


def test_transposed_convolution_expands_by_stride() -> None:
    layer = ConvTranspose1d(
        in_channels=3,
        out_channels=2,
        kernel_size=4,
        stride=2,
        padding=1,
    )
    output = layer(mx.zeros((1, 5, 3)))
    mx.eval(output)

    assert output.shape == (1, 10, 2)


def test_tiny_vocoder_decodes_stereo_at_total_ratio() -> None:
    config = VocoderConfig(
        latent_channels=8,
        decoder_input_dim=12,
        decoder_hidden_dim=16,
        upsampling_ratios=(2, 2),
        sampling_rate=44_100,
    )
    model = Vocoder(config)
    latents = mx.random.normal((1, 6, 8))

    waveform = model(latents)
    mx.eval(waveform)

    assert waveform.shape == (1, 2, 24)
    assert mx.isfinite(waveform).all().item()
    assert (mx.abs(waveform) <= 1.0).all().item()
