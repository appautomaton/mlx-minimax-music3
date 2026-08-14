"""Public package boundary tests."""

from mlx_minimax_music3 import (
    ExperimentalQuantizationWarning,
    GenerationRequest,
    GenerationResult,
    Music3Pipeline,
    PromptQualityWarning,
    __version__,
    instrumental_lyrics,
)


def test_package_version() -> None:
    assert __version__ == "0.0.1a0"


def test_generation_api_is_public() -> None:
    assert ExperimentalQuantizationWarning.__module__ == "mlx_minimax_music3.pipeline"
    assert GenerationRequest.__module__ == "mlx_minimax_music3.pipeline"
    assert GenerationResult.__module__ == "mlx_minimax_music3.pipeline"
    assert Music3Pipeline.__module__ == "mlx_minimax_music3.pipeline"
    assert PromptQualityWarning.__module__ == "mlx_minimax_music3.prompting"
    assert instrumental_lyrics.__module__ == "mlx_minimax_music3.prompting"
