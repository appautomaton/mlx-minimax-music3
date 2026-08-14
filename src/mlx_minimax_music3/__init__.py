"""Pure MLX inference for MiniMax Music 3."""

from ._version import __version__
from .pipeline import (
    ExperimentalPrecisionWarning,
    ExperimentalQuantizationWarning,
    GenerationRequest,
    GenerationResult,
    Music3Pipeline,
)
from .prompting import PromptQualityWarning, instrumental_lyrics

__all__ = [
    "ExperimentalPrecisionWarning",
    "ExperimentalQuantizationWarning",
    "GenerationRequest",
    "GenerationResult",
    "Music3Pipeline",
    "PromptQualityWarning",
    "__version__",
    "instrumental_lyrics",
]
