"""Tests for the checkpoint-compatible prompt contract."""

import warnings

import pytest

from mlx_minimax_music3.prompting import (
    PromptQualityWarning,
    build_prompt,
    clean_caption,
    instrumental_lyrics,
    normalize_lyrics,
)


def test_clean_caption_normalizes_supported_markup() -> None:
    caption = "## Acoustic pop\n- Warm **female** vocal\n\n<|tempo 96|>"

    assert clean_caption(caption) == "Acoustic pop\nWarm female vocal\ntempo is 96"


def test_normalize_lyrics_preserves_only_leading_structure_tags() -> None:
    lyrics = "[Verse] dropped text\nHello world [Chorus]\nSing"

    assert normalize_lyrics(lyrics) == "[start]\n[verse]\nHello world\n[chorus]\nSing"


def test_build_prompt_matches_checkpoint_boundaries() -> None:
    prompt = build_prompt("Warm acoustic pop", "[Verse]\nHello world")

    assert prompt == (
        "<|im_start|><|caption_start|>Warm acoustic pop<|caption_end|>"
        "<|lyrics_start|>[start]\n[verse]\nHello world<|lyrics_end|>"
        "<|im_end|><|audio_start|>"
    )


def test_instrumental_lyrics_adds_explicit_content() -> None:
    assert instrumental_lyrics() == "[intro]\n(instrumental)"
    assert instrumental_lyrics("Intro", "Outro") == (
        "[intro]\n(instrumental)\n[outro]\n(instrumental)"
    )


@pytest.mark.parametrize("section", ["", "[intro]", "intro\noutro"])
def test_instrumental_lyrics_rejects_invalid_sections(section: str) -> None:
    with pytest.raises(ValueError):
        instrumental_lyrics(section)


def test_build_prompt_warns_when_lyrics_have_only_structure_tags() -> None:
    with pytest.warns(PromptQualityWarning, match="only structure tags"):
        build_prompt("Driving techno", "[instrumental]")


def test_explicit_instrumental_content_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_prompt("Driving techno", instrumental_lyrics("instrumental"))


@pytest.mark.parametrize(("caption", "lyrics"), [("", "lyrics"), ("caption", " ")])
def test_build_prompt_rejects_empty_inputs(caption: str, lyrics: str) -> None:
    with pytest.raises(ValueError):
        build_prompt(caption, lyrics)
