"""Contracts that require the complete official Music 3 checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from dev.convert_checkpoint import plan_component
from mlx_minimax_music3.config import Music3Config, VocoderConfig
from mlx_minimax_music3.models.vocoder import Vocoder
from mlx_minimax_music3.prompting import AUDIO_CFG_TOKEN_ID, build_prompt
from mlx_minimax_music3.tokenizer import Qwen2BPETokenizer

pytestmark = pytest.mark.full_checkpoint

_DEFAULT_SOURCE = Path("weights/bf16/MiniMax-Music3")
_SOURCE_ENVIRONMENT = "MUSIC3_SOURCE_CHECKPOINT"


@pytest.fixture(scope="session")
def official_checkpoint() -> Path:
    root = Path(os.environ.get(_SOURCE_ENVIRONMENT, _DEFAULT_SOURCE))
    if not root.is_dir():
        pytest.fail(
            f"Official checkpoint not found at {root}. Set {_SOURCE_ENVIRONMENT} "
            "or download it to the default weights directory.",
            pytrace=False,
        )
    return root


def test_official_checkpoint_configuration(official_checkpoint: Path) -> None:
    config = Music3Config.from_directory(official_checkpoint)

    assert config.language_model.hidden_size == 4096
    assert config.language_model.num_hidden_layers == 36
    assert config.rvq_depth_decoder.num_residual_codebooks == 7
    assert config.transformer.hidden_size == 2048
    assert config.vocoder.total_upsampling_ratio == 512
    assert config.vocoder.sampling_rate == 44_100


@pytest.mark.parametrize(
    "text",
    [
        "A warm synth-pop anthem with bright drums.",
        "[Verse]\n霓虹落在雨里，café déjà vu 🎵\n[Chorus]\nWe're alive!",
        "  leading  spaces\tand\r\nnewlines  ",
        "Numbers 12345; symbols?! — ❤️‍🔥",
        "e\N{COMBINING ACUTE ACCENT} equals é after NFC",
        "<|im_start|><|caption_start|>ambient<|caption_end|>",
    ],
)
def test_matches_official_rust_tokenizer(
    official_checkpoint: Path,
    text: str,
) -> None:
    path = official_checkpoint / "tokenizer/tokenizer.json"
    local = Qwen2BPETokenizer(path)
    reference = Tokenizer.from_file(str(path))

    assert local.encode(text) == tuple(reference.encode(text).ids)


def test_prompt_pair_matches_reference(official_checkpoint: Path) -> None:
    path = official_checkpoint / "tokenizer/tokenizer.json"
    local = Qwen2BPETokenizer(path)
    reference = Tokenizer.from_file(str(path))
    caption = "Dreamy electropop, female vocal, 118 BPM"
    lyrics = "[Verse]\n星光穿过窗口\n[Chorus]\nStay with me tonight"

    prompt = local.encode_prompt(caption, lyrics)
    expected = tuple(reference.encode(build_prompt(caption, lyrics)).ids)

    assert prompt.conditional == expected
    assert prompt.unconditional[0] == prompt.conditional[0]
    assert prompt.unconditional[-2:] == prompt.conditional[-2:]
    assert set(prompt.unconditional[1:-2]) == {AUDIO_CFG_TOKEN_ID}


def test_official_vocoder_topology_matches_dense_plan(
    official_checkpoint: Path,
) -> None:
    config = VocoderConfig.from_file(
        official_checkpoint / "vocoder/config.json"
    )
    model = Vocoder(config)
    actual = {
        name: tuple(value.shape)
        for name, value in tree_flatten(model.parameters())
    }
    planned = {
        mapping.output_name: mapping.output_shape
        for mapping in plan_component(official_checkpoint, "vocoder")
    }

    assert actual == planned
