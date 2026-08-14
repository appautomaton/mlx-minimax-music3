from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from mlx_minimax_music3.prompting import AUDIO_CFG_TOKEN_ID, build_prompt
from mlx_minimax_music3.tokenizer import Qwen2BPETokenizer


@pytest.fixture(scope="module")
def tokenizer_path() -> Path:
    path = Path("weights/bf16/MiniMax-Music3/tokenizer/tokenizer.json")
    if not path.is_file():
        pytest.skip("Local Music 3 tokenizer is not available")
    return path


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
def test_matches_official_rust_tokenizer(tokenizer_path: Path, text: str) -> None:
    local = Qwen2BPETokenizer(tokenizer_path)
    reference = Tokenizer.from_file(str(tokenizer_path))

    assert local.encode(text) == tuple(reference.encode(text).ids)


def test_prompt_pair_matches_reference(tokenizer_path: Path) -> None:
    local = Qwen2BPETokenizer(tokenizer_path)
    reference = Tokenizer.from_file(str(tokenizer_path))
    caption = "Dreamy electropop, female vocal, 118 BPM"
    lyrics = "[Verse]\n星光穿过窗口\n[Chorus]\nStay with me tonight"

    prompt = local.encode_prompt(caption, lyrics)
    expected = tuple(reference.encode(build_prompt(caption, lyrics)).ids)

    assert prompt.conditional == expected
    assert prompt.unconditional[0] == prompt.conditional[0]
    assert prompt.unconditional[-2:] == prompt.conditional[-2:]
    assert set(prompt.unconditional[1:-2]) == {AUDIO_CFG_TOKEN_ID}
