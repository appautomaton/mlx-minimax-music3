"""Weightless contracts for the checkpoint-native Qwen2 BPE tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_minimax_music3.prompting import AUDIO_CFG_TOKEN_ID
from mlx_minimax_music3.tokenizer import Qwen2BPETokenizer, TokenizerError

_PRETOKENIZER_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| "
    r"?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_MUSIC_TOKENS = {
    "<|im_start|>": 151_644,
    "<|im_end|>": 151_645,
    "<|audio_cfg|>": 151_654,
    "<|audio_start|>": 151_669,
    "<|audio_end|>": 151_670,
    "<|caption_start|>": 151_671,
    "<|caption_end|>": 151_672,
    "<|lyrics_start|>": 151_673,
    "<|lyrics_end|>": 151_674,
}


def _byte_alphabet() -> dict[str, int]:
    """Build the canonical GPT-2 byte alphabet with byte values as token IDs."""

    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    byte_values = list(visible)
    codepoints = list(visible)
    extra = 0
    for byte in range(256):
        if byte not in visible:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {
        chr(codepoint): byte
        for byte, codepoint in zip(byte_values, codepoints, strict=True)
    }


def _tokenizer_contract(*, normalizer: object = None) -> dict[str, object]:
    vocab = _byte_alphabet()
    vocab.update({"ab": 256, "abc": 257})
    return {
        "normalizer": {"type": "NFC"} if normalizer is None else normalizer,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": _PRETOKENIZER_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [["a", "b"], ["ab", "c"]],
        },
        "added_tokens": [
            {
                "id": token_id,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
            for content, token_id in _MUSIC_TOKENS.items()
        ],
    }


def _write_tokenizer(tmp_path: Path, **changes: object) -> Path:
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps(_tokenizer_contract(**changes)),
        encoding="utf-8",
    )
    return path


def test_synthetic_tokenizer_preserves_utf8_and_applies_bpe(tmp_path: Path) -> None:
    tokenizer = Qwen2BPETokenizer(_write_tokenizer(tmp_path))

    assert tokenizer.encode("café") == tuple("café".encode())
    assert tokenizer.encode("cafe\N{COMBINING ACUTE ACCENT}") == tuple(
        "café".encode()
    )
    assert tokenizer.encode("abc") == (257,)
    assert tokenizer.encode("<|im_start|>Hi") == (151_644, ord("H"), ord("i"))


def test_synthetic_tokenizer_builds_aligned_prompt_pair(tmp_path: Path) -> None:
    tokenizer = Qwen2BPETokenizer(_write_tokenizer(tmp_path))

    prompt = tokenizer.encode_prompt(
        "Warm synth pop",
        "[Verse]\nStay with me",
    )

    assert prompt.conditional[0] == 151_644
    assert prompt.conditional[-2:] == (151_645, 151_669)
    assert prompt.unconditional[0] == prompt.conditional[0]
    assert prompt.unconditional[-2:] == prompt.conditional[-2:]
    assert set(prompt.unconditional[1:-2]) == {AUDIO_CFG_TOKEN_ID}
    assert prompt.length == len(prompt.conditional) == len(prompt.unconditional)


def test_synthetic_tokenizer_rejects_contract_drift(tmp_path: Path) -> None:
    path = _write_tokenizer(tmp_path, normalizer={"type": "Lowercase"})

    with pytest.raises(TokenizerError, match="Expected NFC normalizer"):
        Qwen2BPETokenizer(path)
