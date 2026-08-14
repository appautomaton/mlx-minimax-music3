"""Checkpoint-native Qwen2 byte-level BPE tokenizer.

The implementation covers the exact normalization, pre-tokenization, added-token,
and BPE configuration published with MiniMax Music 3. It intentionally rejects a
different tokenizer configuration instead of silently producing incompatible
prompt IDs.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from .prompting import (
    AUDIO_CFG_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
    AUDIO_START,
    CAPTION_END,
    CAPTION_START,
    IM_END,
    IM_START,
    LYRICS_END,
    LYRICS_START,
    MAX_PROMPT_TOKENS,
    build_prompt,
)

_PRETOKENIZER_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| "
    r"?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_BPE_CACHE_SIZE = 16_384


class TokenizerError(ValueError):
    """Raised when tokenizer files or encoded prompts violate the contract."""


@dataclass(frozen=True, slots=True)
class TokenizedPrompt:
    """Conditional and classifier-free-guidance token rows."""

    conditional: tuple[int, ...]
    unconditional: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.conditional:
            raise TokenizerError("A tokenized prompt cannot be empty")
        if len(self.conditional) != len(self.unconditional):
            raise TokenizerError("Prompt rows must have the same length")

    @property
    def length(self) -> int:
        return len(self.conditional)

    def rows(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.conditional, self.unconditional


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TokenizerError(f"Missing tokenizer file: {path}") from error
    except json.JSONDecodeError as error:
        raise TokenizerError(f"Invalid tokenizer JSON: {path}") from error
    if not isinstance(value, dict):
        raise TokenizerError(f"Tokenizer root must be an object: {path}")
    return value


def _bytes_to_unicode() -> dict[int, str]:
    """Return the reversible GPT-2 byte alphabet used by ByteLevel."""

    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(byte_values)
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, codepoints), strict=True))


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _is_whitespace(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x0009 <= codepoint <= 0x000D
        or codepoint in {0x0020, 0x0085, 0x00A0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000}
        or 0x2000 <= codepoint <= 0x200A
    )


def _pretokenize(text: str) -> Iterator[str]:
    """Match the published Unicode-aware Qwen2 split expression."""

    contractions = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")
    index = 0
    while index < len(text):
        lowered = text[index : index + 3].lower()
        contraction = next(
            (value for value in contractions if lowered.startswith(value)), None
        )
        if contraction is not None:
            end = index + len(contraction)
            yield text[index:end]
            index = end
            continue

        character = text[index]
        if _is_letter(character):
            end = index + 1
            while end < len(text) and _is_letter(text[end]):
                end += 1
            yield text[index:end]
            index = end
            continue

        if (
            character not in "\r\n"
            and not _is_letter(character)
            and not _is_number(character)
            and index + 1 < len(text)
            and _is_letter(text[index + 1])
        ):
            end = index + 2
            while end < len(text) and _is_letter(text[end]):
                end += 1
            yield text[index:end]
            index = end
            continue

        if _is_number(character):
            yield character
            index += 1
            continue

        punctuation_start = index
        if (
            character == " "
            and index + 1 < len(text)
            and not _is_whitespace(text[index + 1])
            and not _is_letter(text[index + 1])
            and not _is_number(text[index + 1])
        ):
            index += 1
            character = text[index]
        if (
            not _is_whitespace(character)
            and not _is_letter(character)
            and not _is_number(character)
        ):
            index += 1
            while (
                index < len(text)
                and not _is_whitespace(text[index])
                and not _is_letter(text[index])
                and not _is_number(text[index])
            ):
                index += 1
            while index < len(text) and text[index] in "\r\n":
                index += 1
            yield text[punctuation_start:index]
            continue
        index = punctuation_start

        if not _is_whitespace(text[index]):
            raise TokenizerError(
                f"Pretokenizer could not classify U+{ord(text[index]):04X}"
            )
        run_end = index + 1
        while run_end < len(text) and _is_whitespace(text[run_end]):
            run_end += 1
        last_newline = max(
            (position for position in range(index, run_end) if text[position] in "\r\n"),
            default=-1,
        )
        if last_newline >= index:
            end = last_newline + 1
        elif run_end == len(text):
            end = run_end
        elif run_end - index > 1:
            end = run_end - 1
        else:
            end = run_end
        yield text[index:end]
        index = end


class Qwen2BPETokenizer:
    """Pure-Python encoder for the exact tokenizer.json checkpoint contract."""

    def __init__(self, tokenizer_file: str | Path) -> None:
        tokenizer_file = Path(tokenizer_file)
        data = _load_json(tokenizer_file)
        self._validate_pipeline(data)

        model = data["model"]
        raw_vocab = model["vocab"]
        if not isinstance(raw_vocab, dict) or not raw_vocab:
            raise TokenizerError("BPE vocabulary must be a non-empty object")
        if any(
            not isinstance(token, str)
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
            for token, token_id in raw_vocab.items()
        ):
            raise TokenizerError("BPE vocabulary must map strings to integer IDs")
        self._vocab: dict[str, int] = dict(raw_vocab)

        raw_merges = model["merges"]
        if not isinstance(raw_merges, list):
            raise TokenizerError("BPE merges must be an array")
        merges: dict[tuple[str, str], int] = {}
        for rank, pair in enumerate(raw_merges):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(part, str) for part in pair)
            ):
                raise TokenizerError(f"Invalid BPE merge at rank {rank}")
            key = (pair[0], pair[1])
            if key in merges:
                raise TokenizerError(f"Duplicate BPE merge: {key!r}")
            merges[key] = rank
        self._merge_ranks = merges
        self._byte_encoder = _bytes_to_unicode()
        self._piece_cache: dict[str, tuple[int, ...]] = {}

        raw_added_tokens = data.get("added_tokens")
        if not isinstance(raw_added_tokens, list):
            raise TokenizerError("added_tokens must be an array")
        added_tokens: dict[str, int] = {}
        for token in raw_added_tokens:
            if not isinstance(token, dict):
                raise TokenizerError("Every added token must be an object")
            content = token.get("content")
            token_id = token.get("id")
            if not isinstance(content, str) or not isinstance(token_id, int):
                raise TokenizerError("Added tokens need string content and integer IDs")
            if any(token.get(option) for option in ("single_word", "lstrip", "rstrip")):
                raise TokenizerError("Word-boundary or stripping added tokens are unsupported")
            if token.get("normalized") is not False:
                raise TokenizerError("Added tokens must bypass NFC normalization")
            added_tokens[content] = token_id
        self._added_tokens = added_tokens
        alternatives = "|".join(
            re.escape(token) for token in sorted(added_tokens, key=len, reverse=True)
        )
        self._added_pattern = re.compile(alternatives)
        self._validate_music_tokens()

    @staticmethod
    def _validate_pipeline(data: dict[str, Any]) -> None:
        normalizer = data.get("normalizer")
        if normalizer != {"type": "NFC"}:
            raise TokenizerError(f"Expected NFC normalizer, got {normalizer!r}")
        pretokenizer = data.get("pre_tokenizer")
        expected_pretokenizer = {
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
        }
        if pretokenizer != expected_pretokenizer:
            raise TokenizerError("Unsupported pre-tokenizer configuration")
        model = data.get("model")
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise TokenizerError("Expected a BPE tokenizer model")
        expected_model_options = {
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
        }
        for key, expected in expected_model_options.items():
            if model.get(key) != expected:
                raise TokenizerError(
                    f"Unsupported BPE option {key}: {model.get(key)!r}"
                )

    def _validate_music_tokens(self) -> None:
        expected = {
            IM_START: 151_644,
            IM_END: 151_645,
            "<|audio_cfg|>": AUDIO_CFG_TOKEN_ID,
            AUDIO_START: 151_669,
            "<|audio_end|>": AUDIO_END_TOKEN_ID,
            CAPTION_START: 151_671,
            CAPTION_END: 151_672,
            LYRICS_START: 151_673,
            LYRICS_END: 151_674,
        }
        actual = {token: self._added_tokens.get(token) for token in expected}
        if actual != expected:
            raise TokenizerError(
                f"Music special-token IDs do not match the checkpoint: {actual!r}"
            )

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        cached = self._piece_cache.get(piece)
        if cached is not None:
            return cached
        encoded = "".join(self._byte_encoder[byte] for byte in piece.encode("utf-8"))
        symbols = list(encoded)
        while len(symbols) > 1:
            candidate = min(
                (
                    (self._merge_ranks[pair], pair)
                    for pair in pairwise(symbols)
                    if pair in self._merge_ranks
                ),
                default=None,
            )
            if candidate is None:
                break
            _, pair = candidate
            merged = pair[0] + pair[1]
            updated = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and (
                    symbols[index], symbols[index + 1]
                ) == pair:
                    updated.append(merged)
                    index += 2
                else:
                    updated.append(symbols[index])
                    index += 1
            symbols = updated
        try:
            token_ids = tuple(self._vocab[symbol] for symbol in symbols)
        except KeyError as error:
            raise TokenizerError(f"BPE symbol is missing from vocabulary: {error}") from error
        if len(self._piece_cache) >= _BPE_CACHE_SIZE:
            self._piece_cache.pop(next(iter(self._piece_cache)))
        self._piece_cache[piece] = token_ids
        return token_ids

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        normalized = unicodedata.normalize("NFC", text)
        for piece in _pretokenize(normalized):
            yield from self._encode_piece(piece)

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode text without adding implicit beginning or end tokens."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids = []
        cursor = 0
        for match in self._added_pattern.finditer(text):
            if match.start() > cursor:
                token_ids.extend(self._encode_ordinary(text[cursor : match.start()]))
            token_ids.append(self._added_tokens[match.group(0)])
            cursor = match.end()
        if cursor < len(text):
            token_ids.extend(self._encode_ordinary(text[cursor:]))
        return tuple(token_ids)

    def encode_prompt(self, caption: str, lyrics: str) -> TokenizedPrompt:
        """Assemble and encode the conditional/unconditional prompt pair."""

        conditional = self.encode(build_prompt(caption, lyrics))
        if len(conditional) > MAX_PROMPT_TOKENS:
            raise TokenizerError(
                f"The assembled prompt has {len(conditional)} tokens; "
                f"the maximum is {MAX_PROMPT_TOKENS}"
            )
        expected_edges = (151_644, 151_645, 151_669)
        actual_edges = (conditional[0], conditional[-2], conditional[-1])
        if actual_edges != expected_edges:
            raise TokenizerError(
                f"Prompt token boundaries are invalid: {actual_edges!r}"
            )
        unconditional = (
            conditional[:1]
            + (AUDIO_CFG_TOKEN_ID,) * (len(conditional) - 3)
            + conditional[-2:]
        )
        return TokenizedPrompt(conditional, unconditional)

    @classmethod
    def from_directory(cls, checkpoint: str | Path) -> Qwen2BPETokenizer:
        root = Path(checkpoint)
        tokenizer_file = (
            root / "tokenizer.json" if root.name == "tokenizer" else root / "tokenizer/tokenizer.json"
        )
        return cls(tokenizer_file)
