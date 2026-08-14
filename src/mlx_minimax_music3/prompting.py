"""MiniMax Music 3 prompt assembly."""

from __future__ import annotations

import re
import warnings

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
CAPTION_START = "<|caption_start|>"
CAPTION_END = "<|caption_end|>"
LYRICS_START = "<|lyrics_start|>"
LYRICS_END = "<|lyrics_end|>"
AUDIO_START = "<|audio_start|>"

AUDIO_CFG_TOKEN_ID = 151_654
AUDIO_END_TOKEN_ID = 151_670
AUDIO_CODE_OFFSET = 151_675
SEMANTIC_VOCAB_SIZE = 16_384
MAX_PROMPT_TOKENS = 5_000
MAX_AUDIO_FRAMES = 9_000

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")
_STRUCTURE_ONLY_RE = re.compile(r"^(?:\[[^\]\n]+\][ \t]*)+$")


class PromptQualityWarning(UserWarning):
    """Warn when a valid prompt is likely to produce a degenerate result."""


def clean_caption(caption: str) -> str:
    """Normalize the caption exactly as expected by the released checkpoint."""

    def rewrite_special_tag(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(rewrite_special_tag, caption)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines.append(line.rstrip())

    text = "\n".join(lines)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def normalize_lyrics(lyrics: str) -> str:
    """Normalize lyrics and structural tags for checkpoint-compatible input."""

    lines = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        lines.append(match.group(1).strip() if match else line)

    text = "\n".join(lines)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def instrumental_lyrics(*sections: str) -> str:
    """Build non-empty instrumental lyrics in the checkpoint's expected form."""

    if not sections:
        sections = ("intro",)
    lines = []
    for section in sections:
        if not isinstance(section, str):
            raise TypeError("instrumental section names must be strings")
        name = section.strip().lower()
        if not name or any(character in name for character in "[]\r\n"):
            raise ValueError(f"invalid instrumental section name: {section!r}")
        lines.extend((f"[{name}]", "(instrumental)"))
    return "\n".join(lines)


def _has_lyrics_content(normalized_lyrics: str) -> bool:
    return any(
        line != "[start]" and not _STRUCTURE_ONLY_RE.fullmatch(line)
        for raw_line in normalized_lyrics.splitlines()
        if (line := raw_line.strip())
    )


def build_prompt(caption: str, lyrics: str) -> str:
    """Build the exact text prompt consumed by the Music 3 tokenizer."""

    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("caption must be a non-empty string")
    if not isinstance(lyrics, str) or not lyrics.strip():
        raise ValueError("lyrics must be a non-empty string")

    normalized_lyrics = normalize_lyrics(lyrics)
    if not _has_lyrics_content(normalized_lyrics):
        warnings.warn(
            "lyrics contain only structure tags; add lyric text or an explicit "
            "'(instrumental)' line to avoid weak conditioning",
            PromptQualityWarning,
            stacklevel=2,
        )

    return (
        f"{IM_START}{CAPTION_START}{clean_caption(caption)}{CAPTION_END}"
        f"{LYRICS_START}{normalized_lyrics}{LYRICS_END}{IM_END}{AUDIO_START}"
    )


__all__ = [
    "AUDIO_CFG_TOKEN_ID",
    "AUDIO_CODE_OFFSET",
    "AUDIO_END_TOKEN_ID",
    "MAX_AUDIO_FRAMES",
    "MAX_PROMPT_TOKENS",
    "SEMANTIC_VOCAB_SIZE",
    "PromptQualityWarning",
    "build_prompt",
    "clean_caption",
    "instrumental_lyrics",
    "normalize_lyrics",
]
