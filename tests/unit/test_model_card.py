from __future__ import annotations

from pathlib import Path

CARD = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "hugging_face"
    / "model_cards"
    / "appautomaton"
    / "minimax-music3-mlx.md"
)


def _frontmatter() -> dict[str, str]:
    lines = CARD.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---"
    end = lines.index("---", 1)
    fields = {}
    for line in lines[1:end]:
        if line and not line.startswith(("-", " ")) and ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    return fields


def test_dense_model_card_has_hugging_face_metadata() -> None:
    fields = _frontmatter()
    frontmatter = CARD.read_text(encoding="utf-8").split("---", 2)[1]

    assert fields["library_name"] == "mlx"
    assert fields["pipeline_tag"] == "text-to-audio"
    assert fields["license"] == "other"
    assert fields["license_name"] == "minimax-music3-community-license"
    assert fields["license_link"].endswith("/MiniMax-Music3/blob/main/LICENSE")
    assert {
        "mlx",
        "apple-silicon",
        "macos",
        "minimax-music3",
        "audio-generation",
        "generative-audio",
        "music-generation",
        "text-to-audio",
        "text-to-music",
        "local-inference",
        "safetensors",
    } <= {
        line.removeprefix("- ")
        for line in frontmatter.splitlines()
        if line.startswith("- ")
    }


def test_dense_model_card_describes_a_format_conversion_truthfully() -> None:
    fields = _frontmatter()
    card = CARD.read_text(encoding="utf-8")

    # Hugging Face has no relation value for a precision-preserving layout
    # conversion. Omitting these fields avoids mislabeling it as a fine-tune or
    # quantized derivative; source lineage remains explicit in the card body.
    assert "base_model" not in fields
    assert "base_model_relation" not in fields
    assert "fbdf52fbaaca799592917417eb05f1899f1255ec" in card
    assert "No tensor is downcast or quantized" in card
    assert "Music3Pipeline" in card
