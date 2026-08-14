from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_minimax_music3.config import (
    ConfigError,
    FlowTransformerConfig,
    Music3Config,
    Qwen3Config,
    RVQDepthDecoderConfig,
)


def test_loads_downloaded_checkpoint_configuration_if_available() -> None:
    root = Path("weights/bf16/MiniMax-Music3")
    if not root.is_dir():
        pytest.skip("Local checkpoint is not available")

    config = Music3Config.from_directory(root)

    assert config.language_model.hidden_size == 4096
    assert config.language_model.num_hidden_layers == 36
    assert config.rvq_depth_decoder.num_residual_codebooks == 7
    assert config.transformer.hidden_size == 2048
    assert config.vocoder.total_upsampling_ratio == 512
    assert config.vocoder.sampling_rate == 44_100


def test_qwen_rejects_inconsistent_attention_shape() -> None:
    with pytest.raises(ConfigError, match="hidden_size"):
        Qwen3Config(
            hidden_size=24,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            max_position_embeddings=128,
        )


def test_qwen_rejects_unsupported_layer_type(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "vocab_size": 64,
                "max_position_embeddings": 128,
                "model_type": "qwen3",
                "rope_parameters": {
                    "rope_theta": 1_000_000,
                    "rope_type": "default",
                },
                "layer_types": ["sliding_attention"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="full-attention"):
        Qwen3Config.from_file(path)


def test_rvq_requires_one_position_per_codebook() -> None:
    with pytest.raises(ConfigError, match="complete codebook frame"):
        RVQDepthDecoderConfig(
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_attention_heads=2,
            audio_vocab_size=32,
            num_codebooks=8,
            max_position_embeddings=7,
        )


def test_flow_transformer_exposes_derived_hidden_size() -> None:
    config = FlowTransformerConfig(
        in_channels=8,
        condition_dim=32,
        num_layers=2,
        num_attention_heads=4,
        attention_head_dim=8,
        ff_inner_dim=64,
        fourier_embedding_dim=16,
        rotary_dim=4,
    )

    assert config.hidden_size == 32
