from __future__ import annotations

import pytest
from mlx import nn

from mlx_minimax_music3.config import Qwen3Config
from mlx_minimax_music3.manifest import CheckpointManifest
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM
from mlx_minimax_music3.quantization import (
    QuantizationPolicyError,
    apply_q8_topology,
    expected_q8_modules,
)


def _model() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            max_position_embeddings=16,
            published_dtype="float32",
        )
    )


def _manifest(modules: tuple[str, ...]) -> CheckpointManifest:
    return CheckpointManifest(
        profile="q8",
        source_repository="MiniMaxAI/MiniMax-Music3",
        source_revision="f" * 40,
        components=(),
        quantized_modules=modules,
        quantization_mode="affine",
        quantization_bits=8,
        quantization_group_size=32,
    )


def test_q8_topology_quantizes_only_allowlisted_language_model_linears() -> None:
    model = _model()
    modules = expected_q8_modules("language_model", model)

    apply_q8_topology("language_model", model, _manifest(modules))

    layer = model.model.layers[0]
    assert len(modules) == 7
    assert isinstance(layer.self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(layer.mlp.down_proj, nn.QuantizedLinear)
    assert isinstance(model.model.embed_tokens, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)


def test_q8_topology_rejects_manifest_policy_drift() -> None:
    model = _model()
    modules = expected_q8_modules("language_model", model)

    with pytest.raises(QuantizationPolicyError, match="policy mismatch"):
        apply_q8_topology(
            "language_model",
            model,
            _manifest(modules[:-1]),
        )
