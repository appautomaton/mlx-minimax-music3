from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_minimax_music3.config import Qwen3Config
from mlx_minimax_music3.loading import (
    CheckpointLayoutError,
    discover_component_layout,
    load_component_weights,
)
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM


def _config() -> Qwen3Config:
    return Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        max_position_embeddings=16,
        published_dtype="float32",
    )


def test_strict_sharded_load_round_trip(tmp_path: Path) -> None:
    mx.random.seed(17)
    source = Qwen3ForCausalLM(_config())
    component = tmp_path / "language_model"
    component.mkdir()
    checkpoint = component / "model.safetensors"
    source.save_weights(str(checkpoint))
    expected = source(mx.array([[1, 2, 3]])).logits
    mx.eval(expected)

    mx.random.seed(23)
    restored = Qwen3ForCausalLM(_config())
    load_component_weights(
        restored,
        component,
        allowed_dtypes=frozenset({"F32"}),
    )
    actual = restored(mx.array([[1, 2, 3]])).logits
    mx.eval(actual)

    assert mx.array_equal(expected, actual).item()


def test_index_must_point_to_the_actual_tensor_shard(tmp_path: Path) -> None:
    component = tmp_path / "component"
    component.mkdir()
    mx.save_safetensors(
        str(component / "one.safetensors"), {"weight": mx.ones((2, 2))}
    )
    mx.save_safetensors(
        str(component / "two.safetensors"), {"bias": mx.ones((2,))}
    )
    (component / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight": "two.safetensors",
                    "bias": "one.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointLayoutError, match="misplaced"):
        discover_component_layout(component)


def test_index_rejects_parent_traversal(tmp_path: Path) -> None:
    component = tmp_path / "component"
    component.mkdir()
    outside = tmp_path / "outside.safetensors"
    mx.save_safetensors(str(outside), {"weight": mx.ones((1,))})
    (component / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointLayoutError, match="directly inside"):
        discover_component_layout(component)
