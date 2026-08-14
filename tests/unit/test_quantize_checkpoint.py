from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from dev.quantize_checkpoint import _quantize_weights, plan_tensor, quantize_checkpoint
from mlx_minimax_music3.checkpoint import TensorInfo


def _tensor(name: str, shape: tuple[int, ...]) -> TensorInfo:
    return TensorInfo(
        name=name,
        dtype="BF16",
        shape=shape,
        data_offsets=(0, 0),
        numel=0,
        byte_size=0,
    )


def test_allowlisted_linear_expands_into_affine_q8_tensors() -> None:
    tensor = _tensor(
        "model.layers.0.self_attn.q_proj.weight",
        (64, 128),
    )

    plan = plan_tensor("language_model", tensor)

    assert plan.module_name == (
        "language_model.model.layers.0.self_attn.q_proj"
    )
    assert [(output.name, output.shape, output.dtype) for output in plan.outputs] == [
        (tensor.name, (64, 32), "U32"),
        ("model.layers.0.self_attn.q_proj.scales", (64, 2), "BF16"),
        ("model.layers.0.self_attn.q_proj.biases", (64, 2), "BF16"),
    ]


def test_quantized_tensor_payload_matches_planned_shapes() -> None:
    tensor = _tensor(
        "model.layers.0.mlp.down_proj.weight",
        (64, 128),
    )
    plan = plan_tensor("language_model", tensor)
    source = mx.ones(tensor.shape, dtype=mx.bfloat16)

    converted = _quantize_weights({tensor.name: source}, (plan,))
    mx.eval(converted)

    actual = {
        name: (value.shape, str(value.dtype))
        for name, value in converted.items()
    }
    dtype_names = {"BF16": "mlx.core.bfloat16", "U32": "mlx.core.uint32"}
    assert actual == {
        output.name: (output.shape, dtype_names[output.dtype])
        for output in plan.outputs
    }


def test_sampling_head_remains_dense() -> None:
    tensor = _tensor("lm_head.weight", (64, 128))

    plan = plan_tensor("language_model", tensor)

    assert not plan.quantized
    assert plan.outputs[0].name == "lm_head.weight"


def test_q8_converter_rejects_nested_source_and_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="must not be nested"):
        quantize_checkpoint(source, tmp_path)
