from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from dev.convert_checkpoint import (
    _atomic_link_or_copy,
    _convert_weights,
    convert_checkpoint,
    plan_tensor,
)
from mlx_minimax_music3.checkpoint import TensorInfo


def _tensor(name: str, shape: tuple[int, ...]) -> TensorInfo:
    return TensorInfo(
        name=name,
        dtype="F32",
        shape=shape,
        data_offsets=(0, 0),
        numel=0,
        byte_size=0,
    )


def test_condition_conv_is_transposed_to_channels_last() -> None:
    mapping = plan_tensor(
        "condition_encoder", _tensor("proj.weight", (3, 2, 5))
    )
    assert mapping is not None
    source = mx.arange(30).reshape(3, 2, 5)

    converted = _convert_weights(
        "condition_encoder", {"proj.weight": source}, (mapping,)
    )

    assert converted["proj.weight"].shape == (3, 5, 2)
    assert mx.array_equal(
        converted["proj.weight"], mx.transpose(source, (0, 2, 1))
    ).item()


def test_vocoder_weight_norm_is_folded_before_transpose() -> None:
    mapping = plan_tensor(
        "vocoder", _tensor("conv_in.weight_v", (2, 2, 2))
    )
    assert mapping is not None
    weight_v = mx.array(
        [
            [[3.0, 0.0], [0.0, 4.0]],
            [[0.0, 5.0], [12.0, 0.0]],
        ]
    )
    weight_g = mx.array([[[10.0]], [[26.0]]])

    converted = _convert_weights(
        "vocoder",
        {
            "conv_in.weight_v": weight_v,
            "conv_in.weight_g": weight_g,
        },
        (mapping,),
    )["conv_in.weight"]
    mx.eval(converted)

    assert converted.shape == (2, 2, 2)
    assert mx.allclose(
        converted,
        mx.transpose(
            mx.array(
                [
                    [[6.0, 0.0], [0.0, 8.0]],
                    [[0.0, 10.0], [24.0, 0.0]],
                ]
            ),
            (0, 2, 1),
        ),
    ).item()


def test_atomic_link_is_idempotent_without_stale_temporary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "destination.safetensors"
    source.write_bytes(b"weights")

    _atomic_link_or_copy(source, destination)
    _atomic_link_or_copy(source, destination)

    assert destination.read_bytes() == b"weights"
    assert source.samefile(destination)
    assert not destination.with_suffix(".safetensors.tmp").exists()


def test_dense_converter_rejects_nested_source_and_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="must not be nested"):
        convert_checkpoint(source, tmp_path)
