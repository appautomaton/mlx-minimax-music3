from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from dev.convert_checkpoint import convert_checkpoint
from dev.verify_dense_checkpoint import verify_dense_checkpoint


def _save(path: Path, weights: dict[str, mx.array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), weights)


def test_pure_mlx_dense_verifier_checks_every_mapping(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dense = tmp_path / "dense"
    _save(
        source / "language_model/model.safetensors",
        {"weight": mx.arange(8, dtype=mx.float32).reshape(2, 4)},
    )
    _save(
        source / "rvq_depth_decoder/model.safetensors",
        {"weight": mx.arange(6, dtype=mx.float32).reshape(2, 3)},
    )
    _save(
        source / "condition_encoder/model.safetensors",
        {"proj.weight": mx.arange(24, dtype=mx.float32).reshape(3, 2, 4)},
    )
    _save(
        source / "transformer/model.safetensors",
        {
            "preprocess_conv.weight": mx.arange(
                18, dtype=mx.float32
            ).reshape(3, 2, 3)
        },
    )
    _save(
        source / "vocoder/model.safetensors",
        {
            "conv_in.weight_g": mx.array([[[5.0]], [[10.0]]]),
            "conv_in.weight_v": mx.array(
                [
                    [[3.0, 0.0], [0.0, 4.0]],
                    [[0.0, 6.0], [8.0, 0.0]],
                ]
            ),
        },
    )

    convert_checkpoint(source, dense)
    summary = verify_dense_checkpoint(source, dense, verify_digests=True)

    assert summary.tensors_verified == 5
    assert summary.identity_tensors == 2
    assert summary.transposed_tensors == 2
    assert summary.folded_weight_norm_tensors == 1
    assert summary.hardlinked_files == 2
    assert summary.digests_verified
