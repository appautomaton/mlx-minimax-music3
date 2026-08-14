"""Source-free regression tests for the Official Music 3 model contract."""

from __future__ import annotations

import hashlib
import math
from typing import Any, cast

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from dev.convert_checkpoint import plan_tensor
from mlx_minimax_music3.checkpoint import TensorInfo
from mlx_minimax_music3.config import (
    ConditionEncoderConfig,
    FlowTransformerConfig,
    Qwen3Config,
    RVQDepthDecoderConfig,
    VocoderConfig,
)
from mlx_minimax_music3.models.condition_encoder import ConditionEncoder
from mlx_minimax_music3.models.flow_transformer import FlowTransformer
from mlx_minimax_music3.models.qwen3 import Qwen3ForCausalLM
from mlx_minimax_music3.models.rvq_depth import RVQDepthDecoder
from mlx_minimax_music3.models.vocoder import Vocoder
from tests.support.official_schema import load_official_schema_contract

pytestmark = pytest.mark.integration

_CONTRACT = load_official_schema_contract()
_MAPPING_CASES = cast(list[dict[str, Any]], _CONTRACT["mapping_cases"])


def _case_id(case: dict[str, Any]) -> str:
    return f"{case['component']}:{case['source_name']}"


def _schema_digest(model, dtype: str) -> tuple[int, str]:
    rows = sorted(
        f"{name}|{tuple(value.shape)}|{dtype}"
        for name, value in tree_flatten(model.parameters())
    )
    return len(rows), hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def test_mlx_topology_matches_persisted_official_mapping_contract() -> None:
    components = cast(dict[str, dict[str, Any]], _CONTRACT["components"])
    models = {
        "language_model": Qwen3ForCausalLM(
            Qwen3Config.from_dict(components["language_model"]["config"])
        ),
        "rvq_depth_decoder": RVQDepthDecoder(
            RVQDepthDecoderConfig.from_dict(
                components["rvq_depth_decoder"]["config"]
            )
        ),
        "condition_encoder": ConditionEncoder(
            ConditionEncoderConfig.from_dict(
                components["condition_encoder"]["config"]
            )
        ),
        "transformer": FlowTransformer(
            FlowTransformerConfig.from_dict(components["transformer"]["config"])
        ),
        "vocoder": Vocoder(
            VocoderConfig.from_dict(components["vocoder"]["config"])
        ),
    }

    total = 0
    for name, model in models.items():
        component = components[name]
        count, digest = _schema_digest(model, component["mapping_dtype"])
        assert count == component["mapping_count"]
        assert digest == component["mapping_sha256"]
        assert sum(component["operations"].values()) == count
        total += count

    assert total == 982
    del models
    mx.clear_cache()


@pytest.mark.parametrize("case", _MAPPING_CASES, ids=_case_id)
def test_converter_preserves_official_mapping_cases(case: dict[str, Any]) -> None:
    shape = tuple(case["source_shape"])
    tensor = TensorInfo(
        name=case["source_name"],
        dtype=case["dtype"],
        shape=shape,
        data_offsets=(0, 0),
        numel=math.prod(shape),
        byte_size=0,
    )

    actual = plan_tensor(case["component"], tensor)
    expected = case["expected"]
    if expected is None:
        assert actual is None
        return

    assert actual is not None
    assert actual.source_name == case["source_name"]
    assert actual.source_shape == shape
    assert actual.dtype == case["dtype"]
    assert actual.output_name == expected["output_name"]
    assert actual.output_shape == tuple(expected["output_shape"])
    assert actual.operation == expected["operation"]
