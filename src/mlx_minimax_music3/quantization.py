"""Strict selective-quantization policy for optimized Music 3 checkpoints."""

from __future__ import annotations

import re

from mlx import nn
from mlx.utils import tree_flatten

from .manifest import CheckpointManifest

_LANGUAGE_MODEL_LINEAR = re.compile(
    r"model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))"
)
_RVQ_LINEAR = re.compile(
    r"(?:projection|layers\.\d+\."
    r"(?:attn\.(?:to_q|to_k|to_v|to_out)|gate_proj|up_proj|down_proj))"
)
_POLICIES = {
    "language_model": _LANGUAGE_MODEL_LINEAR,
    "rvq_depth_decoder": _RVQ_LINEAR,
}


class QuantizationPolicyError(ValueError):
    """Raised when a q8 manifest and model topology disagree."""


def is_q8_module(component: str, module_path: str) -> bool:
    """Return whether a dense Linear path belongs to the q8 allowlist."""

    pattern = _POLICIES.get(component)
    return pattern is not None and pattern.fullmatch(module_path) is not None


def qualified_module_name(component: str, module_path: str) -> str:
    if not component or not module_path:
        raise ValueError("Component and module path must be non-empty")
    return f"{component}.{module_path}"


def expected_q8_modules(component: str, model: nn.Module) -> tuple[str, ...]:
    """Enumerate every allowlisted Linear in a concrete model topology."""

    modules = []
    for path, module in tree_flatten(
        model.leaf_modules(),
        is_leaf=nn.Module.is_module,
    ):
        if is_q8_module(component, path):
            if not isinstance(module, nn.Linear):
                raise QuantizationPolicyError(
                    f"Allowlisted module is not Linear: {component}.{path}"
                )
            modules.append(qualified_module_name(component, path))
    return tuple(sorted(modules))


def apply_q8_topology(
    component: str,
    model: nn.Module,
    manifest: CheckpointManifest,
) -> nn.Module:
    """Replace exactly the manifest-declared dense Linear modules with q8."""

    if manifest.profile != "q8":
        raise QuantizationPolicyError("q8 topology requires a q8 manifest")
    expected = set(expected_q8_modules(component, model))
    prefix = f"{component}."
    declared = {
        name for name in manifest.quantized_modules if name.startswith(prefix)
    }
    if declared != expected:
        missing = sorted(expected - declared)
        unexpected = sorted(declared - expected)
        raise QuantizationPolicyError(
            f"q8 policy mismatch for {component}: "
            f"{len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})"
        )

    nn.quantize(
        model,
        group_size=manifest.quantization_group_size,
        bits=manifest.quantization_bits,
        mode=manifest.quantization_mode,
        class_predicate=lambda path, module: (
            isinstance(module, nn.Linear)
            and qualified_module_name(component, path) in declared
        ),
    )
    return model


__all__ = [
    "QuantizationPolicyError",
    "apply_q8_topology",
    "expected_q8_modules",
    "is_q8_module",
    "qualified_module_name",
]
