"""Validated configuration objects for the released Music 3 components."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self


class ConfigError(ValueError):
    """Raised when a checkpoint configuration is invalid or unsupported."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Missing configuration file: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration must be a JSON object: {path}")
    return value


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key!r} must be an object")
    return value


def _string(data: Mapping[str, Any], key: str, *, default: str | None = None) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key!r} must be a string")
    return value


def _integer(data: Mapping[str, Any], key: str, *, default: int | None = None) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key!r} must be an integer")
    return value


def _number(
    data: Mapping[str, Any], key: str, *, default: float | None = None
) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{key!r} must be a number")
    return float(value)


def _boolean(
    data: Mapping[str, Any], key: str, *, default: bool | None = None
) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key!r} must be a boolean")
    return value


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value!r}")


def _expect(name: str, value: object, expected: object) -> None:
    if value != expected:
        raise ConfigError(f"Unsupported {name}: expected {expected!r}, got {value!r}")


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    """Configuration for the global Qwen3 causal language model."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    model_type: str = "qwen3"
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_type: str = "default"
    published_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
        ):
            _positive(name, getattr(self, name))
        _positive("rms_norm_eps", self.rms_norm_eps)
        _positive("rope_theta", self.rope_theta)
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ConfigError(
                "hidden_size must equal num_attention_heads * head_dim"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ConfigError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        _expect("model_type", self.model_type, "qwen3")
        _expect("hidden_act", self.hidden_act, "silu")
        _expect("attention_bias", self.attention_bias, False)
        _expect("attention_dropout", self.attention_dropout, 0.0)
        _expect("rope_type", self.rope_type, "default")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        rope = _mapping(data, "rope_parameters")
        layer_types = data.get("layer_types")
        if layer_types is not None:
            if not isinstance(layer_types, Sequence) or isinstance(layer_types, str):
                raise ConfigError("'layer_types' must be an array")
            if any(value != "full_attention" for value in layer_types):
                raise ConfigError("Only full-attention Qwen3 layers are supported")
            if len(layer_types) != _integer(data, "num_hidden_layers"):
                raise ConfigError("layer_types length does not match num_hidden_layers")
        return cls(
            hidden_size=_integer(data, "hidden_size"),
            intermediate_size=_integer(data, "intermediate_size"),
            num_hidden_layers=_integer(data, "num_hidden_layers"),
            num_attention_heads=_integer(data, "num_attention_heads"),
            num_key_value_heads=_integer(data, "num_key_value_heads"),
            head_dim=_integer(data, "head_dim"),
            vocab_size=_integer(data, "vocab_size"),
            max_position_embeddings=_integer(data, "max_position_embeddings"),
            rms_norm_eps=_number(data, "rms_norm_eps", default=1e-6),
            rope_theta=_number(rope, "rope_theta"),
            tie_word_embeddings=_boolean(
                data, "tie_word_embeddings", default=False
            ),
            model_type=_string(data, "model_type"),
            hidden_act=_string(data, "hidden_act", default="silu"),
            attention_bias=_boolean(data, "attention_bias", default=False),
            attention_dropout=_number(data, "attention_dropout", default=0.0),
            rope_type=_string(rope, "rope_type", default="default"),
            published_dtype=_string(data, "dtype", default="bfloat16"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        return cls.from_dict(_load_json(Path(path)))


@dataclass(frozen=True, slots=True)
class RVQDepthDecoderConfig:
    """Configuration for the within-frame residual codebook decoder."""

    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    audio_vocab_size: int
    num_codebooks: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_attention_heads",
            "audio_vocab_size",
            "num_codebooks",
            "max_position_embeddings",
        ):
            _positive(name, getattr(self, name))
        _positive("rms_norm_eps", self.rms_norm_eps)
        if self.hidden_size % self.num_attention_heads:
            raise ConfigError("hidden_size must be divisible by num_attention_heads")
        if self.num_codebooks < 2:
            raise ConfigError("num_codebooks must include semantic and residual codes")
        if self.max_position_embeddings < self.num_codebooks:
            raise ConfigError(
                "max_position_embeddings must fit one complete codebook frame"
            )

    @property
    def num_residual_codebooks(self) -> int:
        return self.num_codebooks - 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            hidden_size=_integer(data, "hidden_size"),
            intermediate_size=_integer(data, "intermediate_size"),
            num_layers=_integer(data, "num_layers"),
            num_attention_heads=_integer(data, "num_attention_heads"),
            audio_vocab_size=_integer(data, "audio_vocab_size"),
            num_codebooks=_integer(data, "num_codebooks"),
            max_position_embeddings=_integer(data, "max_position_embeddings"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        return cls.from_dict(_load_json(Path(path)))


@dataclass(frozen=True, slots=True)
class ConditionEncoderConfig:
    """Configuration for the autoregressive-to-acoustic condition encoder."""

    condition_hidden_dim: int
    num_condition_layers: int
    out_dim: int
    input_sampling_rate: int
    input_hop_length: int
    output_sampling_rate: int
    output_hop_length: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _positive(name, getattr(self, name))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            condition_hidden_dim=_integer(data, "condition_hidden_dim"),
            num_condition_layers=_integer(data, "num_condition_layers"),
            out_dim=_integer(data, "out_dim"),
            input_sampling_rate=_integer(data, "input_sampling_rate"),
            input_hop_length=_integer(data, "input_hop_length"),
            output_sampling_rate=_integer(data, "output_sampling_rate"),
            output_hop_length=_integer(data, "output_hop_length"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        return cls.from_dict(_load_json(Path(path)))


@dataclass(frozen=True, slots=True)
class FlowTransformerConfig:
    """Configuration for the one-dimensional flow-matching transformer."""

    in_channels: int
    condition_dim: int
    num_layers: int
    num_attention_heads: int
    attention_head_dim: int
    ff_inner_dim: int
    fourier_embedding_dim: int
    rotary_dim: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _positive(name, getattr(self, name))
        if self.rotary_dim > self.attention_head_dim:
            raise ConfigError("rotary_dim cannot exceed attention_head_dim")
        if self.rotary_dim % 2:
            raise ConfigError("rotary_dim must be even")

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            in_channels=_integer(data, "in_channels"),
            condition_dim=_integer(data, "condition_dim"),
            num_layers=_integer(data, "num_layers"),
            num_attention_heads=_integer(data, "num_attention_heads"),
            attention_head_dim=_integer(data, "attention_head_dim"),
            ff_inner_dim=_integer(data, "ff_inner_dim"),
            fourier_embedding_dim=_integer(data, "fourier_embedding_dim"),
            rotary_dim=_integer(data, "rotary_dim"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        return cls.from_dict(_load_json(Path(path)))


@dataclass(frozen=True, slots=True)
class VocoderConfig:
    """Configuration for the Flow-VAE/DAC-style waveform decoder."""

    latent_channels: int
    decoder_input_dim: int
    decoder_hidden_dim: int
    upsampling_ratios: tuple[int, ...]
    sampling_rate: int

    def __post_init__(self) -> None:
        for name in (
            "latent_channels",
            "decoder_input_dim",
            "decoder_hidden_dim",
            "sampling_rate",
        ):
            _positive(name, getattr(self, name))
        if not self.upsampling_ratios:
            raise ConfigError("upsampling_ratios cannot be empty")
        for ratio in self.upsampling_ratios:
            _positive("upsampling ratio", ratio)
        divisor = 2 ** len(self.upsampling_ratios)
        if self.decoder_hidden_dim % divisor:
            raise ConfigError(
                "decoder_hidden_dim must support channel halving at every block"
            )

    @property
    def total_upsampling_ratio(self) -> int:
        result = 1
        for ratio in self.upsampling_ratios:
            result *= ratio
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        raw_ratios = data.get("upsampling_ratios")
        if not isinstance(raw_ratios, Sequence) or isinstance(raw_ratios, str):
            raise ConfigError("'upsampling_ratios' must be an array")
        ratios = tuple(
            _integer({"ratio": ratio}, "ratio") for ratio in raw_ratios
        )
        return cls(
            latent_channels=_integer(data, "latent_channels"),
            decoder_input_dim=_integer(data, "decoder_input_dim"),
            decoder_hidden_dim=_integer(data, "decoder_hidden_dim"),
            upsampling_ratios=ratios,
            sampling_rate=_integer(data, "sampling_rate"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        return cls.from_dict(_load_json(Path(path)))


@dataclass(frozen=True, slots=True)
class FlowSchedulerConfig:
    """Supported subset of the released flow-matching Euler scheduler."""

    num_train_timesteps: int
    shift: float
    invert_sigmas: bool
    stochastic_sampling: bool
    use_dynamic_shifting: bool

    def __post_init__(self) -> None:
        _expect("num_train_timesteps", self.num_train_timesteps, 1)
        _expect("shift", self.shift, 1.0)
        _expect("invert_sigmas", self.invert_sigmas, True)
        _expect("stochastic_sampling", self.stochastic_sampling, False)
        _expect("use_dynamic_shifting", self.use_dynamic_shifting, False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            num_train_timesteps=_integer(data, "num_train_timesteps"),
            shift=_number(data, "shift"),
            invert_sigmas=_boolean(data, "invert_sigmas"),
            stochastic_sampling=_boolean(
                data, "stochastic_sampling", default=False
            ),
            use_dynamic_shifting=_boolean(
                data, "use_dynamic_shifting", default=False
            ),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        return cls.from_dict(_load_json(Path(path)))


@dataclass(frozen=True, slots=True)
class Music3Config:
    """Validated configuration bundle for one local Music 3 checkpoint."""

    language_model: Qwen3Config
    rvq_depth_decoder: RVQDepthDecoderConfig
    condition_encoder: ConditionEncoderConfig
    transformer: FlowTransformerConfig
    scheduler: FlowSchedulerConfig
    vocoder: VocoderConfig

    def __post_init__(self) -> None:
        if self.language_model.hidden_size != self.rvq_depth_decoder.hidden_size:
            raise ConfigError(
                "Language model and RVQ decoder hidden sizes must match"
            )
        if self.condition_encoder.condition_hidden_dim != self.language_model.hidden_size:
            raise ConfigError(
                "condition_hidden_dim must match the autoregressive hidden size"
            )
        if (
            self.condition_encoder.num_condition_layers
            != self.rvq_depth_decoder.num_codebooks
        ):
            raise ConfigError(
                "num_condition_layers must match the number of codebook states"
            )
        if self.condition_encoder.out_dim != self.transformer.condition_dim:
            raise ConfigError(
                "Condition encoder output must match transformer condition_dim"
            )
        if self.transformer.in_channels != self.vocoder.latent_channels:
            raise ConfigError(
                "Transformer in_channels must match vocoder latent_channels"
            )
        if self.condition_encoder.output_sampling_rate != self.vocoder.sampling_rate:
            raise ConfigError(
                "Condition encoder and vocoder sampling rates must match"
            )

    @classmethod
    def from_directory(cls, root: str | Path) -> Self:
        root = Path(root)
        return cls(
            language_model=Qwen3Config.from_file(root / "language_model/config.json"),
            rvq_depth_decoder=RVQDepthDecoderConfig.from_file(
                root / "rvq_depth_decoder/config.json"
            ),
            condition_encoder=ConditionEncoderConfig.from_file(
                root / "condition_encoder/config.json"
            ),
            transformer=FlowTransformerConfig.from_file(
                root / "transformer/config.json"
            ),
            scheduler=FlowSchedulerConfig.from_file(
                root / "scheduler/scheduler_config.json"
            ),
            vocoder=VocoderConfig.from_file(root / "vocoder/config.json"),
        )
