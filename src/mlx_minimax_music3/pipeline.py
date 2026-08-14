"""Stage-scoped end-to-end Music 3 inference."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import mlx.core as mx

from .acoustic import (
    AcousticLatents,
    FlowGenerationConfig,
    FlowProgress,
    generate_acoustic_latents,
)
from .audio import AudioFile, write_pcm16_wav
from .autoregressive import (
    AutoregressiveConfig,
    AutoregressiveResult,
    GenerationProgress,
    generate_autoregressive,
)
from .decoding import Waveform, decode_latent_chunks
from .loading import (
    load_condition_encoder,
    load_flow_transformer,
    load_language_model,
    load_rvq_depth_decoder,
    load_vocoder,
)
from .manifest import CheckpointManifest, ManifestError
from .models.condition_encoder import ConditionEncoder
from .models.flow_transformer import FlowTransformer
from .models.qwen3 import Qwen3ForCausalLM
from .models.rvq_depth import RVQDepthDecoder
from .stages import (
    DEFAULT_STAGE_MEMORY_POLICY,
    StageMemoryPolicy,
    StageMemoryReport,
    StageSession,
)
from .tokenizer import Qwen2BPETokenizer, TokenizedPrompt

_REQUIRED_COMPONENTS = frozenset(
    {
        "condition_encoder",
        "language_model",
        "rvq_depth_decoder",
        "scheduler",
        "tokenizer",
        "transformer",
        "vocoder",
    }
)
_GENERATION_LOCK = Lock()
_FLOW_COMPUTE_DTYPES = {
    "float16": mx.float16,
    "float32": mx.float32,
}


class ExperimentalQuantizationWarning(UserWarning):
    """Warn when a checkpoint profile has not passed music-quality validation."""


class ExperimentalPrecisionWarning(UserWarning):
    """Warn when runtime reduced precision needs listening validation."""


@dataclass(frozen=True, slots=True)
class _AutoregressiveModels:
    language_model: Qwen3ForCausalLM
    depth_decoder: RVQDepthDecoder


@dataclass(frozen=True, slots=True)
class _AcousticModels:
    condition_encoder: ConditionEncoder
    transformer: FlowTransformer


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Text conditioning and deterministic generation controls."""

    caption: str
    lyrics: str
    audio_duration: float = 60.0
    seed: int = 0
    autoregressive_cfg_scale: float = 1.5
    autoregressive_top_k: int = 50
    flow_steps: int = 30
    flow_cfg_scale: float = 1.7
    min_audio_duration: float = 0.0

    def __post_init__(self) -> None:
        _ = self.autoregressive_config, self.flow_config

    @property
    def autoregressive_config(self) -> AutoregressiveConfig:
        return AutoregressiveConfig(
            audio_duration=self.audio_duration,
            seed=self.seed,
            cfg_scale=self.autoregressive_cfg_scale,
            top_k=self.autoregressive_top_k,
            min_audio_duration=self.min_audio_duration,
        )

    @property
    def flow_config(self) -> FlowGenerationConfig:
        return FlowGenerationConfig(
            num_steps=self.flow_steps,
            cfg_scale=self.flow_cfg_scale,
        )


@dataclass(frozen=True, slots=True)
class StageTiming:
    label: str
    seconds: float


@dataclass(frozen=True, slots=True)
class GenerationMetadata:
    source_repository: str
    source_revision: str
    checkpoint_profile: str
    flow_compute_dtype: str
    seed: int
    frame_count: int
    chunk_count: int
    stopped_on_audio_end: bool
    sample_rate: int
    stage_timings: tuple[StageTiming, ...]
    memory_reports: tuple[StageMemoryReport, ...]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    waveform: Waveform
    metadata: GenerationMetadata
    audio_file: AudioFile | None = None


def _validate_checkpoint(
    checkpoint: str | Path,
    *,
    verify_digests: bool,
) -> tuple[Path, CheckpointManifest]:
    root = Path(checkpoint).resolve()
    manifest = CheckpointManifest.read(root / "manifest.json")
    components = {component.name for component in manifest.components}
    missing = _REQUIRED_COMPONENTS - components
    if missing:
        raise ManifestError(
            f"Checkpoint manifest is missing components: {sorted(missing)}"
        )
    stale_temporaries = sorted(root.rglob("*.tmp"))
    if stale_temporaries:
        raise ManifestError(
            f"Checkpoint contains stale temporary file: {stale_temporaries[0]}"
        )
    manifest.verify(root, digests=verify_digests)
    return root, manifest


def _load_autoregressive_models(checkpoint: Path) -> _AutoregressiveModels:
    return _AutoregressiveModels(
        language_model=load_language_model(checkpoint),
        depth_decoder=load_rvq_depth_decoder(checkpoint),
    )


def _load_acoustic_models(
    checkpoint: Path,
    flow_compute_dtype: str,
) -> _AcousticModels:
    return _AcousticModels(
        condition_encoder=load_condition_encoder(checkpoint),
        transformer=load_flow_transformer(
            checkpoint,
            compute_dtype=_FLOW_COMPUTE_DTYPES[flow_compute_dtype],
        ),
    )


def _run_autoregressive_stage(
    checkpoint: Path,
    prompt: TokenizedPrompt,
    config: AutoregressiveConfig,
    *,
    policy: StageMemoryPolicy,
    include_footprint: bool,
    progress: Callable[[GenerationProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[AutoregressiveResult, StageMemoryReport]:
    session = StageSession(
        "autoregressive",
        lambda: _load_autoregressive_models(checkpoint),
        policy=policy,
        include_footprint=include_footprint,
    )
    with session:
        result = generate_autoregressive(
            session.require_model().language_model,
            session.require_model().depth_decoder,
            prompt,
            config,
            progress=progress,
            cancelled=cancelled,
        )
        session.handoff(result.codes, result.frame_hiddens)
    if session.report is None:
        raise RuntimeError("Autoregressive stage did not produce a memory report")
    return result, session.report


def _run_acoustic_stage(
    checkpoint: Path,
    frame_hiddens: mx.array,
    *,
    seed: int,
    config: FlowGenerationConfig,
    flow_compute_dtype: str,
    policy: StageMemoryPolicy,
    include_footprint: bool,
    progress: Callable[[FlowProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[AcousticLatents, StageMemoryReport]:
    session = StageSession(
        "acoustic",
        lambda: _load_acoustic_models(checkpoint, flow_compute_dtype),
        policy=policy,
        include_footprint=include_footprint,
    )
    with session:
        result = generate_acoustic_latents(
            session.require_model().transformer,
            session.require_model().condition_encoder,
            frame_hiddens,
            seed=seed,
            config=config,
            progress=progress,
            cancelled=cancelled,
        )
        session.handoff(*(chunk.latents for chunk in result.chunks))
    if session.report is None:
        raise RuntimeError("Acoustic stage did not produce a memory report")
    return result, session.report


def _run_decode_stage(
    checkpoint: Path,
    acoustic: AcousticLatents,
    *,
    policy: StageMemoryPolicy,
    include_footprint: bool,
    progress: Callable[[int, int], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[Waveform, StageMemoryReport]:
    session = StageSession(
        "decode",
        lambda: load_vocoder(checkpoint),
        policy=policy,
        include_footprint=include_footprint,
    )
    with session:
        waveform = decode_latent_chunks(
            session.require_model(),
            acoustic,
            progress=progress,
            cancelled=cancelled,
        )
        session.handoff(waveform.samples)
    if session.report is None:
        raise RuntimeError("Decode stage did not produce a memory report")
    return waveform, session.report


def _generate(
    checkpoint: Path,
    manifest: CheckpointManifest,
    tokenizer: Qwen2BPETokenizer,
    request: GenerationRequest,
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
    memory_policy: StageMemoryPolicy,
    flow_compute_dtype: str,
    include_footprint: bool = True,
    autoregressive_progress: Callable[[GenerationProgress], None] | None = None,
    flow_progress: Callable[[FlowProgress], None] | None = None,
    decode_progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> GenerationResult:
    """Execute one request without retaining model weights between stages."""

    prompt = tokenizer.encode_prompt(request.caption, request.lyrics)
    timings = []
    reports = []
    autoregressive_config = request.autoregressive_config

    started = time.perf_counter()
    autoregressive, report = _run_autoregressive_stage(
        checkpoint,
        prompt,
        autoregressive_config,
        policy=memory_policy,
        include_footprint=include_footprint,
        progress=autoregressive_progress,
        cancelled=cancelled,
    )
    timings.append(StageTiming("autoregressive", time.perf_counter() - started))
    reports.append(report)
    frame_count = autoregressive.num_frames
    stopped_on_audio_end = autoregressive.stopped_on_audio_end
    frame_hiddens = autoregressive.frame_hiddens
    del autoregressive, prompt

    started = time.perf_counter()
    acoustic, report = _run_acoustic_stage(
        checkpoint,
        frame_hiddens,
        seed=request.seed,
        config=request.flow_config,
        flow_compute_dtype=flow_compute_dtype,
        policy=memory_policy,
        include_footprint=include_footprint,
        progress=flow_progress,
        cancelled=cancelled,
    )
    timings.append(StageTiming("acoustic", time.perf_counter() - started))
    reports.append(report)
    chunk_count = acoustic.num_chunks
    del frame_hiddens

    started = time.perf_counter()
    waveform, report = _run_decode_stage(
        checkpoint,
        acoustic,
        policy=memory_policy,
        include_footprint=include_footprint,
        progress=decode_progress,
        cancelled=cancelled,
    )
    timings.append(StageTiming("decode", time.perf_counter() - started))
    reports.append(report)
    del acoustic

    audio_file = None
    if output is not None:
        started = time.perf_counter()
        audio_file = write_pcm16_wav(
            waveform,
            output,
            overwrite=overwrite,
        )
        timings.append(StageTiming("output", time.perf_counter() - started))

    metadata = GenerationMetadata(
        source_repository=manifest.source_repository,
        source_revision=manifest.source_revision,
        checkpoint_profile=manifest.profile,
        flow_compute_dtype=flow_compute_dtype,
        seed=request.seed,
        frame_count=frame_count,
        chunk_count=chunk_count,
        stopped_on_audio_end=stopped_on_audio_end,
        sample_rate=waveform.sample_rate,
        stage_timings=tuple(timings),
        memory_reports=tuple(reports),
    )
    return GenerationResult(
        waveform=waveform,
        metadata=metadata,
        audio_file=audio_file,
    )


class Music3Pipeline:
    """Reusable local pipeline that never retains model weights between calls.

    Checkpoint integrity and tokenizer parsing happen once at construction. A
    lock serializes generation because MLX allocator telemetry and phase-scoped
    residency are process-global concerns.
    """

    __slots__ = (
        "_checkpoint",
        "_manifest",
        "_tokenizer",
        "flow_compute_dtype",
        "include_footprint",
        "memory_policy",
    )

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        verify_digests: bool = False,
        flow_compute_dtype: str = "float32",
        memory_policy: StageMemoryPolicy | None = None,
        include_footprint: bool = True,
    ) -> None:
        self._checkpoint, self._manifest = _validate_checkpoint(
            checkpoint,
            verify_digests=verify_digests,
        )
        if flow_compute_dtype not in _FLOW_COMPUTE_DTYPES:
            raise ValueError(
                "flow_compute_dtype must be 'float32' or 'float16'"
            )
        self.flow_compute_dtype = flow_compute_dtype
        if self._manifest.profile == "q8":
            warnings.warn(
                "the selective-q8 checkpoint is experimental: long-sequence "
                "autoregressive quality has not passed validation; use the dense "
                "profile as the correctness baseline",
                ExperimentalQuantizationWarning,
                stacklevel=2,
            )
        if flow_compute_dtype == "float16":
            warnings.warn(
                "runtime FP16 flow compute is experimental: parameters are cast "
                "after loading while Euler accumulation remains FP32; use "
                "float32 as the correctness baseline",
                ExperimentalPrecisionWarning,
                stacklevel=2,
            )
        self._tokenizer = Qwen2BPETokenizer.from_directory(self._checkpoint)
        self.memory_policy = memory_policy or DEFAULT_STAGE_MEMORY_POLICY
        self.include_footprint = include_footprint

    @property
    def checkpoint(self) -> Path:
        return self._checkpoint

    @property
    def checkpoint_profile(self) -> str:
        return self._manifest.profile

    def generate(
        self,
        request: GenerationRequest,
        *,
        output: str | Path | None = None,
        overwrite: bool = False,
        autoregressive_progress: Callable[[GenerationProgress], None] | None = None,
        flow_progress: Callable[[FlowProgress], None] | None = None,
        decode_progress: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        """Generate native 44.1 kHz stereo audio for one request."""

        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")
        with _GENERATION_LOCK:
            return _generate(
                self._checkpoint,
                self._manifest,
                self._tokenizer,
                request,
                output=output,
                overwrite=overwrite,
                memory_policy=self.memory_policy,
                flow_compute_dtype=self.flow_compute_dtype,
                include_footprint=self.include_footprint,
                autoregressive_progress=autoregressive_progress,
                flow_progress=flow_progress,
                decode_progress=decode_progress,
                cancelled=cancelled,
            )


def _run_pipeline(
    checkpoint: str | Path,
    request: GenerationRequest,
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
    verify_digests: bool = False,
    flow_compute_dtype: str = "float32",
    memory_policy: StageMemoryPolicy | None = None,
    include_footprint: bool = True,
    autoregressive_progress: Callable[[GenerationProgress], None] | None = None,
    flow_progress: Callable[[FlowProgress], None] | None = None,
    decode_progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> GenerationResult:
    """Construct a one-shot pipeline and generate one result."""

    pipeline = Music3Pipeline(
        checkpoint,
        verify_digests=verify_digests,
        flow_compute_dtype=flow_compute_dtype,
        memory_policy=memory_policy,
        include_footprint=include_footprint,
    )
    return pipeline.generate(
        request,
        output=output,
        overwrite=overwrite,
        autoregressive_progress=autoregressive_progress,
        flow_progress=flow_progress,
        decode_progress=decode_progress,
        cancelled=cancelled,
    )


__all__ = [
    "ExperimentalPrecisionWarning",
    "ExperimentalQuantizationWarning",
    "GenerationMetadata",
    "GenerationRequest",
    "GenerationResult",
    "Music3Pipeline",
    "StageTiming",
]
