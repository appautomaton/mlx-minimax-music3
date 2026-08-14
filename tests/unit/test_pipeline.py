from __future__ import annotations

import wave
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_minimax_music3 import pipeline
from mlx_minimax_music3.acoustic import AcousticLatents, LatentChunk
from mlx_minimax_music3.autoregressive import AutoregressiveResult
from mlx_minimax_music3.chunking import ChunkWindow
from mlx_minimax_music3.decoding import Waveform
from mlx_minimax_music3.manifest import CheckpointManifest, ComponentManifest
from mlx_minimax_music3.tokenizer import TokenizedPrompt


def _write_manifest(root: Path, *, profile: str = "dense") -> None:
    components = tuple(
        ComponentManifest(name=name, files=())
        for name in sorted(pipeline._REQUIRED_COMPONENTS)
    )
    CheckpointManifest(
        profile=profile,
        source_repository="MiniMaxAI/MiniMax-Music3",
        source_revision="f" * 40,
        components=components,
        quantized_modules=(
            ("language_model.model.layers.0.self_attn.q_proj",)
            if profile == "q8"
            else ()
        ),
        quantization_mode="affine" if profile == "q8" else None,
        quantization_bits=8 if profile == "q8" else None,
        quantization_group_size=64 if profile == "q8" else None,
    ).write(root / "manifest.json")


@pytest.mark.usefixtures("isolated_stage_memory")
def test_private_pipeline_orders_residency_and_writes_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_manifest(tmp_path)
    load_order = []

    class FakeTokenizer:
        def encode_prompt(self, caption: str, lyrics: str) -> TokenizedPrompt:
            assert caption == "clear caption"
            assert lyrics == "[verse]\nclear lyrics"
            return TokenizedPrompt((1, 2, 3), (1, 2, 3))

    monkeypatch.setattr(
        pipeline.Qwen2BPETokenizer,
        "from_directory",
        classmethod(lambda cls, checkpoint: FakeTokenizer()),
    )

    def load_autoregressive(checkpoint: Path):
        load_order.append("autoregressive")
        return pipeline._AutoregressiveModels(None, None)

    def load_acoustic(checkpoint: Path, flow_compute_dtype: str):
        assert flow_compute_dtype == "float32"
        load_order.append("acoustic")
        return pipeline._AcousticModels(None, None)

    def load_vocoder(checkpoint: Path):
        load_order.append("decode")
        return object()

    monkeypatch.setattr(pipeline, "_load_autoregressive_models", load_autoregressive)
    monkeypatch.setattr(pipeline, "_load_acoustic_models", load_acoustic)
    monkeypatch.setattr(pipeline, "load_vocoder", load_vocoder)
    monkeypatch.setattr(
        pipeline,
        "generate_autoregressive",
        lambda *args, **kwargs: AutoregressiveResult(
            codes=mx.zeros((1, 1, 8), dtype=mx.int32),
            frame_hiddens=mx.zeros((1, 1, 32), dtype=mx.bfloat16),
            stopped_on_audio_end=False,
        ),
    )
    window = ChunkWindow(0, 0, 1, True, True)
    monkeypatch.setattr(
        pipeline,
        "generate_acoustic_latents",
        lambda *args, **kwargs: AcousticLatents(
            chunks=(LatentChunk(window, mx.zeros((1, 3, 128))),)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "decode_latent_chunks",
        lambda *args, **kwargs: Waveform(mx.zeros((2, 8)), sample_rate=44_100),
    )
    output = tmp_path / "outputs/smoke.wav"

    result = pipeline._run_pipeline(
        tmp_path,
        pipeline.GenerationRequest(
            caption="clear caption",
            lyrics="[verse]\nclear lyrics",
            audio_duration=0.04,
            seed=7,
        ),
        output=output,
        include_footprint=False,
    )

    assert load_order == ["autoregressive", "acoustic", "decode"]
    assert result.metadata.frame_count == 1
    assert result.metadata.chunk_count == 1
    assert result.metadata.seed == 7
    assert result.metadata.flow_compute_dtype == "float32"
    assert [report.label for report in result.metadata.memory_reports] == [
        "autoregressive",
        "acoustic",
        "decode",
    ]
    assert result.audio_file is not None
    with wave.open(str(output), "rb") as audio:
        assert audio.getnchannels() == 2
        assert audio.getframerate() == 44_100
        assert audio.getnframes() == 8


def test_checkpoint_validation_accepts_manifest_declared_q8(tmp_path: Path) -> None:
    _write_manifest(tmp_path, profile="q8")

    root, manifest = pipeline._validate_checkpoint(
        tmp_path,
        verify_digests=False,
    )

    assert root == tmp_path.resolve()
    assert manifest.profile == "q8"


def test_q8_pipeline_warns_that_quality_is_experimental(
    tmp_path: Path, monkeypatch
) -> None:
    _write_manifest(tmp_path, profile="q8")
    monkeypatch.setattr(
        pipeline.Qwen2BPETokenizer,
        "from_directory",
        classmethod(lambda cls, checkpoint: object()),
    )

    with pytest.warns(
        pipeline.ExperimentalQuantizationWarning,
        match="correctness baseline",
    ):
        instance = pipeline.Music3Pipeline(tmp_path)

    assert instance.checkpoint_profile == "q8"


def test_runtime_f16_flow_warns_that_quality_is_experimental(
    tmp_path: Path, monkeypatch
) -> None:
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        pipeline.Qwen2BPETokenizer,
        "from_directory",
        classmethod(lambda cls, checkpoint: object()),
    )

    with pytest.warns(
        pipeline.ExperimentalPrecisionWarning,
        match="float32",
    ):
        instance = pipeline.Music3Pipeline(
            tmp_path,
            flow_compute_dtype="float16",
        )

    assert instance.checkpoint_profile == "dense"
    assert instance.flow_compute_dtype == "float16"


def test_pipeline_rejects_unknown_flow_compute_dtype(tmp_path: Path) -> None:
    _write_manifest(tmp_path)

    with pytest.raises(ValueError, match="flow_compute_dtype"):
        pipeline.Music3Pipeline(tmp_path, flow_compute_dtype="float8")


def test_generation_request_builds_validated_stage_configs() -> None:
    request = pipeline.GenerationRequest(
        caption="caption",
        lyrics="lyrics",
        audio_duration=4.0,
        seed=11,
        autoregressive_top_k=32,
        flow_steps=12,
    )

    assert request.autoregressive_config.max_frames == 100
    assert request.autoregressive_config.seed == 11
    assert request.autoregressive_config.top_k == 32
    assert request.flow_config.num_steps == 12
