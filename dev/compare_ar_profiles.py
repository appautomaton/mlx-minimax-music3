"""Compare dense and quantized autoregressive trajectories with forced replay."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mlx_minimax_music3.autoregressive import (
    AutoregressiveConfig,
    AutoregressiveResult,
    generate_autoregressive,
)
from mlx_minimax_music3.loading import load_language_model, load_rvq_depth_decoder
from mlx_minimax_music3.sampling import sample_top_k, sanitize_logits
from mlx_minimax_music3.tokenizer import Qwen2BPETokenizer, TokenizedPrompt


@dataclass(frozen=True, slots=True)
class RecordedDraw:
    top_k: int
    seed: int
    position: int
    column: int


class RecordingSampler:
    """Record reference-sampler decisions without changing their behavior."""

    def __init__(self) -> None:
        self.draws: list[RecordedDraw] = []

    def __call__(
        self,
        logits: mx.array,
        *,
        top_k: int,
        seed: int,
        position: int,
    ) -> mx.array:
        sampled = sample_top_k(
            logits,
            top_k=top_k,
            seed=seed,
            position=position,
        )
        column = int(sampled.item())
        self.draws.append(RecordedDraw(top_k, seed, position, column))
        return sampled


class ReplaySampler:
    """Force recorded decisions while measuring their rank under new logits."""

    def __init__(self, draws: list[RecordedDraw]) -> None:
        self._draws = draws
        self._cursor = 0
        self.ranks: list[int] = []

    def __call__(
        self,
        logits: mx.array,
        *,
        top_k: int,
        seed: int,
        position: int,
    ) -> mx.array:
        if self._cursor >= len(self._draws):
            raise RuntimeError("forced replay exhausted its recorded draws")
        draw = self._draws[self._cursor]
        self._cursor += 1
        if (top_k, seed, position) != (draw.top_k, draw.seed, draw.position):
            raise RuntimeError(
                "forced replay sampling schedule differs from the baseline"
            )

        values = sanitize_logits(logits)
        chosen = values[0, draw.column]
        rank = mx.sum(values[0] > chosen).astype(mx.int32)
        mx.eval(rank)
        self.ranks.append(int(rank.item()))
        return mx.array([draw.column], dtype=mx.int32)

    def finish(self) -> None:
        if self._cursor != len(self._draws):
            raise RuntimeError(
                f"forced replay consumed {self._cursor} of {len(self._draws)} draws"
            )


def _run(
    checkpoint: Path,
    prompt: TokenizedPrompt,
    config: AutoregressiveConfig,
    sampler: RecordingSampler | ReplaySampler,
) -> AutoregressiveResult:
    language_model = load_language_model(checkpoint)
    depth_decoder = load_rvq_depth_decoder(checkpoint)
    try:
        result = generate_autoregressive(
            language_model,
            depth_decoder,
            prompt,
            config,
            sampler=sampler,
        )
        mx.eval(result.codes, result.frame_hiddens)
        return result
    finally:
        del language_model, depth_decoder
        gc.collect()
        mx.synchronize()
        mx.clear_cache()


def _hidden_metrics(reference: mx.array, candidate: mx.array) -> dict[str, float]:
    frames = min(reference.shape[1], candidate.shape[1])
    reference = reference[:, :frames].astype(mx.float32)
    candidate = candidate[:, :frames].astype(mx.float32)
    dot = mx.sum(reference * candidate, axis=-1)
    norms = mx.sqrt(mx.sum(reference * reference, axis=-1)) * mx.sqrt(
        mx.sum(candidate * candidate, axis=-1)
    )
    cosine = dot / mx.maximum(norms, mx.array(1e-20, dtype=mx.float32))
    mae = mx.mean(mx.abs(reference - candidate))
    mx.eval(cosine, mae)
    return {
        "cosine_mean": float(mx.mean(cosine).item()),
        "cosine_min": float(mx.min(cosine).item()),
        "mae": float(mae.item()),
    }


def _first_code_difference(
    reference: mx.array, candidate: mx.array
) -> dict[str, int] | None:
    frames = min(reference.shape[1], candidate.shape[1])
    reference_codes = reference[0, :frames].tolist()
    candidate_codes = candidate[0, :frames].tolist()
    for frame, (reference_frame, candidate_frame) in enumerate(
        zip(reference_codes, candidate_codes, strict=True)
    ):
        for codebook, (reference_code, candidate_code) in enumerate(
            zip(reference_frame, candidate_frame, strict=True)
        ):
            if reference_code != candidate_code:
                return {
                    "frame": frame,
                    "codebook": codebook,
                    "dense_code": reference_code,
                    "q8_code": candidate_code,
                }
    if reference.shape[1] != candidate.shape[1]:
        return {"frame": frames, "codebook": -1, "dense_code": -1, "q8_code": -1}
    return None


def compare_profiles(
    dense_checkpoint: Path,
    q8_checkpoint: Path,
    *,
    caption: str,
    lyrics: str,
    duration: float,
    seed: int,
) -> dict[str, object]:
    tokenizer = Qwen2BPETokenizer.from_directory(dense_checkpoint)
    prompt = tokenizer.encode_prompt(caption, lyrics)
    config = AutoregressiveConfig(audio_duration=duration, seed=seed)

    dense_sampler = RecordingSampler()
    dense = _run(dense_checkpoint, prompt, config, dense_sampler)

    replay_sampler = ReplaySampler(dense_sampler.draws)
    q8_forced = _run(q8_checkpoint, prompt, config, replay_sampler)
    replay_sampler.finish()

    q8_sampler = RecordingSampler()
    q8_free = _run(q8_checkpoint, prompt, config, q8_sampler)

    ranks = replay_sampler.ranks
    report = {
        "duration_requested": duration,
        "seed": seed,
        "dense_frames": dense.num_frames,
        "q8_forced_frames": q8_forced.num_frames,
        "q8_free_frames": q8_free.num_frames,
        "recorded_draws": len(dense_sampler.draws),
        "forced_codes_match": bool(mx.array_equal(dense.codes, q8_forced.codes).item()),
        "forced_hidden": _hidden_metrics(dense.frame_hiddens, q8_forced.frame_hiddens),
        "forced_rank_mean": sum(ranks) / len(ranks),
        "forced_rank_max": max(ranks),
        "forced_outside_top_k": sum(
            rank >= draw.top_k
            for rank, draw in zip(ranks, dense_sampler.draws, strict=True)
        ),
        "free_first_code_difference": _first_code_difference(
            dense.codes, q8_free.codes
        ),
        "free_hidden": _hidden_metrics(dense.frame_hiddens, q8_free.frame_hiddens),
    }
    if not math.isfinite(report["forced_rank_mean"]):
        raise RuntimeError("non-finite rank summary")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_checkpoint", type=Path)
    parser.add_argument("q8_checkpoint", type=Path)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = compare_profiles(
        args.dense_checkpoint,
        args.q8_checkpoint,
        caption=args.caption,
        lyrics=args.lyrics,
        duration=args.duration,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
