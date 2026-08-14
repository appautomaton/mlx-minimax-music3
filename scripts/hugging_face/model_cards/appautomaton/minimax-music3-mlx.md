---
license: other
license_name: minimax-music3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE
library_name: mlx
pipeline_tag: text-to-audio
tags:
- mlx
- apple-silicon
- macos
- minimax
- minimax-music3
- audio-generation
- generative-audio
- music-generation
- text-to-audio
- text-to-music
- local-inference
- safetensors
---

# MiniMax Music 3 — MLX

[![PyPI](https://img.shields.io/pypi/v/mlx-minimax-music3?include_prereleases=true&style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/mlx-minimax-music3/)
[![GitHub](https://img.shields.io/badge/GitHub-mlx--minimax--music3-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/appautomaton/mlx-minimax-music3)
[![Project page](https://img.shields.io/badge/project-appautomaton.renocrypt.com-F59E0B?style=flat-square)](https://appautomaton.renocrypt.com/mlx-minimax-music3/)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb?style=flat-square)](https://appautomaton.renocrypt.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-MiniMax--Music3--MLX-yellow?style=flat-square)](https://huggingface.co/appautomaton/MiniMax-Music3-MLX)

Precision-preserving MLX-native layout conversion of
[`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
for local inference on Apple silicon. It is designed for use with
[`mlx-minimax-music3`](https://github.com/appautomaton/mlx-minimax-music3),
the independent pure-MLX inference project and Python package that generates
complete stereo music from lyrics and a structured music caption without
PyTorch, CUDA, or a cloud API at inference time.

This is a format and tensor-layout conversion. It is not trained, fine-tuned,
merged, or quantized, and it does not claim authorship of the underlying model.
MiniMax developed and released MiniMax Music 3; App Automaton converted the
published checkpoint for the independent MLX runtime.

## Checkpoint contents

| Component | Stored dtype | Size | Role |
| --- | --- | ---: | --- |
| Global language model | BF16 | 15.99 GiB | Long-range structure and semantic music tokens |
| RVQ depth decoder | BF16 | 1.20 GiB | Seven residual acoustic codebooks |
| Condition encoder | FP32 | 0.09 GiB | Continuous hidden-state fusion |
| Flow transformer | FP32 | 9.06 GiB | Flow-matching acoustic synthesis |
| Vocoder | FP32 | 0.20 GiB | Stereo waveform decode |
| Tokenizer, scheduler, and metadata | — | 0.01 GiB | Prompting and checkpoint contract |

The complete checkpoint is 26.56 GiB (28.52 GB decimal). The repository contains
only the dense profile. It does not contain selective-q8 or persistent FP16
derivatives.

## Conversion contract

The conversion is pinned to official source revision
[`fbdf52fbaaca799592917417eb05f1899f1255ec`](https://huggingface.co/MiniMaxAI/MiniMax-Music3/tree/fbdf52fbaaca799592917417eb05f1899f1255ec).
Its `manifest.json` records the source revision, mapping version, component file
sizes, tensor counts, dtypes, and SHA-256 digests.

- Qwen3 and RVQ tensors retain their published BF16 values.
- Condition, flow, and vocoder tensors retain their published FP32 values.
- Convolution kernels are transposed into the channels-last layout expected by
  MLX.
- Vocoder weight normalization is folded into the stored convolution weights.
- No tensor is downcast or quantized.

The converter reads and writes SafeTensors directly through MLX. PyTorch is not
part of conversion or runtime inference.

## Use with MLX

Install the current package from
[`mlx-minimax-music3` on PyPI](https://pypi.org/project/mlx-minimax-music3/):

```sh
uv add --prerelease=allow mlx-minimax-music3
```

Download this checkpoint into a local weight directory:

```sh
hf download appautomaton/MiniMax-Music3-MLX \
  --local-dir weights/mlx-dense/MiniMax-Music3
```

Generate one minute of instrumental melodic techno:

```python
from mlx_minimax_music3 import (
    GenerationRequest,
    Music3Pipeline,
    instrumental_lyrics,
)

pipeline = Music3Pipeline("weights/mlx-dense/MiniMax-Music3")
result = pipeline.generate(
    GenerationRequest(
        caption=(
            "Global Metadata: melodic techno, 128 BPM, A minor, nocturnal and "
            "cinematic, gradually rising energy. Vocal Details: instrumental, "
            "no vocals. Arrangement: deep rounded kick, warm sub-bass, crisp "
            "hats, syncopated percussion, analog arpeggiator, evolving pads, "
            "a glassy bell motif, controlled builds, and a spacious final drop."
        ),
        lyrics=instrumental_lyrics(
            "intro", "groove", "build", "drop", "breakdown", "outro"
        ),
        audio_duration=60.0,
        seed=7,
    ),
    output="outputs/melodic-techno.wav",
)

print(result.metadata.stage_timings)
print(result.metadata.memory_reports)
```

`audio_duration` is a ceiling because the model may emit its end token earlier.
Set `min_audio_duration` when a minimum frame count is required. The default
checkpoint path keeps the official mixed precision: BF16 autoregressive models
and FP32 acoustic models.

## Runtime behavior

The runtime loads one stage at a time. Autoregressive models are released before
the flow transformer is loaded, and acoustic models are released before final
waveform decoding. This bounds unified-memory residency and avoids retaining the
entire checkpoint in memory at once.

The current runtime writes native 44.1 kHz stereo PCM16 WAV. The official serving
profile resamples its output to 32 kHz; reference-output parity for that final
profile remains in progress.

## Validation status

This is an alpha release. The dense checkpoint has passed:

- strict tensor-name, shape, dtype, and shard-index validation;
- tensor-by-tensor conversion checks against the pinned source;
- complete checkpoint manifest digest verification;
- weightless golden regression tests for dense loading and inference; and
- end-to-end local generation, including a three-minute default-FP32 run.

On an Apple M5 Max with 128 GB unified memory, the three-minute validation run
took 18 minutes 16 seconds, peaked at approximately 19.93 GiB of process memory,
and did not increase swap usage. This is one machine-specific observation, not a
portable performance guarantee.

Listening validation across more prompts and seeds, long-form quality parity,
and the reference 32 kHz output profile are still in progress.

## Intended use and limitations

This checkpoint is intended for local research, development, and music
generation with the MLX runtime on Apple silicon.

- Prompt controls such as tempo, key, instrumentation, lyrics, and structure are
  generative guidance rather than strict symbolic guarantees.
- Outputs can contain artifacts, incorrect words, unexpected structure, or
  content that does not follow every requested attribute.
- Users are responsible for evaluating generated content, respecting applicable
  rights, and complying with the model license and acceptable-use policy.
- The checkpoint is not an official MiniMax MLX release, and this project is not
  affiliated with or endorsed by MiniMax.

For the original architecture description, prompt guidance, examples, and model
limitations, read the
[`MiniMaxAI/MiniMax-Music3` model card](https://huggingface.co/MiniMaxAI/MiniMax-Music3).

## License

The converted checkpoint remains governed by the included
[`MiniMax-Music3 Community License`](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE),
including its attribution, acceptable-use, safeguards, and commercial terms.
Review that license before downloading, redistributing, or deploying the model.

The `mlx-minimax-music3` runtime code is separately licensed under MIT.

## Links

- Source model: [`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
- Runtime source: [`appautomaton/mlx-minimax-music3`](https://github.com/appautomaton/mlx-minimax-music3)
- Python package: [`mlx-minimax-music3` on PyPI](https://pypi.org/project/mlx-minimax-music3/)
- Project page: [appautomaton.renocrypt.com/mlx-minimax-music3](https://appautomaton.renocrypt.com/mlx-minimax-music3/)
