<div align="center">

# mlx-minimax-music3

**Pure MLX inference for MiniMax Music 3 on Apple silicon.**

[![Development status](https://img.shields.io/badge/status-pre--alpha-F59E0B?style=flat-square)](https://github.com/appautomaton/mlx-minimax-music3)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-native-000000?style=flat-square&logo=apple&logoColor=white)](https://support.apple.com/mac/)
[![MLX](https://img.shields.io/badge/backend-MLX-7C3AED?style=flat-square)](https://github.com/ml-explore/mlx)

[**appautomaton.renocrypt.com/mlx-minimax-music3**](https://appautomaton.renocrypt.com/mlx-minimax-music3/)

</div>

`mlx-minimax-music3` is an independent project for local MiniMax Music 3
inference with MLX. The runtime accepts lyrics and a structured music
caption, generates the model's autoregressive music representation, synthesizes
the acoustic latents, and returns stereo waveform audio without using PyTorch or
CUDA at runtime.

> [!IMPORTANT]
> Version `0.0.1a0` is an alpha. Dense tensor mapping and waveform execution
> are validated locally, but end-to-end music quality parity, quantized quality,
> long-form generation, and the reference 32 kHz output profile remain in progress.

## Project goals

- Pure MLX inference on Apple silicon.
- End-to-end waveform generation, not token-only output.
- Local checkpoint loading with explicit weight mapping.
- Phase-scoped model residency for predictable unified-memory use.
- Small, testable model components with numerical correctness checks.
- No model weights or generated media in the source distribution.

## Installation

The package will be installable from PyPI after the first pre-release is
published:

```sh
uv add "mlx-minimax-music3==0.0.1a0"
```

Model weights remain a separate, explicit local download.

## Dependency policy

The runtime dependency is only `mlx`. A small local tokenizer reads the
checkpoint's exact Qwen2 BPE vocabulary; its output is continuously checked
against Hugging Face `tokenizers` during development. Dependencies are added only
when a working implementation proves they are necessary.

PyTorch, Diffusers, Transformers, Accelerate, Torchaudio, Librosa, and
`huggingface_hub` are intentionally excluded from the runtime. MLX reads
Safetensors directly, checkpoint paths are local-first, and standard-library WAV
output is preferred.

## Current status

| Area | Status |
|---|---|
| Repository and package metadata | Ready |
| PyPI trusted-publishing workflow | Ready |
| Architecture and porting contract | Ready |
| Official checkpoint inventory and conversion | Ready |
| Prompt and checkpoint-native tokenizer | Validated |
| Global language model and RVQ depth decoder | Validated |
| Flow-matching acoustic model | Validated |
| Waveform decoder | Validated |
| Dense end-to-end music quality | In validation |
| Selective-q8 execution | Experimental; multi-seed listening validation in progress |
| Long-form quality and 32 kHz output parity | In progress |

## Python API

The pipeline keeps only the checkpoint manifest and tokenizer between requests.
Model weights are loaded, evaluated, measured, and released one stage at a time.

```python
from mlx_minimax_music3 import GenerationRequest, Music3Pipeline

pipeline = Music3Pipeline("weights/mlx-dense/MiniMax-Music3")
result = pipeline.generate(
    GenerationRequest(
        caption="Warm acoustic folk, intimate vocal, gentle fingerpicked guitar.",
        lyrics="[verse]\nMorning light across the room\nA quiet road will lead me home",
        audio_duration=10.0,
        seed=0,
    ),
    output="outputs/song.wav",
)

print(result.metadata.checkpoint_profile)
print(result.metadata.memory_reports)
```

The current output profile is native 44.1 kHz stereo PCM16 WAV. The API refuses
to overwrite an existing file unless `overwrite=True` is explicit.

Lyrics must contain more than structure tags alone. For a vocal-free request,
use `instrumental_lyrics()` to add the explicit content expected by the model:

```python
from mlx_minimax_music3 import instrumental_lyrics

lyrics = instrumental_lyrics("intro", "outro")
```

## Intended runtime

```text
lyrics + structured caption
        |
        v
Qwen3 global language model + local RVQ depth decoder
        |
        v
continuous hidden-state conditioning
        |
        v
flow-matching diffusion transformer
        |
        v
Flow-VAE / DAC-style waveform decoder
        |
        v
native 44.1 kHz stereo WAV
```

The implementation is deliberately staged so the autoregressive models can be
released before acoustic synthesis begins. See
[the architecture document](https://github.com/appautomaton/mlx-minimax-music3/blob/main/docs/architecture.md)
for the design.

## Development

```sh
git clone https://github.com/appautomaton/mlx-minimax-music3.git
cd mlx-minimax-music3
uv sync --locked
uv run ruff check .
uv run pytest -q
uv run python dev/check_public_tree.py
uv build --no-sources
```

Convert the componentized official checkpoint to the dense baseline. The second
command creates an experimental selective-q8 profile for memory and quality
research; it is not a quality-validated release profile:

```sh
uv run python -m dev.convert_checkpoint \
  weights/bf16/MiniMax-Music3 \
  weights/mlx-dense/MiniMax-Music3

uv run python -m dev.verify_dense_checkpoint \
  weights/bf16/MiniMax-Music3 \
  weights/mlx-dense/MiniMax-Music3 \
  --verify-digests

uv run python -m dev.quantize_checkpoint \
  weights/mlx-dense/MiniMax-Music3 \
  weights/mlx-8bit/MiniMax-Music3
```

Upstream implementations live in the ignored `.references/` directory. Existing
local checkouts may be reused through Git worktrees instead of downloading a
second copy. Their URLs, revisions, roles, and licenses are documented in
[the reference guide](https://github.com/appautomaton/mlx-minimax-music3/blob/main/docs/references.md).

## First pre-release

The first package version is `0.0.1a0`. After a GitHub pre-release is created from
tag `v0.0.1a0`, `.github/workflows/workflow.yml` will build the distributions and
publish them through PyPI trusted publishing.

The release must remain marked as a pre-release. Publishing is intentionally not
performed from a developer machine. The PyPI Trusted Publisher must match owner
`appautomaton`, repository `mlx-minimax-music3`, workflow `workflow.yml`, and
environment `pypi`. The environment scopes the trusted-publishing identity; this
sole-maintainer project does not require a reviewer approval rule.

## Licensing

The source code in this repository is MIT licensed. MiniMax Music 3 model weights
are distributed separately under the MiniMax-Music3 Community License. Installing
this package does not download or grant additional rights to those weights. See
[the third-party notices](https://github.com/appautomaton/mlx-minimax-music3/blob/main/THIRD_PARTY_NOTICES.md).

This project is not affiliated with or endorsed by MiniMax.
