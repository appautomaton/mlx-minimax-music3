<div align="center">

# mlx-minimax-music3

**Pure MLX inference for MiniMax Music 3 on Apple silicon.**

[![Development status](https://img.shields.io/badge/status-pre--alpha-F59E0B?style=flat-square)](https://github.com/appautomaton/mlx-minimax-music3)
[![PyPI](https://img.shields.io/pypi/v/mlx-minimax-music3?include_prereleases=true&style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/mlx-minimax-music3/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-native-000000?style=flat-square&logo=apple&logoColor=white)](https://support.apple.com/mac/)
[![MLX](https://img.shields.io/badge/backend-MLX-7C3AED?style=flat-square)](https://github.com/ml-explore/mlx)

[**PyPI**](https://pypi.org/project/mlx-minimax-music3/) ·
[**Project site**](https://appautomaton.renocrypt.com/mlx-minimax-music3/)

</div>

`mlx-minimax-music3` is an independent project for local MiniMax Music 3
inference with MLX. The runtime accepts lyrics and a structured music
caption, generates the model's autoregressive music representation, synthesizes
the acoustic latents, and returns stereo waveform audio without using PyTorch or
CUDA at runtime.

> [!IMPORTANT]
> The current release is an alpha. Dense tensor mapping and waveform execution
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

Install the current alpha release from
[PyPI](https://pypi.org/project/mlx-minimax-music3/):

```sh
uv add --prerelease=allow mlx-minimax-music3
```

Model weights remain a separate, explicit local download.

## Dependency policy

The runtime dependency is only `mlx`. A small local tokenizer reads the
checkpoint's exact Qwen2 BPE vocabulary and is covered by runtime-generated,
weightless tokenizer contracts. Dependencies are added only when a working
implementation proves they are necessary.

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
| Selective-q8 execution | Supported compatibility profile; local conversion only |
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

`audio_duration` is the generation ceiling because the model may emit its audio
end token sooner. Set `min_audio_duration` to suppress early stopping until a
required minimum; setting both values to the same duration requests an exact
autoregressive frame count.

To test lower-precision acoustic inference without creating another checkpoint,
cast the FP32 flow parameters once as their shards load. The model keeps its
Euler state and final waveform decode in FP32:

```python
pipeline = Music3Pipeline(
    "weights/mlx-dense/MiniMax-Music3",
    flow_compute_dtype="float16",
)
```

Runtime FP16 flow compute is experimental; the default remains the checkpoint's
FP32 correctness path.

Selective-q8 checkpoints use the same pipeline API. The loader reads the
checkpoint manifest, reconstructs the declared quantized topology, and validates
every stored tensor before inference; no quantization flag is required:

```python
pipeline = Music3Pipeline("weights/mlx-8bit/MiniMax-Music3")
```

Selective-q8 is a memory-oriented compatibility profile, not the recommended
quality or throughput baseline. This project provides and tests the local
converter but does not publish or distribute q8 model weights. Dense weights
remain the reference inference profile.

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
uv run pytest -q tests/unit
uv run pytest -q tests/integration
uv run python dev/check_public_tree.py
uv build --no-sources
```

Unit tests and weightless integration tests are separate default gates. Every
pytest test runs without the Official Music 3 checkpoint, downloaded tokenizer
assets, network access, or other local model files. See the
[testing guide](https://github.com/appautomaton/mlx-minimax-music3/blob/main/docs/testing.md)
for the test boundaries and GitHub CI cadence.

Convert the componentized official checkpoint to the dense baseline. The final
command optionally creates a local selective-q8 compatibility profile for
memory-constrained systems. It is not a published weight artifact or a
throughput optimization:

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

## Release process

Published versions are available from the natural
[PyPI project URL](https://pypi.org/project/mlx-minimax-music3/) and correspond
to GitHub release tags. For each release, `.github/workflows/workflow.yml` builds
and verifies the distributions before publishing them through PyPI trusted
publishing.

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
