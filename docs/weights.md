# Checkpoint layout

Model files are local runtime inputs and never belong in Git, wheels, or source
distributions. The initial checkpoint contract is pinned to the official Hugging
Face revision `fbdf52fbaaca799592917417eb05f1899f1255ec`.

## Required components

The official repository contains both a componentized Diffusers layout and the
legacy source checkpoints used to create it. The MLX runtime needs exactly one
copy of the componentized layout:

| Component | Runtime role | Published weight size |
|---|---|---:|
| `language_model/` | Qwen3-based global autoregressive model | 15.99 GiB |
| `rvq_depth_decoder/` | Seven residual acoustic codebooks | 1.20 GiB |
| `condition_encoder/` | Hidden-state fusion and acoustic projection | 0.09 GiB |
| `transformer/` | Flow-matching Diffusion Transformer | 9.06 GiB |
| `vocoder/` | Flow-VAE / DAC-style waveform decoder | 0.20 GiB |
| `tokenizer/` and `scheduler/` | Prompt tokenization and Euler solver configuration | 0.01 GiB |
| Root metadata | Component index, model configuration, model card, and license | less than 0.01 GiB |

The selected 25 files total 26.56 GiB. The legacy `qwen_7B/`,
`flowmatching_vae.pth`, and `dav.pth` files are another 26.82 GiB representation
of the same model and must not be downloaded for normal MLX development.

## Local directory structure

Follow the sibling `mlx-video` separation between published dense checkpoints
and MLX-optimized checkpoints while preserving the official component names:

```text
weights/
├── README.md
├── bf16/
│   └── MiniMax-Music3/
│       ├── LICENSE
│       ├── README.md
│       ├── config.json
│       ├── modular_model_index.json
│       ├── tokenizer/
│       ├── language_model/
│       ├── rvq_depth_decoder/
│       ├── condition_encoder/
│       ├── transformer/
│       ├── scheduler/
│       └── vocoder/
├── mlx-dense/
│   └── MiniMax-Music3/
│       ├── manifest.json
│       └── <MLX-native component tree>
└── mlx-8bit/
    └── MiniMax-Music3/
        ├── manifest.json
        ├── tokenizer/
        ├── language_model/
        ├── rvq_depth_decoder/
        ├── condition_encoder/
        ├── transformer/
        ├── scheduler/
        └── vocoder/
```

`bf16/` is the established local name for the published dense-source bucket. It
does not assert that every upstream tensor is BF16: the acoustic checkpoints
include higher-precision tensors and must retain their published dtypes until
component parity is established.

`mlx-dense/` contains the losslessly remapped MLX-native parity checkpoint.
`mlx-8bit/` contains its optional quantized derivative. Both layouts mirror the
source component tree so loaders do not need a second component-discovery scheme.
`manifest.json` records source repository, source revision, tensor mapping
version, quantization mode, group size, output dtype, and a digest for every
generated file.

## Selective, resumable download

Use the current `hf` command through `uvx`. This keeps `huggingface_hub` out of
the runtime dependencies. Start with a dry run:

```sh
uvx hf download MiniMaxAI/MiniMax-Music3 \
  LICENSE README.md config.json modular_model_index.json \
  tokenizer/ language_model/ rvq_depth_decoder/ condition_encoder/ \
  transformer/ scheduler/ vocoder/ \
  --revision fbdf52fbaaca799592917417eb05f1899f1255ec \
  --local-dir weights/bf16/MiniMax-Music3 \
  --dry-run
```

Remove `--dry-run` to download. The command writes local transfer metadata under
`weights/bf16/MiniMax-Music3/.cache/huggingface/`; retain that ignored directory
so an interrupted transfer can resume without restarting completed files.

Before downloading, allow at least 30 GiB for the source checkpoint. Allow at
least 80 GiB when the source, dense MLX conversion, quantized conversion, and
temporary conversion files will coexist.
Do not use a branch name as the revision in automated setup: a commit hash makes
the local manifest reproducible even when the upstream repository changes.

## Loading and residency rules

- Parse and validate every configuration before allocating model parameters.
- Load Safetensors directly with MLX; do not add PyTorch merely to read weights.
- Keep tensor-name remapping explicit and independently testable.
- Reject missing, unexpected, duplicate, or shape-mismatched tensors with an
  actionable component and tensor name.
- Preserve published precision until that component passes numerical parity.
- Load the autoregressive and acoustic phases in separate residency scopes.
- Do not let the top-level pipeline retain references to every model component.
- Synchronize required outputs before releasing a stage, then release its model
  objects and cached buffers before loading the next large stage.

The dense autoregressive weights are approximately 17.19 GiB, while the dense
condition encoder, flow transformer, and vocoder together are approximately
9.35 GiB. Activations, key-value caches, generated hidden states, temporary
conversion buffers, and Metal allocator overhead are additional and must be
measured rather than inferred from file size.

## Conversion policy

MLX can read the official componentized SafeTensors directly, but the published
keys and some kernel layouts follow Transformers and Diffusers modules. A
development-only converter therefore reads one source shard at a time, applies
an explicit component mapping, validates every destination tensor, and writes a
lossless MLX-native checkpoint under `weights/mlx-dense/`. This conversion uses
MLX and does not require PyTorch.

MLX affine 8-bit conversion is a separate optional mode and never runs implicitly
during inference. Quantize one component only after its dense MLX implementation
matches the reference. The initial profile quantizes allowlisted language-model
and RVQ linear layers while leaving every published FP32 acoustic tensor dense.
Keep numerically sensitive normalization, embeddings, sampling heads, scheduler
state, and waveform-decoder operations at validated precision.

Converted artifacts remain under `weights/mlx-dense/` or `weights/mlx-8bit/` and
are never included in a package distribution. A converted checkpoint is usable
only when its manifest matches the loader's tensor-mapping version and the
recorded official source revision.

Create the validated dense profile and experimental selective affine-q8 profile
with:

```sh
uv run python -m dev.convert_checkpoint \
  weights/bf16/MiniMax-Music3 \
  weights/mlx-dense/MiniMax-Music3

uv run python -m dev.quantize_checkpoint \
  weights/mlx-dense/MiniMax-Music3 \
  weights/mlx-8bit/MiniMax-Music3
```

The initial q8 policy quantizes 281 large BF16 affine modules: 252 Qwen3
attention/MLP projections and 29 RVQ attention/MLP projections. Sampling heads,
audio heads, embeddings, norms, the FP32 flow transformer, condition encoder,
and vocoder remain dense. The manifest names every quantized module; the loader
rejects any policy or topology drift. This policy is retained for sensitivity
analysis only: a 250-frame generation diverged materially from dense and did not
pass listening quality, so it is not a release-quality profile.
