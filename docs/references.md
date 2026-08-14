# Reference implementations

Upstream source is kept in the ignored `.references/` directory for static
inspection. Reference code is read-only: it is not executed, vendored, imported
by the runtime, or included in distributions.

MiniMax H3 is a video model and is not a predecessor or implementation reference
for MiniMax Music 3. The sibling `mlx-video` project is useful only for general
MLX engineering patterns such as packaging, phased model residency, quantization,
and release automation.

## Local Music 3 references

Only repositories with a current MiniMax Music 3 implementation are checked out
locally:

| Local path | Upstream | Role | License |
|---|---|---|---|
| `.references/diffusers` | `huggingface/diffusers` | Primary model equations, tensor mapping, and component pipeline semantics | Apache-2.0 |
| `.references/sglang-omni` | `sgl-project/sglang-omni` | End-to-end request behavior, autoregressive generation, chunking, scheduler, and waveform assembly | Apache-2.0 |

Both checkouts are shallow, blob-filtered, and sparse. Their working trees contain
only the Music 3 source, tests, documentation, conversion code, and license needed
for the port. They remain normal Git checkouts, so they can be advanced without
changing the sibling H3 project:

```sh
git -C .references/diffusers pull --ff-only
git -C .references/diffusers sparse-checkout reapply

git -C .references/sglang-omni pull --ff-only
git -C .references/sglang-omni sparse-checkout reapply
```

After updating either checkout, audit relevant source changes and rerun the
pure-MLX regression suite before changing the revision record below.

## Authoritative external sources

These sources do not need another full local checkout:

| Source | Role |
|---|---|
| `MiniMax-AI/MiniMax-Music3` | Official architecture, prompt contract, supported-runtime instructions, and caption-rewriter skill |
| `MiniMaxAI/MiniMax-Music3` on Hugging Face | Authoritative configurations, tokenizer, weights, component index, model card, and model license |
| `mlx-video/.references/mlx-vlm` | Existing local MLX-native Qwen3 precedent; consult in place rather than copying it |

The MiniMax GitHub repository describes inference and points users to
SGLang-Omni, but it does not contain the numerical model implementation. The
official Hugging Face `scripts/end_to_end/minimax_ttm_test.py` file is an HTTP
client for a running SGLang-Omni server, not a standalone inference pipeline.

ComfyUI has an independent GPL-3.0 Music 3 implementation. It may be inspected
online when an additional behavioral comparison is necessary, but it is not part
of the default local reference set and its implementation must not be copied into
this MIT-licensed project.

## Revision record

These are the exact sources used to establish the initial porting contract on
2026-08-14:

| Reference | Revision | Stars at retrieval |
|---|---|---:|
| Diffusers | `90b4e34e79a86ec5e7f2437634fe95ecd2108796` | 34,314 |
| SGLang-Omni | `68abc7eec59ff7b0dce484cb501ffbbb338f9e46` | 805 |
| MiniMax official GitHub | `945655064d59b98004dd70002e7eb5c8c6e11373` | 295 |
| MiniMax official Hugging Face | `fbdf52fbaaca799592917417eb05f1899f1255ec` | n/a |
| Existing local MLX Qwen3 | `c2fe301bb3f2` (`mlx-vlm`) | n/a |

Stars are recorded only as a repository-identity sanity check and will naturally
change. Revisions, not star counts, define source-conformance expectations.
