# Porting plan

The goal is numerical and behavioral parity with the released MiniMax Music 3
checkpoint while keeping runtime inference pure MLX.

## Reference priority

1. **Official Hugging Face checkpoint** — authoritative configuration, tokenizer,
   weight names, generation configuration, model license, and expected artifacts.
2. **Hugging Face Diffusers** — primary source for model equations, tensor shapes,
   weight conversion, flow matching, and waveform decoding.
3. **SGLang-Omni** — source for end-to-end behavior, prompt construction,
   autoregressive caching, acoustic chunking, and production defaults.
4. **Existing MLX Qwen3 implementations** — source for MLX-native attention,
   rotary embeddings, key-value caching, and quantized linear layers.
5. **ComfyUI** — independent behavioral cross-check only. Its GPL-licensed code
   must not be copied into this MIT-licensed runtime.

The official `scripts/end_to_end/minimax_ttm_test.py` file is only an HTTP client
for a running SGLang-Omni server. It is not an inference implementation.

## Implementation slices

### 1. Checkpoint inspection and conversion

- Parse every component configuration without importing PyTorch or Diffusers.
- Read SafeTensors headers and inventory every tensor name, shape, dtype, shard,
  and byte range before allocating model arrays.
- Ingest only the checkpoint's componentized SafeTensors layout; do not download
  or depend on the duplicate legacy `.pth` layout.
- Stream one source shard at a time through explicit, versioned component
  mappings into an MLX-native dense checkpoint.
- Validate missing, unexpected, duplicate, and shape-mismatched tensors before
  writing a conversion manifest.
- Keep inspection and conversion under development tooling, outside the runtime
  package and dependency graph.

### 2. Prompt and tokenizer

- Reproduce the official chat template and special-token placement.
- Preserve lyrics and structured-caption boundaries exactly.
- Add golden token-ID fixtures for short public test inputs.

### 3. Global and local autoregressive models

- Port Qwen3 attention, rotary position embeddings, normalization, and cache use.
- Add the music-token embedding and global output head.
- Port the seven-step RVQ depth decoder.
- Validate one frame before implementing long generation.
- Keep language-model and RVQ weights at their published BF16 precision while
  evaluating logits, guidance, filtering, and probabilities in FP32.

### 4. Acoustic conditioning and flow matching

- Port hidden-state fusion and the condition encoder.
- Port one Flow Transformer block and validate intermediate tensors.
- Implement the Euler solver and classifier-free guidance behavior.
- Validate one acoustic window before overlap synthesis.
- Preserve the published FP32 acoustic path for the dense baseline.

### 5. Waveform decoding

- Port the Flow-VAE / DAC-style decoder.
- Validate shape, sample rate, channels, and short deterministic output.
- Add overlap and crossfade parity tests for long-form output.

### 6. End-to-end runtime

- Connect the phases with stage sessions and verified model release points.
- Use one pipeline implementation for dense and quantized checkpoint profiles.
- Reproduce the SGLang request-seed derivation, codebook-position schedule,
  narrowed c0 vocabulary order, and MurmurHash32 Gumbel-max sampler exactly.
- Add maximum-frame handling, progress events, and cancellation at safe frame,
  denoising-step, and chunk boundaries.
- Measure active, cached, and peak memory per stage and fail before allocation
  when a known checkpoint profile exceeds the configured budget.
- Emit generation metadata containing source revision, conversion manifest,
  checkpoint profile, seed schedule, frame count, chunk count, sample rate, and
  timing without recording private prompts or lyrics by default.
- Expose the public Python API and command-line interface only after a waveform
  smoke test succeeds.

### 7. Selective quantization

- Start from the validated MLX-native dense checkpoint, never from `.pth`.
- Quantize through an explicit allowlist of large linear modules.
- Keep published FP32 modules unchanged in the initial optimized profile.
- Keep norms, embeddings, sampling heads, convolutional waveform paths, and
  numerical integration out of the default quantization policy.
- Compare each quantized component and a multi-second token trajectory against
  dense MLX before accepting an end-to-end audio regression. One-frame waveform
  similarity is not sufficient for an autoregressive quantization policy.
- Store per-module quantization declarations and tensor digests in the checkpoint
  manifest; inference never performs conversion implicitly.

## Validation tiers

| Tier | Requirement | Purpose |
|---|---|---|
| Configuration | No weights | Catch layout and default-value mistakes |
| Tiny module | Synthetic weights | Catch transpose, normalization, and masking errors |
| Component parity | Local checkpoint | Compare autoregressive, DiT, and decoder stages |
| End-to-end smoke | Full checkpoint | Produce a valid deterministic WAV |
| Quality regression | Full checkpoint | Compare audio structure and prompt adherence |
| Quantized component | Dense and q8 checkpoints | Bound error introduced by each allowlisted module group |
| Memory lifecycle | Dense and q8 checkpoints | Verify stage teardown and peak-memory budgets |

Default CI must run the first two tiers without model downloads. PyTorch may be
used by separate development-only fixture generators, but it must never be a
runtime dependency or a requirement for normal tests.
