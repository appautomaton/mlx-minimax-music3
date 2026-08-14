# Changelog

All notable project changes are documented here.

## 0.0.1a0 - Unreleased

- Establish the pure-MLX project contract.
- Add architecture, porting, weight, and reference documentation.
- Implement the checkpoint-native tokenizer and pure-MLX model pipeline.
- Add strict dense conversion and selective affine-q8 conversion with manifests.
- Add phase-scoped model residency with MLX, process-footprint, and swap checks.
- Add native 44.1 kHz stereo PCM16 WAV output and the public Python API.
- Add guarded instrumental prompt construction to prevent tag-only conditioning.
- Align autoregressive seed derivation, c0 column order, and Gumbel-max sampling
  with the SGLang reference.
- Correct the acoustic carry window from 86 to 172 latent frames.
- Mark selective q8 as experimental after long-sequence listening quality failed.
- Add package metadata, unit tests, and public-tree validation.
- Add a PyPI trusted-publishing workflow for GitHub pre-releases.
