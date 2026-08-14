# Agent Guide

This file applies to the entire repository.

## Mission

`mlx-minimax-music3` is an independent, pure-MLX inference runtime for
MiniMax Music 3 on Apple silicon.

## Non-Negotiable Constraints

- Runtime inference must use MLX only. Do not add PyTorch, CUDA, or Triton
  execution paths under the MLX package.
- Code, tests, comments, commit messages, and public documentation must be
  written in clear English.
- Keep runtime inference separate from checkpoint inspection and conversion.
- Never commit model weights, tokenizer assets, generated audio, private lyrics,
  prompts, local references, or machine-specific state.
- `.references/` contains read-only upstream checkouts and is never packaged.
- Keep the runtime dependency budget minimal. Add a package only when runtime code
  imports it and the implementation demonstrates why the standard library or MLX
  is insufficient.
- Do not publish placeholder APIs that imply inference works before end-to-end
  waveform generation is validated.
- Preserve upstream license notices when code is directly adapted. ComfyUI is a
  behavioral reference only; do not copy GPL-licensed implementation code into
  this permissively licensed package.

## Start Here

Before changing implementation code, read:

1. [README.md](README.md) for the public project contract.
2. [docs/architecture.md](docs/architecture.md) for the planned runtime stages.
3. [docs/porting.md](docs/porting.md) for reference priority and validation.
4. [docs/weights.md](docs/weights.md) before changing loading or conversion.
5. [docs/references.md](docs/references.md) before consulting upstream code.

## Working Loop

Make the smallest coherent change and validate it with:

```sh
uv run ruff check .
uv run pytest -q
uv run python dev/check_public_tree.py
uv build --no-sources
```

Do not start full-checkpoint or long-form generation runs unless the task requires
them. Default tests must remain weightless.

## Release Contract

- Distribution: `mlx-minimax-music3`
- Import package: `mlx_minimax_music3`
- Version, Git tag `v{version}`, and GitHub pre-release state must agree.
- `.github/workflows/workflow.yml` is the trusted-publishing workflow for the
  PyPI `pypi` environment.
- Publishing is an external action and requires an explicit user request.
