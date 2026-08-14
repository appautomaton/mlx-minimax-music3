# Testing

The test suite is split by dependency boundary and execution cost. A test belongs
to the lowest tier that can verify the behavior without weakening its contract.

| Tier | Location | Contract | Default cadence |
| --- | --- | --- | --- |
| Unit | `tests/unit/` | One module or behavior in isolation; no network or local model assets | During implementation, then on every push, pull request, and release verification |
| Weightless integration | `tests/integration/` | Multiple real components using runtime-generated synthetic checkpoints | After affected unit tests, then on every push, pull request, and release verification |
| Full checkpoint | `tests/full_checkpoint/` | Contracts that require the complete official Music 3 source checkpoint | Explicit local validation before releases that change configs, mapping, tokenization, model topology, or checkpoint conversion |

Run the default gates in order so a cheap, focused failure stops before the
cross-component checks:

```sh
uv run pytest -q tests/unit
uv run pytest -q tests/integration
```

Plain `uv run pytest -q` discovers the same two default tiers. GitHub CI keeps
them as separate steps for clear failure attribution without running any test
twice.

Full-checkpoint tests are intentionally excluded from default discovery. They do
not download weights and never run in GitHub-hosted CI. By default they read
`weights/bf16/MiniMax-Music3`; an explicit path may be supplied for an existing
checkout:

```sh
MUSIC3_SOURCE_CHECKPOINT=/path/to/MiniMax-Music3 \
  uv run pytest -q -m full_checkpoint tests/full_checkpoint
```

The golden integration fixture is versioned in
`tests/fixtures/music3_golden_v1.json`. It persists the miniature model contract,
inference inputs, expected numerical outputs, topology digests, and tolerances.
Its deterministic dense and selective-q8 SafeTensors files are materialized only
inside pytest's temporary directory.

`tests/fixtures/music3_official_schema_v1.json` separately persists the Official
Music 3 configs, all 982 mapped-parameter topology digests, and representative
source-to-MLX mapping cases. It contains metadata only, so the default integration
tier continues to verify the real model contract after the local source weights
are removed.

Listening validation with complete dense or quantized weights is a separate
release-quality activity. It must not be represented as a pytest pass/fail check
or added to the default CI suite.
