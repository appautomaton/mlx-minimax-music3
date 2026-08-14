# Testing

Every pytest test is weightless: tests never use the Official Music 3 checkpoint,
network access, downloaded tokenizer assets, or other local model files. The suite
is split by component boundary and execution cost. A test belongs to the lowest
tier that can verify the behavior without weakening its contract.

| Tier | Location | Contract | Default cadence |
| --- | --- | --- | --- |
| Unit | `tests/unit/` | One module or behavior in isolation, using small arrays, generated inputs, or narrow stubs | During implementation, then on every push, pull request, and release verification |
| Weightless integration | `tests/integration/` | Multiple real components using runtime-generated synthetic checkpoints and persisted metadata contracts | After affected unit tests, then on every push, pull request, and release verification |

Run the default gates in order so a cheap, focused failure stops before the
cross-component checks:

```sh
uv run pytest -q tests/unit
uv run pytest -q tests/integration
```

Plain `uv run pytest -q` discovers the same two default tiers. GitHub CI keeps
them as separate steps for clear failure attribution without running any test
twice. The workflow runs automatically for pull requests and pushes to `main`,
and it can also be started manually.

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

Official checkpoints may be consulted outside pytest when initially calibrating
or intentionally revising one of these persisted contracts. Only the minimal,
reviewable metadata or numerical reference vectors belong in the test suite. A
fixture update must explain the intended contract change and must never be an
automatic response to a failing golden assertion.

Listening validation with complete dense or quantized weights is a separate
release-quality activity. It must not be represented as a pytest pass/fail check
or added to the default CI suite.
