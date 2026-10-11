# Contributing

Use the Docker/Compose or locked Python 3.12 environment in the README. Infrastructure is
for development, without deployment or production services.

## Checks

After syncing the CPU environment with the analysis, tuning, and diagnostics extras:

```sh
uv run --locked --extra cpu ruff check src tests scripts examples
uv run --locked --extra cpu ruff format --check src tests scripts examples
uv run --locked --extra cpu mypy
uv run --locked --extra cpu --extra analysis --extra tuning --extra diagnostics pytest -m 'not gpu'
uv build --no-sources
```

Type checking currently covers configuration and public contracts; extend coverage for new
typed subsystems. Install optional hooks with `pre-commit install`. Dependency changes update
`pyproject.toml` and `uv.lock` together. Exported `requirements.txt` is generated from the CPU
lock, not edited manually.

Ray tests start local workers and sockets. Shared memory defaults to 2 GB in Compose and can
be changed with `RAY_SHM_SIZE`. Ray creates authentication tokens in the user's home directory;
restricted environments can set `RAY_AUTH_TOKEN_PATH` to an existing private token file. Never
commit tokens.

## Extension boundaries

Implement `DecisionStrategy.decide()` and register the method in the factory. Return a legal
action, a finite policy target covering configured action order, a finite root value, and
serialisable diagnostics. Policies must be normalised and zero for illegal root actions.
Keep trees private to algorithms and leave self-play and replay untouched.

Implement `Game` and return `Transition` with raw/scaled rewards, termination/truncation flags,
and the configured observation shape. Seed through `reset(seed=...)` and close resources.
Add contract tests and short training/evaluation checks for maintained tasks.

Fitness objectives must be serialisable and deterministic for fixed candidates/model weights.
Both evaluators must return scores in candidate order.

## Research changes

Document changes in `docs/behaviour-changes.md`. Keep sample-model hashes and prediction
fixtures. Do not relabel corrected experiments as dissertation results. New outputs belong
under ignored `runs/` directories.

Regenerate the current AST index with `python scripts/index_code.py`. The baseline index and
historical prediction/physics fixtures are immutable reference artifacts.
