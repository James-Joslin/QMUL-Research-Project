# MuZero decision-method research

This master's research project compares alternatives to MuZero's Monte Carlo Tree Search:
**MCTS**, **heuristic beam search**, **genetic trajectory search**, and an **NES-trained
controller**. It builds on [MuZero-General](https://github.com/werner-duvaud/muzero-general)
by Werner Duvaud.

The code uses one Python package, shared training and replay logic, explicit decision
interfaces, typed YAML configuration, and a reproducible development environment. Historical
models and figures are retained. Corrected experiments are distinct from the original results;
see [behaviour changes](docs/behaviour-changes.md).

## Start with Docker

Install Docker with Compose v2. On Linux, set `LOCAL_UID` and `LOCAL_GID` in `.env` to the
values from `id -u` and `id -g` so generated files belong to you.

```sh
docker compose build dev
docker compose run --rm dev
```

Inside the shell:

```sh
pytest -m 'not gpu' -q
muzero-research train --config configs/smoke/mcts-gridworld.yaml --output runs/quickstart
muzero-research evaluate --checkpoint runs/quickstart/model.checkpoint --episodes 3 --output runs/quickstart/evaluation.csv
muzero-research plot runs/quickstart/evaluation.csv --output runs/quickstart/returns.png
```

The source is mounted for editing; the container environment lives in `/opt/venv`, separate
from any host `.venv`. Run outputs persist under `runs/`. VS Code can use **Dev Containers:
Reopen in Container** for the same environment.

## Native development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked --extra cpu --extra analysis --extra tuning --extra diagnostics
uv run --locked --extra cpu --extra analysis --extra tuning --extra diagnostics pytest -m 'not gpu' -q
uv run --locked --extra cpu muzero-research train --config configs/smoke/beam-lunarlander.yaml --output runs/beam-demo
```

Python 3.12 and exact dependencies are recorded in the lockfile. Linux containers are the
reference environment; the CPU code and Ray integration checks also run on Windows.
Commands do not start workers or training during module import.

Repeat the required extras with `uv run`: it synchronises the selected environment and may
remove optional dependencies omitted from that invocation. Alternatively, use `.venv/bin/`
commands on Linux or `.venv/Scripts/` commands on Windows after syncing.

## Configurations and tasks

| Presets | Purpose |
|---|---|
| `configs/smoke/` | Small runs for all four methods on both primary environments |
| `configs/research/` | Dissertation hyperparameters with corrected runtime/environment conventions |
| `configs/legacy/` | Original architecture, observation ordering, and reward scaling for sample models |
| `configs/optional/` | Board games, CartPole, Atari, and OpenSpiel |

GridWorld is MiniGrid's random-start empty 6×6 task with three actions. New image observations
are channel-first. `lunarlander` is the custom task: fixed terrain, no engine dispersion,
random initial force with magnitude parameter 1000, and training rewards divided by three.
`lunarlander_standard` is a separate Gymnasium benchmark. Training and evaluation use the
same configured task and report both raw and scaled returns.

Research presets have the original large budgets. Start with a smoke preset before a long run.

## Historical models

All eight models are under `artifacts/checkpoints/`. Original files remain unchanged; the
manifest records their hashes and matching profiles. For example:

```sh
uv run --locked --extra cpu muzero-research convert-checkpoint artifacts/checkpoints/mcts/lunarlander/model.checkpoint converted-checkpoints/mcts-lunarlander.checkpoint --config configs/legacy/mcts-lunarlander.yaml
uv run --locked --extra cpu muzero-research evaluate --checkpoint converted-checkpoints/mcts-lunarlander.checkpoint --episodes 1 --output runs/historical-evaluation.csv
```

Converted files embed configuration and provenance. Conversion never overwrites its source
or an existing destination. Files outside the verified bundle require `--trust-legacy` if
their historical pickle cannot load in weights-only mode.

See [sample-model details](docs/sample-models.md). Original figures and available NPZ metrics
are under `artifacts/`; the full historical training dataset is not included.

## GPU and tools

```sh
docker compose --profile gpu build gpu
docker compose --profile gpu run --rm gpu muzero-research train --config configs/smoke/nes-lunarlander.yaml --device cuda --output runs/gpu-demo
docker compose --profile metrics up tensorboard
```

The GPU profile needs an NVIDIA-compatible Docker host and driver. TensorBoard is available
at `http://localhost:6006`. Native CUDA development selects `--extra cuda` instead of
`--extra cpu`; these variants are mutually exclusive.

The CLI also provides `tune`, `diagnose`, and optional `import-database` commands.
See the [experiment guide](docs/experiments.md), [architecture](docs/architecture.md),
[code index](docs/code-index.md), and [contributor guide](CONTRIBUTING.md).
The [validation record](docs/validation.md) lists completed checks and remaining hardware checks.

## Attribution

Preserve [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) when redistributing.

- Schrittwieser et al. (2020), [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://doi.org/10.1038/s41586-020-03051-4).
- Salimans et al. (2017), [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864).
- Werner Duvaud, [MuZero-General](https://github.com/werner-duvaud/muzero-general).
