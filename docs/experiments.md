# Experiments

CLI options override the YAML config. New/converted checkpoints provide stored configuration
when `--config` is omitted; old checkpoints need a matching legacy profile. Preserve the layout,
architecture, and reward scaling for old models.

## Train and resume

```sh
muzero-research train --config configs/research/genetic-gridworld.yaml --seed 42 --output runs/genetic-42
muzero-research train --checkpoint runs/genetic-42/model.checkpoint --steps 35000
```

`--steps` is the total update count, including restored updates. Runs write resolved YAML,
CSV metrics, TensorBoard events, and atomic checkpoint replacements. Use a new directory for
a new run. Resume restores weights, optimizer, replay, RNG, and NES optimiser state.

Set `runtime.workers`/`runtime.cpus` for parallel collection. Set `population_backend: ray`
and `fitness_workers` for driver-side parallel population scoring. Leave scoring local for
small workloads. Increase `startup_timeout` for long collection batches. Ctrl-C saves final state.

## Evaluate and inspect

```sh
muzero-research evaluate --checkpoint runs/genetic-42/model.checkpoint --episodes 20 --output runs/genetic-42/evaluation.csv
muzero-research evaluate --checkpoint runs/genetic-42/model.checkpoint --render-dir runs/genetic-42/render
muzero-research diagnose --checkpoint runs/genetic-42/model.checkpoint --graph --output runs/genetic-42/diagnosis.json
```

Evaluation seeds are `config.seed + episode_index`. Results contain method, task, seed, steps,
raw/scaled returns, boundary flags, and mean decision-only timing. NPZ output has named arrays
and metadata. `--render-dir` saves GIFs headlessly; native `--render` opens a display. Optional
board games keep text rendering; saved RGB rendering supports primary Gymnasium tasks.

Diagnostics report initial predictions, decision policy/value, method diagnostics, and legal
one-step reward/value predictions. `--graph` saves these one-step predictions as DOT, not the
full internal search tree.

## Tune and plot

```sh
muzero-research tune --config configs/smoke/beam-lunarlander.yaml --samples 4 --output runs/beam-tuning
muzero-research plot runs/genetic-42/evaluation.csv --output runs/genetic-42/returns.png
muzero-research plot runs/genetic-42/training.csv --kind training --window 20 --output runs/genetic-42/training.png
```

Install tuning/analysis extras. Tuning samples learning rate and root-noise parameters, then
ranks trials by mean raw return across three evaluation episodes. Each trial owns its output.
The noise range retains the existing beam search range. Original Nevergrad routines remain
available in Git history.

Plotting accepts multiple files and unequal episode counts. Historical NPZ `arr_0` rewards/
`arr_1` timings and CSV `total_reward`/`domain`/`id` names are supported. Summaries use actual
episode/update coordinates for integrals. Missing historical data is not synthesised.

## Optional games and database

Optional MCTS presets cover CartPole, Connect4, Gomoku, TicTacToe, TwentyOne, SimpleGrid,
Atari/Breakout, and OpenSpiel TicTacToe. Board games are smoke-checked; primary research tasks
have the full training/evaluation matrix. Alternative methods reject multiplayer; NES rejects
residual latents.

Atari needs the `games` extra and ALE ROMs. `ALE/Breakout-v5` is distinct from old Breakout-v4;
old Atari checkpoint compatibility is not claimed. OpenSpiel requires the `spiel` extra on Linux.

For optional database export, install `database`, privately set `DATABASE_URL`, then run:

```sh
muzero-research import-database mcts_gridworld --output data/mcts_gridworld.csv
```

No database container is needed. Exact historical figure reproduction requires the complete
training dataset formerly linked from a private university repository.
