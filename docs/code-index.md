# Code index and migration map

The baseline AST scan indexed **84 Python modules, 27,971 lines, and 46 byte-distinct files**
at the commit in `baseline/code-index.json`. It records live definitions, line numbers, imports,
entry points, duplicate groups, original config expressions, and checkpoint hashes without
importing or executing the modules.

`code-index.json` describes the current implementation. Regenerate it with:

```sh
python scripts/index_code.py
```

| Previous subsystem | Current subsystem |
|---|---|
| Four orchestrators | `experiment`, `cli`, `tuning`, `runtime` |
| Four trainers/replay buffers | `training.trainer`, `training.replay` |
| Model copies | `networks.models`, explicit factory, `networks.controller` |
| Algorithms in self-play | `decision.mcts`, `beam`, `genetic`, `nes` |
| Self-play/history copies | `training.self_play`, `training.history` |
| Controller trainer | `NESOptimizer` plus separate optimisation episode |
| Mutable game configs | Dataclasses and YAML presets |
| Copied lander physics | One `environments.research_lander` |
| Other games | Lazy optional bridge and modern CartPole/ALE adapters |
| Inference scripts | `evaluate` and shared result format |
| Analysis scripts | `analysis.results`, optional `analysis.database` |
| Confirming-NES | `examples/optimise_functions.py` with shared estimator |
| Models, figures, metrics | `artifacts/` with provenance manifest |

Original code is recoverable from the baseline Git commit. Caches, backups, obsolete duplicate
modules, and machine-specific paths are removed. The historical dependency freeze is retained
as context, not used as the development installer.
