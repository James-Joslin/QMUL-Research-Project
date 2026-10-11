# Behaviour changes

Historical models and figures are preserved. New training runs are corrected experiments.

| Area | Original behaviour | Corrected behaviour |
|---|---|---|
| Framework | Four framework copies | One shared core with decision strategies |
| LunarLander | Custom training task, standard inference task | Same configured task; standard benchmark separately named |
| GridWorld | HWC described as CHW | CHW for new runs; legacy HWC profiles retained |
| Episode end | Single `done`, max moves treated as terminal | Termination/truncation separate; truncation bootstraps |
| Evaluation | Root Dirichlet noise enabled | Noise disabled; temperature zero |
| Timing | Included environment step; asynchronous GPU work | Decision-only timing, CUDA synchronised |
| NES observations | Unstacked self-play input | Consistent stacking |
| NES update | Sigma-scaled noise divided by sigma | Unit Gaussian noise with estimator divided by sigma |
| NES targets | Torch probabilities in synthetic nodes | Detached NumPy targets, legal masking |
| NES decay | Config decay fields unused | Applied and saved; original defaults preserve constant schedules |
| Boundary policy | Invented uniform supervision | Masked when no decision was observed |
| Gradient scaling | Hooks captured changing loop variables | Explicit scaling per unroll position |
| Priorities | Zero totals could create NaNs | Positive floor and finite checks |
| Genetic variants | Broken unused roulette/cycle routines | Tournament/uniform crossover, shared evaluator |
| Randomness | Python RNG inconsistently seeded | Python/NumPy/PyTorch/environment seeds recorded |
| Shutdown | Actor references discarded | Cancellation, actor cleanup, final checkpoint |
| Scheduling | Continuous asynchronous actors | Fixed-snapshot collection rounds |
| Tuning | Interactive Nevergrad/Ray paths | One Ray Tune workflow with isolated trials |
| Analytics | Embedded credential and fixed paths | File arguments; optional database URL from environment |

Beam search retains the dissertation's heuristic trajectory method: expand top-prior children,
continue the best immediate reward/value heuristic, and back up trajectories. It is not a
conventional beam frontier. Genetic/NES ranking retains the intentionally undiscounted sum
of imagined rewards plus final value. Tree backups and replay retain configured discounting.

NES uses `epsilon ~ N(0, I)`, candidates `theta + sigma * epsilon`, and update
`alpha / (N * sigma) * sum(fitness * epsilon)`. The toy optimisation example uses a zero-mean
fitness baseline; the research controller retains raw fitness. See [Salimans et al.](https://arxiv.org/abs/1703.03864).

Model prediction fixtures were captured using the original networks. Lander physics matches
a seeded original 64-step trajectory. Reset/step interfaces and rendering have changed;
the retained research physics equations have not.
