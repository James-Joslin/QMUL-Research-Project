# Architecture

`Experiment` owns the world model, optional controller, trainer, replay, RNG, and cancellation
state. Construction does not start Ray. Each training round collects episodes from a fixed
weight snapshot, stores them, samples replay, and performs a MuZero update.

| Boundary | Interface | Implementations |
|---|---|---|
| Environments | `Game` → `Transition` | Gymnasium/MiniGrid, research lander, optional game bridge |
| Learned dynamics | `WorldModel` | Fully connected and residual networks |
| Decisions | `DecisionStrategy` → `DecisionResult` | MCTS, heuristic beam, genetic, NES policy |
| Candidate scoring | `FitnessEvaluator` | Local and Ray evaluation |

Network construction preserves original state-dictionary names, including `DataParallel`
wrappers. Search methods share inference/policy utilities without imposing a tree interface
on replay or NES. Action IDs are contiguous from zero, matching the dynamics encoding.

## Training data

`play_episode` collects observations, actions, raw/scaled rewards, policy targets, root values,
player IDs, timings, and episode boundaries. `Episode` begins with an observation and dummy
action/reward. Decisions and replay use the same observation-stacking implementation.

Replay computes discounted n-step targets. Truncation bootstraps from the last observed state;
terminal states bootstrap zero. Policy loss is masked where no decision exists. Padding past
the final observed state is masked. Priorities have a positive floor.

`Trainer` unrolls dynamics, trains categorical heads, scales hidden-state/unroll gradients, and
updates optional replay priorities. Reanalysis refreshes model value estimates and truncation
bootstraps.

NES optimisation is separate from policy consumption. An optimisation episode improves the
controller through imagined trajectories and is not stored in replay. `NESPolicy` then collects
replay episodes using controller probabilities as targets. Its root value is mean one-step Q
over legal actions, matching the original synthetic-tree calculation. Optimiser state resumes
alongside controller weights.

## Runtime

One collection worker and local population scoring require no Ray startup. Multiple collection
actors use CPU snapshots; GPU allocation stays with the trainer. Driver-side Ray population
tasks support genetic and NES scoring. Collection actors score populations locally to avoid
nested resource deadlocks. Parallel actors plus Ray NES scoring need an additional CPU.

`RayRuntime` cancels pending work and shuts down clusters it owns. Existing external clusters
remain running. Experiment cleanup kills its actors, closes evaluators, saves final state, and
closes TensorBoard. Worker failures propagate to the caller.

The corrected runner uses synchronised collection rounds rather than the original continuous
asynchronous scheduling. This makes snapshots and small runs reproducible; it does not claim
bit-for-bit equivalence to historical training.
