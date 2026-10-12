# Historical sample models

Eight original checkpoints are retained byte-for-byte. `artifacts/manifest.json` records
full hashes, original locations, training steps, config profiles, and prediction fixtures.

| Method | GridWorld updates | LunarLander updates | Legacy profiles |
|---|---:|---:|---|
| MCTS | 34,999 | 277,599 | `mcts-gridworld.yaml`, `mcts-lunarlander.yaml` |
| Beam | 34,999 | 319,999 | `beam-gridworld.yaml`, `beam-lunarlander.yaml` |
| Genetic | 34,999 | 319,999 | `genetic-gridworld.yaml`, `genetic-lunarlander.yaml` |
| NES | 29,249 | 142,109 | `nes-gridworld.yaml`, `nes-lunarlander.yaml` |

Checkpoints are located at `artifacts/checkpoints/<method>/<game>/model.checkpoint`; profiles
are under `configs/legacy/`. These recorded update counts describe individual sample files,
not a claim that the methods received equal training budgets.

GridWorld samples expect HWC-shaped `(7,7,3)` observations and their original flattened order.
LunarLander samples expect `(1,1,8)` observations, the custom research physics, and rewards
scaled by one third. Preserve architecture, ordering, and scaling when loading these weights.
Original NES controllers used no additional hidden layers despite a misleading old config field.

Convert into a new destination with `convert-checkpoint`. Network weights, optimizer state,
controller weights, and original scalar checkpoint fields are preserved; metadata is added.
Legacy files contain no complete replay/RNG state, so resuming them cannot reproduce the exact
old training continuation. New checkpoints store the complete supported resume state.

Reference NPZ fixtures contain initial/recurrent network outputs captured before migration.
The tests compare these outputs and converted weights and execute bounded sample evaluation.
`lunarlander-physics.npz` captures the seeded original physics trajectory independently.
The physics regression keeps tight observation checks and exact termination checks; rewards
use an absolute tolerance of `1e-4` to accommodate float32 rounding across Box2D platform
builds, amplified by the reward shaping coefficients. The fixture is retained unchanged.

Historical inference NPZ files and figures are preserved under `artifacts/historical-results/`
and `artifacts/figures/`. Those inference files used the old standard LunarLander evaluation
workflow and include environment-step timing, so their numbers must not be pooled with corrected
research-task evaluation or decision-only timings. Original tree figures are retained separately.
