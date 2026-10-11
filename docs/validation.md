# Validation record

Validated on Windows with CPython 3.12.15 and the locked CPU environment on 11 October 2026.

| Check | Result |
|---|---|
| Locked CPU installation, analysis, tuning, diagnostics, and development tools | Passed |
| Complete pytest suite | 61 passed, 4 NVIDIA tests skipped |
| All four methods on GridWorld and custom LunarLander | Three training updates per combination, finite losses, checkpoints, and evaluation |
| All eight historical checkpoints | Original hashes unchanged; captured initial/recurrent predictions match; conversion and bounded evaluation pass |
| Custom LunarLander physics | Matches the captured original seeded 64-step trajectory |
| Save/resume | MCTS and NES resumed weights match uninterrupted training |
| Ray | Local/remote fitness equivalence, real-model genetic/NES evaluation, parallel self-play, cancellation/failure cleanup, borrowed cluster ownership, and one tuning trial pass |
| CLI and analysis | Training, evaluation, diagnostics, CSV/NPZ plotting pass |
| Optional games | CartPole and five retained board/card adapters pass smoke checks |
| Ruff lint and formatting | Passed |
| Incremental type checks | Passed for configuration and abstract contracts |
| Wheel and source distribution | Build successfully; installed wheel imports and constructs an experiment outside the editable package; no toolchains/caches included |
| Compose configuration including GPU and metrics profiles | Validates with the official Compose CLI |

Ray tests used an authentication token in an ignored local file and access to local worker
sockets. No authentication token is included in the repository. No Ray service processes
remained after the suite finished.

Docker Desktop/a Docker daemon and NVIDIA hardware were unavailable on this machine. The
container build, fresh-container CPU workflow, and GPU runtime checks therefore remain
unverified locally. CI defines the container build and CPU workflow; the NVIDIA tests are
available for a compatible host. Optional Atari and OpenSpiel integrations have not been
runtime validated here.

These short runs verify implementation behaviour; they do not establish research performance
or reproduce the historical training curves. Full figure reproduction requires the missing
historical source data.
