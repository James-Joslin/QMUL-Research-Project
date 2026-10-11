"""Typed, serialisable experiment configuration; no environment imports or side effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvironmentConfig:
    name: str = "gridworld"
    observation_shape: tuple[int, int, int] = (3, 7, 7)
    action_space: tuple[int, ...] = (0, 1, 2)
    players: tuple[int, ...] = (0,)
    observation_layout: str = "CHW"
    reward_scale: float = 1.0
    max_steps: int = 15
    stacked_observations: int = 0


@dataclass
class NetworkConfig:
    kind: str = "fullyconnected"
    encoding_size: int = 8
    support_size: int = 10
    fc_representation_layers: list[int] = field(default_factory=list)
    fc_dynamics_layers: list[int] = field(default_factory=lambda: [16])
    fc_reward_layers: list[int] = field(default_factory=lambda: [16])
    fc_value_layers: list[int] = field(default_factory=lambda: [16])
    fc_policy_layers: list[int] = field(default_factory=lambda: [16])
    blocks: int = 1
    channels: int = 16
    reduced_channels_reward: int = 2
    reduced_channels_value: int = 2
    reduced_channels_policy: int = 2
    resnet_fc_reward_layers: list[int] = field(default_factory=list)
    resnet_fc_value_layers: list[int] = field(default_factory=list)
    resnet_fc_policy_layers: list[int] = field(default_factory=list)
    downsample: bool | str = False


@dataclass
class TrainingConfig:
    steps: int = 30_000
    batch_size: int = 128
    replay_capacity: int = 5000
    unroll_steps: int = 10
    td_steps: int = 20
    discount: float = 0.997
    optimizer: str = "Adam"
    learning_rate: float = 0.005
    lr_decay_rate: float = 1.0
    lr_decay_steps: int = 1000
    weight_decay: float = 1e-4
    momentum: float = 0.9
    value_loss_weight: float = 1.0
    prioritized_replay: bool = False
    priority_alpha: float = 0.5
    reanalyse: bool = False
    checkpoint_interval: int = 10
    temperature_threshold: int | None = None


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    workers: int = 1
    fitness_workers: int = 1
    cpus: int = 2
    population_backend: str = "local"
    startup_timeout: float = 120.0
    ray_object_store_bytes: int | None = None
    torch_threads: int = 1


@dataclass
class DecisionConfig:
    method: str = "mcts"
    simulations: int = 20
    dirichlet_alpha: float = 0.25
    exploration_fraction: float = 0.25
    pb_c_base: float = 19652
    pb_c_init: float = 1.25
    max_depth: int = 12
    beam_width: int = 2
    reward_heuristic_weight: float = 1.0
    value_heuristic_weight: float = 1.0
    population_size: int = 10
    generations: int = 5
    horizon: int = 15
    tournament_size: int = 6
    mutation_rate: float = 0.2
    controller_width: int = 32
    controller_hidden_layers: int = 0
    noise_std: float = 0.1
    es_learning_rate: float = 0.1
    noise_decay: float = 1.0
    es_lr_decay: float = 1.0


@dataclass
class ExperimentConfig:
    seed: int = 0
    output_dir: str = "runs"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def validate(self) -> None:
        e, n, t, r, d = self.environment, self.network, self.training, self.runtime, self.decision
        if d.method not in {"mcts", "beam", "genetic", "nes"}:
            raise ValueError(f"Unknown decision method: {d.method}")
        if n.kind not in {"fullyconnected", "resnet"}:
            raise ValueError(f"Unknown network: {n.kind}")
        if not e.action_space or e.action_space != tuple(range(len(e.action_space))):
            raise ValueError("Action IDs must be contiguous and start at zero")
        if e.players not in {(0,), (0, 1)}:
            raise ValueError("Only one-player or alternating two-player games are supported")
        if len(e.players) > 1 and d.method != "mcts":
            raise ValueError("The alternative decision methods support single-player tasks")
        if e.observation_layout not in {"CHW", "legacy_HWC"}:
            raise ValueError("Observation layout must be CHW or legacy_HWC")
        if n.kind == "resnet" and e.observation_layout != "CHW":
            raise ValueError("Residual networks require channel-first observations")
        if d.method == "nes" and n.kind != "fullyconnected":
            raise ValueError("NES requires a fully connected latent vector")
        positive = {
            "max_steps": e.max_steps,
            "support_size": n.support_size,
            "encoding_size": n.encoding_size,
            "batch_size": t.batch_size,
            "replay_capacity": t.replay_capacity,
            "td_steps": t.td_steps,
            "unroll_steps": t.unroll_steps,
            "checkpoint_interval": t.checkpoint_interval,
            "workers": r.workers,
            "fitness_workers": r.fitness_workers,
            "cpus": r.cpus,
            "torch_threads": r.torch_threads,
            "simulations": d.simulations,
            "beam_width": d.beam_width,
            "max_depth": d.max_depth,
            "horizon": d.horizon,
            "generations": d.generations,
            "controller_width": d.controller_width,
        }
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in positive.values()):
            raise ValueError(f"Positive integer budgets required: {positive}")
        if (
            any(not isinstance(x, int) or x <= 0 for x in e.observation_shape)
            or len(e.observation_shape) != 3
        ):
            raise ValueError("An observation must have three positive dimensions")
        if t.steps < 0 or e.stacked_observations < 0 or d.controller_hidden_layers < 0:
            raise ValueError("Steps, stacking, and hidden-layer counts must be nonnegative")
        if d.population_size < 2 or not 1 <= d.tournament_size <= d.population_size:
            raise ValueError(
                "Population must contain at least two candidates and cover tournament_size"
            )
        if (
            not 0 <= d.mutation_rate <= 1
            or not 0 <= d.exploration_fraction <= 1
            or not 0 <= t.discount <= 1
        ):
            raise ValueError("Mutation, exploration, and discount must lie in [0, 1]")
        import math

        positive_floats = (
            e.reward_scale,
            t.learning_rate,
            t.lr_decay_steps,
            t.lr_decay_rate,
            r.startup_timeout,
            d.noise_std,
            d.es_learning_rate,
            d.noise_decay,
            d.es_lr_decay,
            d.dirichlet_alpha,
            d.pb_c_base,
        )
        if any(not math.isfinite(v) or v <= 0 for v in positive_floats):
            raise ValueError("Scales, rates, timeouts, and noise must be finite and positive")
        if r.device not in {"cpu", "cuda"} or r.population_backend not in {"local", "ray"}:
            raise ValueError("Use cpu/cuda devices and local/ray population evaluation")
        if r.workers > r.cpus:
            raise ValueError("runtime.cpus must cover all self-play workers")
        if (
            d.method == "nes"
            and r.population_backend == "ray"
            and r.workers > 1
            and r.workers >= r.cpus
        ):
            raise ValueError("Reserve at least one additional CPU for NES fitness tasks")
        if not all(
            math.isfinite(value) and value >= 0
            for value in (
                t.weight_decay,
                t.value_loss_weight,
                t.priority_alpha,
                d.pb_c_init,
                d.reward_heuristic_weight,
                d.value_heuristic_weight,
            )
        ):
            raise ValueError(
                "Loss, priority, UCB, and heuristic weights must be finite and nonnegative"
            )
        if t.optimizer not in {"Adam", "SGD"}:
            raise ValueError("Optimizer must be Adam or SGD")

    def to_dict(self) -> dict[str, Any]:
        # YAML/JSON roundtrips turn tuples into lists; normalise both representations.
        import json

        return json.loads(json.dumps(asdict(self)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        data = dict(data)
        if unknown := set(data) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown experiment keys: {sorted(unknown)}")
        for key, constructor in (
            ("environment", EnvironmentConfig),
            ("network", NetworkConfig),
            ("training", TrainingConfig),
            ("runtime", RuntimeConfig),
            ("decision", DecisionConfig),
        ):
            section = dict(data.get(key, {}))
            if unknown := set(section) - {f.name for f in fields(constructor)}:
                raise ValueError(f"Unknown {key} keys: {sorted(unknown)}")
            if key == "environment":
                for sequence in ("observation_shape", "action_space", "players"):
                    if sequence in section:
                        section[sequence] = tuple(section[sequence])
            data[key] = constructor(**section)
        result = cls(**data)
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a YAML mapping")
        return cls.from_dict(data)


def preset(
    game: str = "gridworld", method: str = "mcts", *, smoke: bool = False
) -> ExperimentConfig:
    config = ExperimentConfig()
    config.environment.name = game
    config.decision.method = method
    if game in {"lunarlander", "lunarlander_standard"}:
        config.environment.observation_shape = (1, 1, 8)
        config.environment.action_space = (0, 1, 2, 3)
        config.environment.max_steps = 700
        config.environment.reward_scale = 1 / 3 if game == "lunarlander" else 1.0
        config.training.discount = 0.999
        config.network.encoding_size = 10
        for name in (
            "fc_dynamics_layers",
            "fc_reward_layers",
            "fc_value_layers",
            "fc_policy_layers",
        ):
            setattr(config.network, name, [64])
        config.decision.simulations = 50
        config.decision.max_depth = 5
        config.decision.population_size = 12
        config.decision.tournament_size = 8
        config.decision.generations = 4
        config.decision.horizon = 35
    if method == "nes":
        config.decision.population_size = 120 if game == "gridworld" else 80
        config.decision.generations = 1
        config.decision.horizon = 15
    if smoke:
        config.training.steps = 3
        config.training.batch_size = 4
        config.training.unroll_steps = 2
        config.training.td_steps = 3
        config.training.checkpoint_interval = 1
        config.environment.max_steps = 8
        config.decision.simulations = 4
        config.decision.max_depth = 2
        config.decision.horizon = 2
        config.decision.generations = 1
        config.decision.population_size = 4
        config.decision.tournament_size = 2
    config.validate()
    return config
