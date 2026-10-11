"""One experiment orchestrator for all four decision methods."""

from __future__ import annotations

import copy
import csv
import json
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint, metadata, save_checkpoint
from .config import ExperimentConfig
from .decision import make_strategy
from .decision.common import initial
from .decision.nes import NESOptimizer, build_controller
from .networks import build_network
from .runtime import RayRuntime
from .runtime.fitness import make_evaluator
from .runtime.workers import make_self_play_actor
from .training.replay import ReplayBuffer
from .training.self_play import play_episode, seed_everything
from .training.trainer import Trainer


class Experiment:
    def __init__(self, config: ExperimentConfig):
        config.validate()
        self.config = copy.deepcopy(config)
        if config.runtime.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        seed_everything(config.seed, config.runtime.torch_threads)
        self.model = build_network(self.config).to(config.runtime.device)
        self.controller = (
            build_controller(self.config).to(config.runtime.device)
            if config.decision.method == "nes"
            else None
        )
        self.trainer = Trainer(self.model, self.config)
        self.replay = ReplayBuffer(self.config)
        self.rng = np.random.default_rng(config.seed)
        self.cancelled = threading.Event()
        self.nes_state = None
        self.last_metrics = {}
        self.output = Path(config.output_dir)

    @classmethod
    def from_checkpoint(cls, path, *, config=None, trust_legacy=False):
        checkpoint, saved_config = load_checkpoint(
            path, legacy_config=config, trust_legacy=trust_legacy
        )
        result = cls(config if config is not None else saved_config)
        result._restore(checkpoint)
        return result

    def _restore(self, checkpoint):
        self.model.set_weights(checkpoint["weights"])
        if self.controller is not None:
            if checkpoint.get("controller_weights") is None:
                raise ValueError("NES checkpoints must contain controller weights")
            self.controller.load_state_dict(checkpoint["controller_weights"])
        self.trainer.load_state_dict(
            {
                "step": checkpoint.get("training_step", 0),
                "optimizer": checkpoint.get("optimizer_state"),
            }
        )
        if "replay" in checkpoint:
            self.replay.load_state_dict(checkpoint["replay"])
        if "rng" in checkpoint:
            self.rng.bit_generator.state = checkpoint["rng"]
        self.nes_state = checkpoint.get("nes_state")

    def checkpoint(self):
        return {
            "metadata": metadata(self.config),
            "weights": self.model.get_weights(),
            "controller_weights": None if self.controller is None else self.controller.state_dict(),
            "optimizer_state": self.trainer.optimizer.state_dict(),
            "training_step": self.trainer.step,
            "replay": self.replay.state_dict(),
            "rng": self.rng.bit_generator.state,
            "nes_state": self.nes_state,
            "last_metrics": self.last_metrics,
        }

    def save(self, path=None):
        path = Path(path) if path is not None else self.output / "model.checkpoint"
        save_checkpoint(path, self.checkpoint())
        return path

    def cancel(self):
        self.cancelled.set()

    def _seed(self):
        return int(self.rng.integers(0, 2**31 - 1))

    def _reanalyse(self):
        self.model.eval()
        with torch.no_grad():
            for episode in self.replay.episodes:
                episode.reanalysed_values = [
                    initial(
                        self.model,
                        episode.stacked(
                            i,
                            self.config.environment.stacked_observations,
                            len(self.config.environment.action_space),
                        ),
                        self.config.network.support_size,
                    )[0]
                    for i in range(len(episode.root_values))
                ]
                if episode.truncated and not episode.terminated:
                    episode.final_value = initial(
                        self.model,
                        episode.stacked(
                            -1,
                            self.config.environment.stacked_observations,
                            len(self.config.environment.action_space),
                        ),
                        self.config.network.support_size,
                    )[0]

    def train(self, *, tensorboard=True):
        if (self.output / "training.csv").exists() and self.trainer.step == 0:
            raise FileExistsError(
                "Output contains an existing run; choose a new directory or resume explicitly"
            )
        self.output.mkdir(parents=True, exist_ok=True)
        self.config.save(self.output / "config.yaml")
        writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(self.output / "tensorboard"))
        needs_ray = (
            self.config.runtime.workers > 1 or self.config.runtime.population_backend == "ray"
        )
        runtime = RayRuntime(self.config.runtime) if needs_ray else nullcontext()
        workers, evaluator = [], None
        try:
            with runtime as active_runtime:
                evaluator = make_evaluator(self.config)
                strategy = make_strategy(
                    self.config, evaluator=evaluator, controller=self.controller
                )
                nes = (
                    NESOptimizer(self.config, self.controller, evaluator)
                    if self.controller is not None
                    else None
                )
                if nes is not None and self.nes_state is not None:
                    nes.load_state_dict(self.nes_state)
                if self.config.runtime.workers > 1:
                    actor = make_self_play_actor()
                    workers = [
                        actor.remote(self.config) for _ in range(self.config.runtime.workers)
                    ]
                try:
                    metrics_path = self.output / "training.csv"
                    columns = [
                        "training_step",
                        "total_loss",
                        "value_loss",
                        "reward_loss",
                        "policy_loss",
                        "learning_rate",
                        "played_games",
                        "played_steps",
                        "raw_return",
                        "scaled_return",
                        "seed",
                    ]
                    if metrics_path.exists() and self.trainer.step == 0:
                        raise FileExistsError(
                            "Output contains an existing run; choose a new directory or resume explicitly"
                        )
                    with metrics_path.open("a", newline="", encoding="utf-8") as stream:
                        metrics_writer = csv.DictWriter(stream, fieldnames=columns)
                        if stream.tell() == 0:
                            metrics_writer.writeheader()
                        while self.trainer.step < self.config.training.steps:
                            if self.cancelled.is_set():
                                raise InterruptedError("Training cancelled")
                            self.model.eval()
                            if nes is not None:
                                # A separate optimisation episode updates the controller, not replay.
                                play_episode(
                                    self.config,
                                    self.model,
                                    strategy,
                                    seed=self._seed(),
                                    training=True,
                                    cancelled=self.cancelled,
                                    controller_optimizer=nes,
                                )
                                self.nes_state = nes.state_dict()
                            if workers:
                                controller_weights = (
                                    None
                                    if self.controller is None
                                    else {
                                        key: value.detach().cpu()
                                        for key, value in self.controller.state_dict().items()
                                    }
                                )
                                references = [
                                    worker.collect.remote(
                                        self.model.get_weights(), controller_weights, self._seed()
                                    )
                                    for worker in workers
                                ]
                                episodes = active_runtime.collect(
                                    references, cancelled=self.cancelled
                                )
                            else:
                                episodes = [
                                    play_episode(
                                        self.config,
                                        self.model,
                                        strategy,
                                        seed=self._seed(),
                                        training=True,
                                        cancelled=self.cancelled,
                                    )
                                ]
                            for episode in episodes:
                                self.replay.add(episode)
                            if self.config.training.reanalyse:
                                self._reanalyse()
                            batch = self.replay.sample(self.rng)
                            priorities, metrics = self.trainer.update(batch)
                            if self.config.training.prioritized_replay:
                                self.replay.update_priorities(batch["indices"], priorities)
                            metrics.update(
                                {
                                    "played_games": self.replay.played_games,
                                    "played_steps": self.replay.played_steps,
                                    "raw_return": float(
                                        np.mean([sum(ep.raw_rewards) for ep in episodes])
                                    ),
                                    "scaled_return": float(
                                        np.mean([sum(ep.rewards) for ep in episodes])
                                    ),
                                    "seed": self.config.seed,
                                }
                            )
                            self.last_metrics = metrics
                            metrics_writer.writerow(metrics)
                            stream.flush()
                            if writer is not None:
                                for name, value in metrics.items():
                                    if name != "training_step":
                                        writer.add_scalar(name, value, self.trainer.step)
                            if self.trainer.step % self.config.training.checkpoint_interval == 0:
                                self.save()
                finally:
                    if evaluator is not None:
                        evaluator.close()
                    if workers:
                        import ray

                        for worker in workers:
                            ray.kill(worker, no_restart=True)
        except (KeyboardInterrupt, InterruptedError):
            self.cancelled.set()
            raise
        finally:
            self.save()
            if writer is not None:
                writer.close()
        return self.last_metrics

    def evaluate(self, episodes=1, *, render_mode=None, render_dir=None):
        if episodes < 1:
            raise ValueError("Evaluation needs at least one episode")
        config = copy.deepcopy(self.config)
        config.runtime.population_backend = "local"
        strategy = make_strategy(config, controller=self.controller)
        rows = []
        for index in range(episodes):
            frames = [] if render_dir is not None else None
            episode = play_episode(
                config,
                self.model,
                strategy,
                seed=config.seed + index,
                training=False,
                cancelled=self.cancelled,
                render_mode="rgb_array" if frames is not None else render_mode,
                frames=frames,
            )
            rows.append(
                {
                    "episode": index,
                    "seed": config.seed + index,
                    "method": config.decision.method,
                    "environment": config.environment.name,
                    "raw_return": sum(episode.raw_rewards),
                    "scaled_return": sum(episode.rewards),
                    "steps": len(episode.root_values),
                    "decision_seconds": float(np.mean(episode.decision_seconds)),
                    "terminated": episode.terminated,
                    "truncated": episode.truncated,
                }
            )
            if frames:
                from PIL import Image

                directory = Path(render_dir)
                directory.mkdir(parents=True, exist_ok=True)
                images = [Image.fromarray(frame) for frame in frames]
                images[0].save(
                    directory / f"episode-{index}.gif",
                    save_all=True,
                    append_images=images[1:],
                    duration=20,
                    loop=0,
                )
        return rows

    def write_evaluation(self, path, episodes=1, **options):
        rows = self.evaluate(episodes, **options)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npz":
            np.savez(
                path,
                rewards=np.asarray([row["raw_return"] for row in rows]),
                timings=np.asarray([row["decision_seconds"] for row in rows]),
                scaled_rewards=np.asarray([row["scaled_return"] for row in rows]),
                metadata=json.dumps(metadata(self.config)),
            )
        else:
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return rows
