"""Thin Ray actor wrappers; worker algorithms remain testable Python classes."""

import copy

from ..decision import make_strategy
from ..decision.nes import build_controller
from ..networks import build_network
from ..training.self_play import play_episode, seed_everything


class SelfPlayWorker:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        # Keep GPU allocation with the trainer; collection actors use CPU snapshots.
        self.config.runtime.device = "cpu"
        self.config.runtime.population_backend = "local"
        seed_everything(config.seed, config.runtime.torch_threads)
        self.model = build_network(self.config)
        self.controller = build_controller(self.config) if config.decision.method == "nes" else None
        self.strategy = make_strategy(self.config, controller=self.controller)

    def collect(self, weights, controller_weights, seed):
        self.model.set_weights(weights)
        if self.controller is not None:
            self.controller.load_state_dict(controller_weights)
        return play_episode(self.config, self.model, self.strategy, seed=seed, training=True)


def make_self_play_actor():
    import ray

    return ray.remote(num_cpus=1)(SelfPlayWorker)
