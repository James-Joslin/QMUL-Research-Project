"""Controller policy and a separate Gaussian evolution-strategy optimiser."""

import copy
from dataclasses import dataclass

import numpy as np
import torch

from ..contracts import DecisionStrategy
from ..networks.controller import Controller
from ..runtime.fitness import LocalFitnessEvaluator
from .common import initial, recurrent, result


def es_update(parameters, unit_noise, fitness, learning_rate, sigma):
    """Salimans et al. estimator: epsilon is N(0,I), not sigma-scaled noise."""
    if sigma <= 0 or len(unit_noise) != len(fitness) or len(fitness) == 0:
        raise ValueError("Positive sigma and matching nonempty candidates required")
    return parameters + learning_rate / (len(fitness) * sigma) * (
        fitness[:, None] * unit_noise
    ).sum(axis=0)


@dataclass
class ControllerObjective:
    model: object
    state: object
    controller: object
    horizon: int
    support_size: int

    @torch.no_grad()
    def __call__(self, parameters):
        device = self.state.device
        self.controller.to(device)
        self.controller.set_vector(torch.as_tensor(parameters, dtype=torch.float32, device=device))
        state, rewards, value = self.state, [], 0.0
        for _ in range(self.horizon):
            action = int(self.controller(state).argmax(dim=-1).item())
            value, reward, _, state = recurrent(self.model, state, action, self.support_size)
            rewards.append(reward)
        return sum(rewards) + value


class NESOptimizer:
    def __init__(self, config, controller, evaluator=None):
        self.config = config
        self.controller = controller
        self.evaluator = evaluator or LocalFitnessEvaluator()
        self.sigma = config.decision.noise_std
        self.learning_rate = config.decision.es_learning_rate
        self.updates = 0

    @torch.no_grad()
    def optimise(self, model, observation, rng):
        c = self.config.decision
        _, _, _, state = initial(model, observation, self.config.network.support_size)
        parameters = self.controller.vector().cpu().numpy().astype(np.float64)
        objective = ControllerObjective(
            model,
            state,
            copy.deepcopy(self.controller),
            c.horizon,
            self.config.network.support_size,
        )
        for _ in range(c.generations):
            noise = rng.standard_normal((c.population_size, len(parameters)))
            fitness = self.evaluator.evaluate(objective, parameters + self.sigma * noise)
            parameters = es_update(parameters, noise, fitness, self.learning_rate, self.sigma)
            if not np.isfinite(parameters).all():
                raise FloatingPointError("NES update produced nonfinite parameters")
        self.controller.set_vector(
            torch.as_tensor(
                parameters, dtype=torch.float32, device=next(self.controller.parameters()).device
            )
        )
        self.sigma *= c.noise_decay
        self.learning_rate *= c.es_lr_decay
        self.updates += 1

    def state_dict(self):
        return {"sigma": self.sigma, "learning_rate": self.learning_rate, "updates": self.updates}

    def load_state_dict(self, state):
        self.sigma = state["sigma"]
        self.learning_rate = state["learning_rate"]
        self.updates = state["updates"]


class NESPolicy(DecisionStrategy):
    def __init__(self, config, controller):
        self.config, self.controller = config, controller

    @torch.no_grad()
    def decide(self, model, observation, legal_actions, to_play, *, training, temperature, rng):
        e = self.config.environment
        if not legal_actions or not set(legal_actions).issubset(e.action_space):
            raise ValueError("Invalid root legal actions")
        _, _, _, state = initial(model, observation, self.config.network.support_size)
        self.controller.to(state.device).eval()
        probabilities = self.controller(state).flatten().cpu().numpy()[legal_actions]
        action = legal_actions[int(np.argmax(probabilities))]
        q_values = []
        for legal in legal_actions:
            value, reward, _, _ = recurrent(model, state, legal, self.config.network.support_size)
            q_values.append(reward + self.config.training.discount * value)
        # Match the original synthetic-tree root target: mean Q over legal actions.
        return result(
            e.action_space,
            legal_actions,
            probabilities,
            float(np.mean(q_values)),
            temperature=temperature,
            rng=rng,
            action=action,
            diagnostics={"controller_policy": True, "model_calls": len(legal_actions) + 1},
        )


def build_controller(config):
    c = config.decision
    return Controller(
        config.network.encoding_size,
        c.controller_width,
        len(config.environment.action_space),
        c.controller_hidden_layers,
    )
