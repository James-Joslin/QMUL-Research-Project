"""Shared inference and policy utilities, independent of Ray and episode storage."""

from __future__ import annotations

import numpy as np
import torch

from ..contracts import DecisionResult
from ..networks.models import support_to_scalar


def model_device(model):
    return next(model.parameters(), torch.empty(0)).device


def initial(model, observation, support_size):
    tensor = torch.as_tensor(
        np.ascontiguousarray(observation), dtype=torch.float32, device=model_device(model)
    ).unsqueeze(0)
    value, reward, logits, state = model.initial_inference(tensor)
    return (
        support_to_scalar(value, support_size).item(),
        support_to_scalar(reward, support_size).item(),
        logits,
        state,
    )


def recurrent(model, state, action, support_size):
    value, reward, logits, state = model.recurrent_inference(
        state, torch.tensor([[action]], device=state.device)
    )
    return (
        support_to_scalar(value, support_size).item(),
        support_to_scalar(reward, support_size).item(),
        logits,
        state,
    )


def priors(logits, actions, *, training, config, rng):
    probabilities = (
        torch.softmax(logits[0, actions], dim=0).detach().cpu().numpy().astype(np.float64)
    )
    if training:
        noise = rng.dirichlet(np.full(len(actions), config.dirichlet_alpha))
        probabilities = (
            1 - config.exploration_fraction
        ) * probabilities + config.exploration_fraction * noise
    return probabilities


def result(
    action_space, legal, weights, root_value, *, temperature, rng, diagnostics=None, action=None
):
    if not legal or not set(legal).issubset(action_space):
        raise ValueError("Legal actions must be a nonempty subset of the configured action space")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(legal),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Decision weights must be finite and nonnegative")
    if weights.sum() <= 0:
        weights = np.ones(len(legal))
    probabilities = weights / weights.sum()
    policy = np.zeros(len(action_space), dtype=np.float32)
    for a, p in zip(legal, probabilities, strict=True):
        policy[action_space.index(a)] = p
    if action is None:
        if temperature == 0:
            action = legal[int(np.argmax(probabilities))]
        elif temperature == float("inf"):
            action = int(rng.choice(legal))
        elif np.isfinite(temperature) and temperature > 0:
            logs = np.full(len(probabilities), -np.inf)
            positive = probabilities > 0
            logs[positive] = np.log(probabilities[positive]) / temperature
            sampling = np.exp(logs - logs.max())
            action = int(rng.choice(legal, p=sampling / sampling.sum()))
        else:
            raise ValueError("Temperature must be nonnegative")
    decision = DecisionResult(action, policy, float(root_value), diagnostics or {})
    decision.validate(action_space, legal)
    return decision
