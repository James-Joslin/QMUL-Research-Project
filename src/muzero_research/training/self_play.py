"""Shared episode collection used by training and evaluation."""

import random
import time

import numpy as np
import torch

from ..decision.common import initial, model_device
from ..environments import make_game
from .history import Episode


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def play_episode(
    config,
    model,
    strategy,
    *,
    seed,
    training,
    cancelled=None,
    render_mode=None,
    controller_optimizer=None,
    frames=None,
):
    seed_everything(seed, config.runtime.torch_threads)
    rng = np.random.default_rng(seed)
    model.eval()
    episode = Episode()
    game = make_game(config.environment, render_mode=render_mode)
    try:
        episode.observations.append(game.reset(seed=seed))
        episode.players.append(game.to_play())
        for step in range(config.environment.max_steps):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("Episode collection cancelled")
            stacked = episode.stacked(
                -1, config.environment.stacked_observations, len(config.environment.action_space)
            )
            if controller_optimizer is not None:
                controller_optimizer.optimise(model, stacked, rng)
            if frames is not None:
                frame = game.render()
                if frame is not None:
                    frames.append(frame)
            temperature = 1.0 if training else 0.0
            if (
                training
                and config.training.temperature_threshold is not None
                and step >= config.training.temperature_threshold
            ):
                temperature = 0.0
            if model_device(model).type == "cuda":
                torch.cuda.synchronize(model_device(model))
            started = time.perf_counter()
            decision = strategy.decide(
                model,
                stacked,
                game.legal_actions(),
                game.to_play(),
                training=training,
                temperature=temperature,
                rng=rng,
            )
            if model_device(model).type == "cuda":
                torch.cuda.synchronize(model_device(model))
            episode.decision_seconds.append(time.perf_counter() - started)
            decision.validate(config.environment.action_space, game.legal_actions())
            transition = game.step(decision.action)
            episode.policies.append(decision.policy.copy())
            episode.root_values.append(decision.root_value)
            episode.actions.append(decision.action)
            episode.observations.append(transition.observation)
            episode.rewards.append(transition.reward)
            episode.raw_rewards.append(transition.raw_reward)
            episode.players.append(game.to_play())
            episode.terminated = transition.terminated
            episode.truncated = transition.truncated or (
                step + 1 == config.environment.max_steps and not transition.terminated
            )
            if episode.terminated or episode.truncated:
                if episode.truncated and not episode.terminated:
                    final = episode.stacked(
                        -1,
                        config.environment.stacked_observations,
                        len(config.environment.action_space),
                    )
                    with torch.no_grad():
                        episode.final_value = initial(model, final, config.network.support_size)[0]
                break
        return episode
    finally:
        game.close()
