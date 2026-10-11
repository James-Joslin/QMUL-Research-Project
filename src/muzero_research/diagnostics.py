"""Inspect predictions and decisions without coupling search algorithms to plotting."""

import json
from pathlib import Path

import numpy as np
import torch

from .decision import make_strategy
from .decision.common import initial, recurrent
from .environments import make_game
from .training.history import Episode


def diagnose(experiment, output, *, graph=False):
    config = experiment.config
    game = make_game(config.environment)
    experiment.model.eval()
    try:
        history = Episode(observations=[game.reset(seed=config.seed)])
        observation = history.stacked(
            0, config.environment.stacked_observations, len(config.environment.action_space)
        )
        legal = game.legal_actions()
        strategy = make_strategy(config, controller=experiment.controller)
        decision = strategy.decide(
            experiment.model,
            observation,
            legal,
            game.to_play(),
            training=False,
            temperature=0,
            rng=np.random.default_rng(config.seed),
        )
        with torch.no_grad():
            value, reward, _, state = initial(
                experiment.model, observation, config.network.support_size
            )
            children = []
            for action in legal:
                predicted_value, predicted_reward, _, _ = recurrent(
                    experiment.model, state, action, config.network.support_size
                )
                children.append(
                    {"action": action, "reward": predicted_reward, "value": predicted_value}
                )
        data = {
            "environment": config.environment.name,
            "method": config.decision.method,
            "selected_action": decision.action,
            "policy_target": decision.policy.tolist(),
            "root_value": decision.root_value,
            "initial_value": value,
            "initial_reward": reward,
            "diagnostics": decision.diagnostics,
            "one_step_predictions": children,
        }
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        if graph:
            from graphviz import Digraph

            drawing = Digraph("one-step-model")
            drawing.node("root", f"Initial value {value:.3f}")
            for child in children:
                drawing.node(
                    str(child["action"]),
                    f"Reward {child['reward']:.3f}\nValue {child['value']:.3f}",
                )
                drawing.edge("root", str(child["action"]), label=str(child["action"]))
            drawing.save(str(path.with_suffix(".dot")))
        return data
    finally:
        game.close()
