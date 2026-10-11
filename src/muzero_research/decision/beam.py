"""The dissertation's heuristic trajectory search, not a conventional beam frontier."""

import torch

from ..contracts import DecisionStrategy
from .common import initial, priors, recurrent, result
from .trajectory import TrajectoryNode


class BeamSearch(DecisionStrategy):
    def __init__(self, config):
        self.config = config

    @torch.no_grad()
    def decide(self, model, observation, legal_actions, to_play, *, training, temperature, rng):
        c, e = self.config.decision, self.config.environment
        if not legal_actions or not set(legal_actions).issubset(e.action_space):
            raise ValueError("Invalid root legal actions")
        value, reward, logits, state = initial(model, observation, self.config.network.support_size)
        root = TrajectoryNode(state, reward, value)
        actions = legal_actions
        probabilities = priors(logits, actions, training=training, config=c, rng=rng)
        node = root
        calls = 1
        for _ in range(c.max_depth):
            node.children = {
                a: TrajectoryNode(prior=float(p), parent=node)
                for a, p in zip(actions, probabilities, strict=True)
            }
            chosen = sorted(actions, key=lambda a: node.children[a].prior, reverse=True)[
                : c.beam_width
            ]
            expanded = []
            for action in chosen:
                child = node.children[action]
                child.predicted_value, child.reward, child_logits, child.state = recurrent(
                    model, node.state, action, self.config.network.support_size
                )
                expanded.append((child, child_logits))
                calls += 1
            best, logits = max(
                expanded,
                key=lambda pair: (
                    c.reward_heuristic_weight * pair[0].reward
                    + c.value_heuristic_weight * pair[0].predicted_value
                ),
            )
            for child, _ in expanded:
                if child is not best:
                    child.backup(child.predicted_value, self.config.training.discount)
            node = best
            actions = list(e.action_space)
            probabilities = priors(logits, actions, training=False, config=c, rng=rng)
        node.backup(node.predicted_value, self.config.training.discount)
        return result(
            e.action_space,
            legal_actions,
            [root.children[a].visits for a in legal_actions],
            root.value,
            temperature=temperature,
            rng=rng,
            diagnostics={"max_tree_depth": c.max_depth, "model_calls": calls},
        )
