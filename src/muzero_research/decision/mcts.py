"""MuZero MCTS, derived from MuZero-General (Werner Duvaud, MIT)."""

import math
from dataclasses import dataclass, field

import torch

from ..contracts import DecisionStrategy
from .common import initial, priors, recurrent, result


@dataclass
class Node:
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    player: int = 0
    state: torch.Tensor | None = None
    children: dict = field(default_factory=dict)

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits else 0.0


def backup(path, value, leaf_player, discount, players):
    for node in reversed(path):
        node.value_sum += value if len(players) == 1 or node.player == leaf_player else -value
        node.visits += 1
        reward = node.reward if len(players) == 1 or node.player != leaf_player else -node.reward
        value = reward + discount * value


class MCTS(DecisionStrategy):
    def __init__(self, config):
        self.config = config

    @torch.no_grad()
    def decide(self, model, observation, legal_actions, to_play, *, training, temperature, rng):
        c, e = self.config.decision, self.config.environment
        if not legal_actions or not set(legal_actions).issubset(e.action_space):
            raise ValueError("Invalid root legal actions")
        value, reward, logits, state = initial(model, observation, self.config.network.support_size)
        root = Node(reward=reward, player=to_play, state=state)
        root.children = {
            a: Node(prior=float(p))
            for a, p in zip(
                legal_actions,
                priors(logits, legal_actions, training=training, config=c, rng=rng),
                strict=True,
            )
        }
        minimum, maximum, max_depth = float("inf"), -float("inf"), 0
        discount = self.config.training.discount
        for _ in range(c.simulations):
            node, path, player, depth = root, [root], to_play, 0
            while node.children:
                scores = []
                actions = list(node.children)
                for child in node.children.values():
                    pb_c = (
                        (math.log((node.visits + c.pb_c_base + 1) / c.pb_c_base) + c.pb_c_init)
                        * math.sqrt(node.visits)
                        / (child.visits + 1)
                    )
                    q = child.reward + discount * (
                        child.value if len(e.players) == 1 else -child.value
                    )
                    if maximum > minimum:
                        q = (q - minimum) / (maximum - minimum)
                    scores.append(pb_c * child.prior + (q if child.visits else 0))
                best = max(scores)
                action = int(
                    rng.choice(
                        [a for a, score in zip(actions, scores, strict=True) if score == best]
                    )
                )
                parent, node = node, node.children[action]
                player = e.players[(e.players.index(player) + 1) % len(e.players)]
                path.append(node)
                depth += 1
            leaf_value, node.reward, leaf_logits, node.state = recurrent(
                model, parent.state, action, self.config.network.support_size
            )
            node.player = player
            probabilities = priors(
                leaf_logits, list(e.action_space), training=False, config=c, rng=rng
            )
            node.children = {
                a: Node(prior=float(p)) for a, p in zip(e.action_space, probabilities, strict=True)
            }
            backup(path, leaf_value, player, discount, e.players)
            for visited in path:
                q = visited.reward + discount * (
                    visited.value if len(e.players) == 1 else -visited.value
                )
                minimum, maximum = min(minimum, q), max(maximum, q)
            max_depth = max(max_depth, depth)
        return result(
            e.action_space,
            legal_actions,
            [root.children[a].visits for a in legal_actions],
            root.value,
            temperature=temperature,
            rng=rng,
            diagnostics={
                "max_tree_depth": max_depth,
                "root_predicted_value": value,
                "model_calls": c.simulations + 1,
            },
        )
