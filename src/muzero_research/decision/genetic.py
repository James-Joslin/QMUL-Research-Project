"""Tournament/ uniform-crossover genetic trajectory search."""

from dataclasses import dataclass

import numpy as np
import torch

from ..contracts import DecisionStrategy
from ..runtime.fitness import LocalFitnessEvaluator
from .common import initial, recurrent, result
from .trajectory import TrajectoryNode


@dataclass
class TrajectoryObjective:
    model: object
    state: object
    support_size: int

    @torch.no_grad()
    def __call__(self, actions):
        state, rewards, value = self.state, [], 0.0
        for action in actions:
            value, reward, _, state = recurrent(self.model, state, int(action), self.support_size)
            rewards.append(reward)
        # Preserve the dissertation's intentionally undiscounted ranking objective.
        return sum(rewards) + value


class GeneticSearch(DecisionStrategy):
    def __init__(self, config, evaluator=None):
        self.config = config
        self.evaluator = evaluator or LocalFitnessEvaluator()

    @torch.no_grad()
    def decide(self, model, observation, legal_actions, to_play, *, training, temperature, rng):
        c, e = self.config.decision, self.config.environment
        if not legal_actions or not set(legal_actions).issubset(e.action_space):
            raise ValueError("Invalid root legal actions")
        value, reward, _, state = initial(model, observation, self.config.network.support_size)
        population = rng.choice(e.action_space, size=(c.population_size, c.horizon))
        population[:, 0] = rng.choice(legal_actions, size=c.population_size)
        objective = TrajectoryObjective(model, state, self.config.network.support_size)
        for _ in range(c.generations):
            fitness = self.evaluator.evaluate(objective, population)
            survivors = []
            for _ in population:
                entrants = rng.choice(c.population_size, c.tournament_size, replace=False)
                survivors.append(population[entrants[np.argmax(fitness[entrants])]])
            children = []
            for _ in population:
                parent_ids = rng.choice(len(survivors), 2, replace=False)
                child = np.where(
                    rng.integers(0, 2, c.horizon).astype(bool),
                    survivors[parent_ids[0]],
                    survivors[parent_ids[1]],
                ).copy()
                positions = rng.choice(c.horizon, int(c.horizon * c.mutation_rate), replace=False)
                mutation_probs = rng.dirichlet(np.full(len(e.action_space), c.dirichlet_alpha))
                for position in positions:
                    if position == 0:
                        legal_probabilities = mutation_probs[legal_actions]
                        child[position] = rng.choice(
                            legal_actions, p=legal_probabilities / legal_probabilities.sum()
                        )
                    else:
                        child[position] = rng.choice(e.action_space, p=mutation_probs)
                children.append(child)
            population = np.asarray(children)
        root = TrajectoryNode(state, reward, value)
        for policy in population:
            node = root
            for action in policy:
                action = int(action)
                if action not in node.children:
                    v, r, _, next_state = recurrent(
                        model, node.state, action, self.config.network.support_size
                    )
                    node.children[action] = TrajectoryNode(next_state, r, v, parent=node)
                node = node.children[action]
            node.backup(node.predicted_value, self.config.training.discount)
        return result(
            e.action_space,
            legal_actions,
            [root.children[a].visits if a in root.children else 0 for a in legal_actions],
            root.value,
            temperature=temperature,
            rng=rng,
            diagnostics={
                "population_size": c.population_size,
                "generations": c.generations,
                "max_tree_depth": c.horizon,
            },
        )
