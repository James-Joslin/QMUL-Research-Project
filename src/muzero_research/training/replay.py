"""Replay targets with genuine terminal and time-limit truncation semantics."""

import numpy as np

from .history import Episode


def target_value(episode: Episode, index: int, td_steps: int, discount: float) -> float:
    length = len(episode.root_values)
    if index > length:
        return 0.0
    bootstrap = min(index + td_steps, length)
    roots = (
        episode.reanalysed_values if episode.reanalysed_values is not None else episode.root_values
    )
    value = (
        roots[bootstrap]
        if bootstrap < length
        else episode.final_value
        if episode.truncated and not episode.terminated
        else 0.0
    )
    if episode.players[bootstrap] != episode.players[index]:
        value = -value
    value *= discount ** (bootstrap - index)
    for reward_index in range(index + 1, bootstrap + 1):
        reward = episode.rewards[reward_index]
        if episode.players[index] != episode.players[reward_index - 1]:
            reward = -reward
        value += reward * discount ** (reward_index - index - 1)
    return float(value)


class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.episodes: list[Episode] = []
        self.played_games = 0
        self.played_steps = 0

    def add(self, episode):
        if not episode.root_values:
            raise ValueError("Cannot store an empty episode")
        t = self.config.training
        episode.priorities = np.maximum(
            np.asarray(
                [
                    abs(v - target_value(episode, i, t.td_steps, t.discount)) ** t.priority_alpha
                    for i, v in enumerate(episode.root_values)
                ]
            ),
            1e-8,
        )
        self.episodes.append(episode)
        self.episodes = self.episodes[-t.replay_capacity :]
        self.played_games += 1
        self.played_steps += len(episode.root_values)

    def sample(self, rng):
        if not self.episodes:
            raise ValueError("Replay is empty")
        t, e = self.config.training, self.config.environment
        observations, actions, values, rewards, policies = [], [], [], [], []
        masks, policy_masks, weights, indices, scales = [], [], [], [], []
        game_probabilities = np.ones(len(self.episodes))
        if t.prioritized_replay:
            game_probabilities = np.asarray([np.max(ep.priorities) for ep in self.episodes])
        game_probabilities = game_probabilities / game_probabilities.sum()
        total_samples = sum(len(ep.root_values) for ep in self.episodes)
        for _ in range(t.batch_size):
            game_id = int(rng.choice(len(self.episodes), p=game_probabilities))
            episode = self.episodes[game_id]
            length = len(episode.root_values)
            position_probabilities = (
                episode.priorities / np.sum(episode.priorities)
                if t.prioritized_replay
                else np.full(length, 1 / length)
            )
            start = int(rng.choice(length, p=position_probabilities))
            observations.append(episode.stacked(start, e.stacked_observations, len(e.action_space)))
            indices.append((game_id, start))
            weights.append(
                1 / (total_samples * game_probabilities[game_id] * position_probabilities[start])
                if t.prioritized_replay
                else 1.0
            )
            sample_values, sample_rewards, sample_policies, sample_actions = [], [], [], []
            mask, policy_mask = [], []
            for position in range(start, start + t.unroll_steps + 1):
                sample_values.append(target_value(episode, position, t.td_steps, t.discount))
                sample_rewards.append(episode.rewards[position] if position <= length else 0.0)
                sample_policies.append(
                    episode.policies[position]
                    if position < length
                    else np.full(len(e.action_space), 1 / len(e.action_space))
                )
                sample_actions.append(episode.actions[position] if position <= length else 0)
                mask.append(float(position <= length))
                policy_mask.append(float(position < length))
            values.append(sample_values)
            rewards.append(sample_rewards)
            policies.append(sample_policies)
            actions.append(sample_actions)
            masks.append(mask)
            policy_masks.append(policy_mask)
            scales.append(max(1, min(t.unroll_steps, length - start)))
        weight_array = np.asarray(weights, dtype=np.float32)
        return {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "values": np.asarray(values, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "policies": np.asarray(policies, dtype=np.float32),
            "masks": np.asarray(masks, dtype=np.float32),
            "policy_masks": np.asarray(policy_masks, dtype=np.float32),
            "weights": weight_array / weight_array.max(),
            "scales": np.asarray(scales, dtype=np.float32),
            "indices": indices,
        }

    def update_priorities(self, indices, priorities):
        for (game_id, start), values in zip(indices, priorities, strict=True):
            episode = self.episodes[game_id]
            count = min(len(values), len(episode.root_values) - start)
            episode.priorities[start : start + count] = np.maximum(values[:count], 1e-8)

    def state_dict(self):
        return {
            "episodes": [ep.to_dict() for ep in self.episodes],
            "played_games": self.played_games,
            "played_steps": self.played_steps,
        }

    def load_state_dict(self, state):
        self.episodes = [Episode.from_dict(ep) for ep in state["episodes"]]
        self.played_games = state["played_games"]
        self.played_steps = state["played_steps"]
