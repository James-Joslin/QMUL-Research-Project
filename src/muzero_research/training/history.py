"""Episode data and observation stacking, independent of tree implementations."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    observations: list = field(default_factory=list)
    actions: list[int] = field(default_factory=lambda: [0])
    rewards: list[float] = field(default_factory=lambda: [0.0])
    raw_rewards: list[float] = field(default_factory=lambda: [0.0])
    players: list[int] = field(default_factory=list)
    policies: list = field(default_factory=list)
    root_values: list[float] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    final_value: float = 0.0
    reanalysed_values: list[float] | None = None
    priorities: object = None
    decision_seconds: list[float] = field(default_factory=list)

    def stacked(self, index, count, action_space_size):
        index %= len(self.observations)
        observation = self.observations[index]
        chunks = [observation]
        for previous in range(index - count, index):
            if previous >= 0:
                chunks.extend(
                    (
                        self.observations[previous],
                        np.full_like(
                            observation[:1], self.actions[previous + 1] / action_space_size
                        ),
                    )
                )
            else:
                chunks.extend((np.zeros_like(observation), np.zeros_like(observation[:1])))
        return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)

    def to_dict(self):
        return {
            "observations": np.asarray(self.observations).tolist(),
            "actions": self.actions,
            "rewards": self.rewards,
            "raw_rewards": self.raw_rewards,
            "players": self.players,
            "policies": np.asarray(self.policies).tolist(),
            "root_values": self.root_values,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "final_value": self.final_value,
            "reanalysed_values": self.reanalysed_values,
            "priorities": None if self.priorities is None else np.asarray(self.priorities).tolist(),
            "decision_seconds": self.decision_seconds,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["observations"] = [np.asarray(o, dtype=np.float32) for o in data["observations"]]
        data["policies"] = [np.asarray(p, dtype=np.float32) for p in data["policies"]]
        if data["priorities"] is not None:
            data["priorities"] = np.asarray(data["priorities"], dtype=np.float64)
        return cls(**data)
