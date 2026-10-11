"""Internal trajectory-tree aggregation shared by the two trajectory search methods."""

from dataclasses import dataclass, field


@dataclass
class TrajectoryNode:
    state: object = None
    reward: float = 0.0
    predicted_value: float = 0.0
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict = field(default_factory=dict)
    parent: object = None

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits else 0.0

    def backup(self, value, discount):
        node = self
        while node is not None:
            node.value_sum += value
            node.visits += 1
            value = node.reward + discount * value
            node = node.parent
