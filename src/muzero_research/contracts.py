"""Public interfaces used by environments, models, and interchangeable decision methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class Transition:
    observation: NDArray
    reward: float
    raw_reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class Game(ABC):
    @abstractmethod
    def reset(self, *, seed: int | None = None) -> NDArray: ...

    @abstractmethod
    def step(self, action: int) -> Transition: ...

    @abstractmethod
    def legal_actions(self) -> list[int]: ...

    def to_play(self) -> int:
        return 0

    def render(self) -> NDArray | None:
        return None

    def close(self) -> None:
        return None


class WorldModel(ABC, torch.nn.Module):
    @abstractmethod
    def initial_inference(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def recurrent_inference(
        self, encoded_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def get_weights(self) -> dict[str, torch.Tensor]:
        return {key: tensor.detach().cpu().clone() for key, tensor in self.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(weights)


@dataclass(frozen=True)
class DecisionResult:
    action: int
    policy: NDArray[np.float32]
    root_value: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self, action_space: tuple[int, ...], legal_actions: list[int]) -> None:
        if self.action not in legal_actions:
            raise ValueError("Selected action must be legal")
        if self.policy.shape != (len(action_space),) or not np.isfinite(self.policy).all():
            raise ValueError("Policy must be a finite vector covering the configured actions")
        if (self.policy < 0).any() or not np.isclose(self.policy.sum(), 1.0, atol=1e-6):
            raise ValueError("Policy probabilities must be nonnegative and sum to one")
        if any(
            self.policy[i] != 0
            for i, action in enumerate(action_space)
            if action not in legal_actions
        ):
            raise ValueError("Illegal actions must have zero target probability")
        if not np.isfinite(self.root_value):
            raise ValueError("Root value must be finite")


class DecisionStrategy(ABC):
    @abstractmethod
    def decide(
        self,
        model: WorldModel,
        observation: NDArray,
        legal_actions: list[int],
        to_play: int,
        *,
        training: bool,
        temperature: float,
        rng: np.random.Generator,
    ) -> DecisionResult: ...


class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(self, objective: Any, candidates: NDArray) -> NDArray[np.float64]: ...

    def close(self) -> None:
        return None
