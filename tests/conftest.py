import numpy as np
import pytest
import torch

from muzero_research.contracts import WorldModel
from muzero_research.networks.models import scalar_to_support


class ToyModel(WorldModel):
    """Known rewards, zero terminal predictions, and uniform action priors."""

    def __init__(self, actions=3, support=10):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.actions = actions
        self.support = support

    def categorical(self, value):
        probabilities = scalar_to_support(
            torch.tensor([[value]], device=self.anchor.device), self.support
        )[:, 0]
        return probabilities.log()

    def initial_inference(self, observation):
        return (
            self.categorical(0.0),
            self.categorical(0.0),
            torch.zeros(1, self.actions, device=self.anchor.device),
            torch.zeros(1, 8, device=self.anchor.device),
        )

    def recurrent_inference(self, encoded_state, action):
        reward = float(action.item() + 1)
        return (
            self.categorical(0.0),
            self.categorical(reward),
            torch.zeros(1, self.actions, device=self.anchor.device),
            encoded_state + 0.1,
        )


@pytest.fixture
def toy_model():
    return ToyModel()


@pytest.fixture
def rng():
    return np.random.default_rng(4)
