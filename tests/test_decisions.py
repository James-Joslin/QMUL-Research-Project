import numpy as np
import pytest
import torch

from muzero_research.config import preset
from muzero_research.decision import make_strategy
from muzero_research.decision.common import result
from muzero_research.decision.mcts import Node, backup
from muzero_research.decision.nes import es_update


@pytest.mark.parametrize("method", ["mcts", "beam", "genetic", "nes"])
@pytest.mark.parametrize("training", [False, True])
def test_decision_contract_and_legal_mask(method, training, toy_model, rng):
    config = preset(method=method, smoke=True)
    config.decision.simulations = 40
    strategy = make_strategy(config)
    decision = strategy.decide(
        toy_model,
        np.zeros((3, 7, 7)),
        [0, 2],
        0,
        training=training,
        temperature=1 if training else 0,
        rng=rng,
    )
    decision.validate(config.environment.action_space, [0, 2])
    assert decision.policy[1] == 0
    assert isinstance(decision.policy, np.ndarray)
    if method in {"mcts", "beam"} and not training:
        assert decision.action == 2


@pytest.mark.parametrize("method", ["mcts", "beam", "genetic", "nes"])
def test_evaluation_has_no_exploration_noise(method, toy_model):
    config = preset(method=method, smoke=True)
    strategy = make_strategy(config)
    observation = np.zeros((3, 7, 7))
    first = strategy.decide(
        toy_model,
        observation,
        [0, 1, 2],
        0,
        training=False,
        temperature=0,
        rng=np.random.default_rng(42),
    )
    config.decision.exploration_fraction = 1.0
    second = strategy.decide(
        toy_model,
        observation,
        [0, 1, 2],
        0,
        training=False,
        temperature=0,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(first.policy, second.policy)
    assert first.action == second.action


def test_discounted_backup():
    root, leaf = Node(player=0), Node(reward=2.0, player=0)
    backup([root, leaf], 4.0, 0, 0.5, (0,))
    assert leaf.value == 4
    assert root.value == 4
    root, leaf = Node(player=0), Node(player=1)
    backup([root, leaf], 4.0, 1, 0.5, (0, 1))
    assert root.value == -2


def test_zero_visit_fallback_and_temperature(rng):
    decision = result((0, 1, 2), [0, 2], [0, 0], 0, temperature=0, rng=rng)
    np.testing.assert_array_equal(decision.policy, [0.5, 0, 0.5])
    with pytest.raises(ValueError, match="Temperature"):
        result((0, 1), [0, 1], [1, 1], 0, temperature=-1, rng=rng)


def test_nes_estimator_uses_unit_noise_and_sigma():
    noise = np.array([[1.0, -1.0], [-1.0, 1.0]])
    fitness = np.array([3.0, 1.0])
    updated = es_update(np.array([2.0, 2.0]), noise, fitness, 0.1, 0.5)
    np.testing.assert_allclose(updated, [2.2, 1.8])
    with pytest.raises(ValueError):
        es_update(np.zeros(2), noise, fitness, 0.1, 0)


def test_controller_vector_roundtrip():
    from muzero_research.networks.controller import Controller

    controller = Controller(8, 4, 3, hidden_layers=1)
    vector = controller.vector()
    controller.set_vector(torch.zeros_like(vector))
    controller.set_vector(vector)
    torch.testing.assert_close(controller.vector(), vector)
    with pytest.raises(ValueError):
        controller.set_vector(vector[:2])
