import numpy as np
import pytest

from muzero_research.config import preset
from muzero_research.training.history import Episode
from muzero_research.training.replay import ReplayBuffer, target_value


def episode(*, truncated=False):
    return Episode(
        observations=[np.ones((1, 1, 2)) * i for i in (1, 2, 3)],
        actions=[0, 1, 0],
        rewards=[0, 3, 4],
        raw_rewards=[0, 3, 4],
        players=[0, 0, 0],
        policies=[np.array([0.5, 0.5, 0]), np.array([0, 0.5, 0.5])],
        root_values=[1, 2],
        terminated=not truncated,
        truncated=truncated,
        final_value=5,
    )


def test_terminal_vs_truncation_bootstrap():
    assert target_value(episode(), 0, 3, 0.5) == 5
    assert target_value(episode(truncated=True), 0, 3, 0.5) == 6.25
    assert target_value(episode(), 2, 3, 0.5) == 0
    assert target_value(episode(truncated=True), 2, 3, 0.5) == 5
    assert target_value(episode(), 0, 1, 0.5) == 4
    assert target_value(episode(), 3, 3, 0.5) == 0


def test_stacking_history_actions_and_padding():
    history = episode()
    stacked = history.stacked(1, 2, 3)
    assert stacked.shape == (5, 1, 2)
    np.testing.assert_array_equal(stacked[0], [[2, 2]])
    np.testing.assert_array_equal(stacked[1:3], 0)
    np.testing.assert_array_equal(stacked[3], [[1, 1]])
    np.testing.assert_allclose(stacked[4], 1 / 3)


@pytest.mark.parametrize("prioritized", [False, True])
def test_replay_masks_capacity_and_serialisation(prioritized, rng):
    config = preset(smoke=True)
    config.training.prioritized_replay = prioritized
    config.training.replay_capacity = 1
    replay = ReplayBuffer(config)
    replay.add(episode())
    replay.add(episode(truncated=True))
    assert len(replay.episodes) == 1
    batch = replay.sample(rng)
    assert np.isfinite(batch["weights"]).all()
    for (_, start), policies, mask in zip(
        batch["indices"], batch["policies"], batch["policy_masks"], strict=True
    ):
        for i, valid in enumerate(mask):
            if start + i >= 2:
                assert valid == 0
            assert np.isclose(policies[i].sum(), 1)
    restored = ReplayBuffer(config)
    restored.load_state_dict(replay.state_dict())
    assert restored.played_steps == 4
    np.testing.assert_array_equal(restored.episodes[0].policies, replay.episodes[0].policies)


def test_multiplayer_target_orientation():
    history = episode()
    history.players = [0, 1, 0]
    assert target_value(history, 0, 3, 0.5) == 1
