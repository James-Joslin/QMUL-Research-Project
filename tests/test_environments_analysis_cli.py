from pathlib import Path

import numpy as np
import pytest

from muzero_research.cli import main
from muzero_research.config import ExperimentConfig, preset
from muzero_research.environments import make_game

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["gridworld", "lunarlander", "lunarlander_standard"])
def test_seeded_environment_actions_and_rgb(name):
    config = preset(name, smoke=True)
    game = make_game(config.environment, render_mode="rgb_array")
    try:
        first = game.reset(seed=12)
        first_transition = game.step(0)
        second = game.reset(seed=12)
        second_transition = game.step(0)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(
            first_transition.observation, second_transition.observation, atol=1e-7
        )
        image = game.render()
        assert image.ndim == 3 and image.shape[2] == 3
    finally:
        game.close()


def test_research_lander_preserves_baseline_physics():
    from muzero_research.environments.research_lander import DeterministicLunarLander

    with np.load(ROOT / "artifacts/baseline-predictions/lunarlander-physics.npz") as fixture:
        game = DeterministicLunarLander()
        try:
            observation, _ = game.reset(seed=int(fixture["seed"]))
            observations = [observation]
            rewards = []
            for index, action in enumerate(fixture["actions"]):
                observation, reward, terminated, truncated, _ = game.step(int(action))
                observations.append(observation)
                rewards.append(reward)
                assert terminated == fixture["done"][index]
                assert not truncated

            # Box2D platform builds accumulate small float32 differences. Compare
            # the full trajectory so failures report the maximum drift, not just
            # the first step over tolerance. Contact flags remain exact.
            observations = np.asarray(observations)
            np.testing.assert_allclose(
                observations[:, :6], fixture["observations"][:, :6], rtol=0, atol=1e-6
            )
            np.testing.assert_array_equal(observations[:, 6:], fixture["observations"][:, 6:])
            # Two norms and an angle, scaled by 100 at two consecutive steps,
            # amplify 1e-6 state drift by at most 200 * (2 * sqrt(2) + 1).
            rewards = np.asarray(rewards)
            np.testing.assert_allclose(rewards, fixture["rewards"], rtol=0, atol=1e-3)
            np.testing.assert_array_equal(
                rewards[fixture["done"]], fixture["rewards"][fixture["done"]]
            )
        finally:
            game.close()


@pytest.mark.parametrize(
    "name", ["cartpole", "connect4", "gomoku", "tictactoe", "twentyone", "simple_grid"]
)
def test_optional_games_are_lazy_and_playable(name):
    config = ExperimentConfig.load(ROOT / f"configs/optional/{name}.yaml")
    game = make_game(config.environment)
    try:
        observation = game.reset(seed=1)
        assert observation.shape == config.environment.observation_shape
        transition = game.step(game.legal_actions()[0])
        assert np.isfinite(transition.reward)
    finally:
        game.close()


def test_cli_training_evaluation_diagnostics_and_plots(tmp_path):
    config_path = ROOT / "configs/smoke/beam-gridworld.yaml"
    output = tmp_path / "run"
    assert (
        main(
            [
                "train",
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--steps",
                "1",
                "--no-tensorboard",
            ]
        )
        == 0
    )
    checkpoint = output / "model.checkpoint"
    evaluation = tmp_path / "evaluation.npz"
    assert (
        main(
            [
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--episodes",
                "2",
                "--output",
                str(evaluation),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "diagnose",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(tmp_path / "diagnosis.json"),
                "--graph",
            ]
        )
        == 0
    )
    assert (tmp_path / "diagnosis.dot").exists()
    assert main(["plot", str(evaluation), "--output", str(tmp_path / "returns.png")]) == 0
    assert (tmp_path / "returns.png").stat().st_size > 0


def test_ragged_historical_npz_plotting(tmp_path):
    from muzero_research.analysis.results import plot_results

    first, second = tmp_path / "mcts.npz", tmp_path / "beam.npz"
    np.savez(first, np.array([1, 2]), np.array([0.1, 0.2]))
    np.savez(second, np.array([1, 2, 3]), np.array([0.2, 0.3, 0.4]))
    rows = plot_results([first, second], tmp_path / "plot.png")
    assert sorted(row["episodes"] for row in rows) == [2, 3]


def test_database_adapter_requires_configuration(monkeypatch):
    from muzero_research.analysis.database import import_table

    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL"):
        import_table("training", "unused.csv")
