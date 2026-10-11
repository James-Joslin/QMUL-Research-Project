import json
from pathlib import Path

import numpy as np
import pytest
import torch

from muzero_research.checkpoints import convert_checkpoint
from muzero_research.config import ExperimentConfig, preset
from muzero_research.experiment import Experiment
from muzero_research.networks import build_network

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.parametrize("method", ["mcts", "beam", "genetic", "nes"])
@pytest.mark.parametrize("game", ["gridworld", "lunarlander"])
def test_primary_training_matrix(game, method, tmp_path):
    config = preset(game, method, smoke=True)
    config.output_dir = str(tmp_path / f"{method}-{game}")
    experiment = Experiment(config)
    metrics = experiment.train(tensorboard=False)
    assert metrics["training_step"] == 3
    assert np.isfinite(metrics["total_loss"])
    checkpoint = Path(config.output_dir) / "model.checkpoint"
    assert checkpoint.exists()
    restored = Experiment.from_checkpoint(checkpoint)
    rows = restored.write_evaluation(tmp_path / "evaluation.csv", episodes=2)
    assert len(rows) == 2
    assert all(np.isfinite(row["raw_return"]) for row in rows)
    if game == "lunarlander":
        for row in rows:
            assert row["scaled_return"] == pytest.approx(row["raw_return"] / 3)
    assert all(row["decision_seconds"] > 0 for row in rows)


@pytest.mark.parametrize(
    "record",
    json.loads((ROOT / "artifacts/manifest.json").read_text())["checkpoints"],
    ids=lambda record: record["config"],
)
def test_historical_checkpoint_predictions_and_conversion(record, tmp_path):
    config = ExperimentConfig.load(ROOT / record["config"])
    source = ROOT / record["path"]
    fixture = np.load(ROOT / record["prediction_fixture"], allow_pickle=False)
    experiment = Experiment.from_checkpoint(source, config=config)
    experiment.model.eval()
    with torch.no_grad():
        first = experiment.model.initial_inference(torch.tensor(fixture["observation"]))
        second = experiment.model.recurrent_inference(first[3], torch.tensor([[0]]))
    for prefix, tensors in (("initial", first), ("recurrent", second)):
        for index, tensor in enumerate(tensors):
            np.testing.assert_allclose(
                tensor.numpy(), fixture[f"{prefix}_{index}"], rtol=1e-5, atol=1e-6
            )
    destination = tmp_path / "converted.checkpoint"
    convert_checkpoint(source, destination, config)
    converted = Experiment.from_checkpoint(destination)
    for key, value in experiment.model.state_dict().items():
        torch.testing.assert_close(value, converted.model.state_dict()[key], rtol=0, atol=0)
    if experiment.controller is not None:
        for key, value in experiment.controller.state_dict().items():
            torch.testing.assert_close(
                value, converted.controller.state_dict()[key], rtol=0, atol=0
            )
    # Exercise historical input ordering through self-play with bounded search.
    converted.config.environment.max_steps = 4
    converted.config.decision.simulations = 4
    converted.config.decision.max_depth = 2
    converted.config.decision.horizon = 2
    converted.config.decision.generations = 1
    assert np.isfinite(converted.evaluate()[0]["raw_return"])
    with pytest.raises(FileExistsError):
        convert_checkpoint(source, destination, config)


@pytest.mark.parametrize("method", ["mcts", "nes"])
def test_save_resume_matches_uninterrupted_training(method, tmp_path):
    full_config = preset(method=method, smoke=True)
    full_config.training.steps = 2
    full_config.output_dir = str(tmp_path / "full")
    full = Experiment(full_config)
    full.train(tensorboard=False)
    partial_config = ExperimentConfig.from_dict(full_config.to_dict())
    partial_config.training.steps = 1
    partial_config.output_dir = str(tmp_path / "resumed")
    partial = Experiment(partial_config)
    partial.train(tensorboard=False)
    partial_config.training.steps = 2
    resumed = Experiment.from_checkpoint(
        Path(partial_config.output_dir) / "model.checkpoint", config=partial_config
    )
    resumed.train(tensorboard=False)
    for key, weights in full.model.state_dict().items():
        torch.testing.assert_close(weights, resumed.model.state_dict()[key], rtol=0, atol=0)
    if full.controller is not None:
        for key, weights in full.controller.state_dict().items():
            torch.testing.assert_close(
                weights, resumed.controller.state_dict()[key], rtol=0, atol=0
            )


def test_cancel_before_collection_saves_checkpoint(tmp_path):
    config = preset(smoke=True)
    config.output_dir = str(tmp_path)
    experiment = Experiment(config)
    experiment.cancel()
    with pytest.raises(InterruptedError):
        experiment.train(tensorboard=False)
    assert (tmp_path / "model.checkpoint").exists()


def test_failed_collection_saves_and_closes(tmp_path, monkeypatch):
    from muzero_research.training import self_play

    closed = []

    class BrokenGame:
        def reset(self, **kwargs):
            raise RuntimeError("Environment failure")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(self_play, "make_game", lambda *args, **kwargs: BrokenGame())
    config = preset(smoke=True)
    config.output_dir = str(tmp_path)
    experiment = Experiment(config)
    with pytest.raises(RuntimeError, match="Environment failure"):
        experiment.train(tensorboard=False)
    assert closed == [True]
    assert (tmp_path / "model.checkpoint").exists()


def test_residual_network_training(tmp_path):
    config = preset(smoke=True)
    config.network.kind = "resnet"
    config.training.steps = 1
    config.output_dir = str(tmp_path)
    experiment = Experiment(config)
    assert np.isfinite(experiment.train(tensorboard=False)["total_loss"])


def test_config_roundtrip_and_unknown_keys(tmp_path):
    config = preset("lunarlander", "beam", smoke=True)
    config.save(tmp_path / "config.yaml")
    assert ExperimentConfig.load(tmp_path / "config.yaml").to_dict() == config.to_dict()
    with pytest.raises(ValueError, match="Unknown training"):
        ExperimentConfig.from_dict({"training": {"stepps": 1}})
    config.decision.method = "nes"
    config.network.kind = "resnet"
    with pytest.raises(ValueError, match="NES"):
        build_network(config)
