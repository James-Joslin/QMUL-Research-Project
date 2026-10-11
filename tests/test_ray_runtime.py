import threading

import numpy as np
import pytest

from muzero_research.config import preset
from muzero_research.runtime import RayRuntime
from muzero_research.runtime.fitness import LocalFitnessEvaluator, RayFitnessEvaluator


def sphere(candidate):
    return -float(np.square(candidate).sum())


@pytest.mark.ray
def test_ray_population_equivalence_failure_cancellation_and_cleanup():
    import ray

    config = preset(smoke=True)
    candidates = np.arange(20).reshape(5, 4)
    with RayRuntime(config.runtime) as runtime:
        with RayRuntime(config.runtime) as borrowed:
            assert not borrowed.owned
        assert ray.is_initialized()
        objective = lambda candidate: -float(np.square(candidate).sum())  # noqa: E731
        local = LocalFitnessEvaluator().evaluate(objective, candidates)
        remote = RayFitnessEvaluator(2).evaluate(objective, candidates)
        np.testing.assert_array_equal(remote, local)

        def failure():
            raise ValueError("worker failed")

        failing = ray.remote(failure).remote()
        with pytest.raises(Exception, match="worker failed"):
            runtime.collect([failing])
        flag = threading.Event()
        flag.set()
        reference = ray.remote(objective).remote(candidates[0])
        with pytest.raises(InterruptedError):
            runtime.collect([reference], cancelled=flag)
    assert not ray.is_initialized()


@pytest.mark.ray
def test_parallel_selfplay_training(tmp_path):
    import ray

    from muzero_research.experiment import Experiment

    config = preset(smoke=True)
    config.training.steps = 1
    config.runtime.workers = 2
    config.output_dir = str(tmp_path)
    experiment = Experiment(config)
    metrics = experiment.train(tensorboard=False)
    assert metrics["played_games"] == 2
    assert not ray.is_initialized()


@pytest.mark.ray
@pytest.mark.parametrize("method", ["genetic", "nes"])
def test_population_workers_with_real_world_model(method, tmp_path):
    import ray

    from muzero_research.experiment import Experiment

    config = preset(method=method, smoke=True)
    config.training.steps = 1
    config.runtime.population_backend = "ray"
    config.runtime.fitness_workers = 2
    config.output_dir = str(tmp_path / method)
    metrics = Experiment(config).train(tensorboard=False)
    assert np.isfinite(metrics["total_loss"])
    assert not ray.is_initialized()


@pytest.mark.ray
def test_tuning_runs_a_complete_trial(tmp_path):
    import ray

    from muzero_research.tuning import tune_experiment

    config = preset(method="beam", smoke=True)
    config.training.steps = 1
    config.output_dir = str(tmp_path / "tuning")
    result = tune_experiment(config, samples=1)
    assert np.isfinite(result["mean_return"])
    assert "learning_rate" in result["parameters"]
    assert not ray.is_initialized()
