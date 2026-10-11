"""One noninteractive Ray Tune workflow for every decision method."""

import copy
from pathlib import Path


def tune_experiment(config, *, samples=4):
    if samples < 1:
        raise ValueError("Tuning sample count must be positive")
    from ray import tune

    from .config import ExperimentConfig
    from .experiment import Experiment
    from .runtime import RayRuntime

    data = config.to_dict()

    def trainable(parameters):
        trial_config = copy.deepcopy(data)
        trial_config["training"]["learning_rate"] = parameters["learning_rate"]
        trial_config["decision"]["dirichlet_alpha"] = parameters["dirichlet_alpha"]
        trial_config["decision"]["exploration_fraction"] = parameters["exploration_fraction"]
        trial_config["runtime"].update({"workers": 1, "population_backend": "local"})
        trial_config["output_dir"] = str(tune.get_context().get_trial_dir())
        experiment = Experiment(ExperimentConfig.from_dict(trial_config))
        metrics = experiment.train(tensorboard=False)
        evaluation = experiment.evaluate(episodes=3)
        tune.report(
            {
                "mean_return": sum(row["raw_return"] for row in evaluation) / len(evaluation),
                "total_loss": metrics.get("total_loss", 0.0),
            }
        )

    with RayRuntime(config.runtime):
        resources = {"cpu": 1, "gpu": 1 if config.runtime.device == "cuda" else 0}
        search = {
            "learning_rate": tune.loguniform(
                config.training.learning_rate / 2, config.training.learning_rate * 2
            ),
            "dirichlet_alpha": tune.uniform(0.225, 0.275),
            "exploration_fraction": tune.uniform(0.225, 0.275),
        }
        tuner = tune.Tuner(
            tune.with_resources(trainable, resources),
            param_space=search,
            tune_config=tune.TuneConfig(metric="mean_return", mode="max", num_samples=samples),
            run_config=tune.RunConfig(
                storage_path=str(Path(config.output_dir).resolve()),
                name="tuning",
            ),
        )
        results = tuner.fit()
        if any(result.error is not None for result in results):
            raise RuntimeError("A tuning trial failed; inspect the trial output")
        best = results.get_best_result(metric="mean_return", mode="max")
        return {"parameters": best.config, "mean_return": best.metrics["mean_return"]}
