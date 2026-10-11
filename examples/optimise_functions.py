"""Small NES sanity experiment, replacing the duplicate Confirming-NES optimiser."""

import argparse

import numpy as np

from muzero_research.decision.nes import es_update
from muzero_research.runtime.fitness import LocalFitnessEvaluator


def sphere(parameters):
    return -float(np.square(parameters).sum())


def rastrigin(parameters):
    return -float(
        10 * len(parameters) + (np.square(parameters) - 10 * np.cos(2 * np.pi * parameters)).sum()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=["sphere", "rastrigin"], default="sphere")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    objective = sphere if args.objective == "sphere" else rastrigin
    parameters = np.array([2.0, -2.0])
    evaluator = LocalFitnessEvaluator()
    for _ in range(args.generations):
        noise = rng.standard_normal((100, 2))
        fitness = evaluator.evaluate(objective, parameters + 0.2 * noise)
        # A zero-mean fitness baseline reduces variance without changing the estimator.
        parameters = es_update(parameters, noise, fitness - fitness.mean(), 0.02, 0.2)
    print({"parameters": parameters.tolist(), "fitness": objective(parameters)})


if __name__ == "__main__":
    main()
