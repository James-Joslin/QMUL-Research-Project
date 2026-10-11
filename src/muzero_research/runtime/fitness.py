"""Order-preserving local and Ray batch evaluation with explicit ownership."""

from __future__ import annotations

import numpy as np

from ..contracts import FitnessEvaluator


def evaluate_batch(objective, candidates):
    values = np.asarray([objective(candidate) for candidate in candidates], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Population fitness must be finite")
    return values


class LocalFitnessEvaluator(FitnessEvaluator):
    def evaluate(self, objective, candidates):
        return evaluate_batch(objective, candidates)


class RayFitnessEvaluator(FitnessEvaluator):
    def __init__(self, workers=1, *, timeout=120.0, device="cpu"):
        import ray

        if not ray.is_initialized():
            raise RuntimeError("Start Ray through RayRuntime before constructing a Ray evaluator")
        self.workers = workers
        self.timeout = timeout
        self.task = ray.remote(num_cpus=1, num_gpus=1 if device == "cuda" else 0)(evaluate_batch)
        self.pending = []

    def evaluate(self, objective, candidates):
        import ray

        batches = [
            batch
            for batch in np.array_split(candidates, min(self.workers, len(candidates)))
            if len(batch)
        ]
        # One immutable objective/model snapshot for the entire population.
        objective_ref = ray.put(objective)
        self.pending = [self.task.remote(objective_ref, batch) for batch in batches]
        try:
            values = ray.get(self.pending, timeout=self.timeout)
            return np.concatenate(values)
        finally:
            self.close()

    def close(self):
        import ray

        for ref in self.pending:
            ray.cancel(ref, force=True)
        self.pending = []


def make_evaluator(config):
    if config.runtime.population_backend == "ray":
        return RayFitnessEvaluator(
            config.runtime.fitness_workers,
            timeout=config.runtime.startup_timeout,
            device=config.runtime.device,
        )
    return LocalFitnessEvaluator()
