"""Context-managed Ray lifecycle with cancellation and resource checks."""

import time


class RayRuntime:
    def __init__(self, config):
        self.config = config
        self.owned = False
        self.pending = []

    def __enter__(self):
        import ray

        if not ray.is_initialized():
            options = {
                "num_cpus": self.config.cpus,
                "num_gpus": 1 if self.config.device == "cuda" else 0,
                "include_dashboard": False,
                "log_to_driver": False,
            }
            if self.config.ray_object_store_bytes is not None:
                options["object_store_memory"] = self.config.ray_object_store_bytes
            try:
                ray.init(**options)
            except BaseException:
                ray.shutdown()
                raise
            self.owned = True
        return self

    def collect(self, references, *, cancelled=None):
        import ray

        self.pending = list(references)
        remaining, values = list(references), {}
        deadline = time.monotonic() + self.config.startup_timeout
        while remaining:
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("Experiment cancelled while waiting for workers")
            if time.monotonic() > deadline:
                raise TimeoutError("Workers did not complete within runtime.startup_timeout")
            ready, remaining = ray.wait(remaining, num_returns=1, timeout=0.2)
            for reference in ready:
                values[reference] = ray.get(reference)
        self.pending = []
        return [values[reference] for reference in references]

    def __exit__(self, exc_type, exc_value, traceback):
        import ray

        for reference in self.pending:
            ray.cancel(reference, force=True)
        self.pending = []
        if self.owned:
            ray.shutdown()
            self.owned = False
