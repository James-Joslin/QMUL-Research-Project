"""MuZero decision-method research, with explicit experiment configuration."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "Experiment":
        from .experiment import Experiment

        return Experiment
    raise AttributeError(name)
