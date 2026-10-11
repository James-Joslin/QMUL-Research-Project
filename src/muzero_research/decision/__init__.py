"""Explicit decision-method registry."""

from .beam import BeamSearch
from .genetic import GeneticSearch
from .mcts import MCTS
from .nes import NESPolicy, build_controller


def make_strategy(config, *, evaluator=None, controller=None):
    config.validate()
    constructors = {
        "mcts": lambda: MCTS(config),
        "beam": lambda: BeamSearch(config),
        "genetic": lambda: GeneticSearch(config, evaluator),
        "nes": lambda: NESPolicy(
            config, controller if controller is not None else build_controller(config)
        ),
    }
    return constructors[config.decision.method]()
