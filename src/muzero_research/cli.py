"""Noninteractive command-line interface for reproducible research workflows."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .config import ExperimentConfig, preset


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "tune", "diagnose"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--game", choices=["gridworld", "lunarlander", "lunarlander_standard"])
        command.add_argument("--method", choices=["mcts", "beam", "genetic", "nes"])
        command.add_argument("--seed", type=int)
        command.add_argument("--device", choices=["cpu", "cuda"])
        command.add_argument("--output", type=Path)
        command.add_argument("--checkpoint", type=Path)
        command.add_argument("--trust-legacy", action="store_true")
        if name in {"train", "tune"}:
            command.add_argument("--steps", type=int)
            command.add_argument("--workers", type=int)
        if name == "train":
            command.add_argument("--no-tensorboard", action="store_true")
        if name == "evaluate":
            command.add_argument("--episodes", type=int, default=1)
            command.add_argument("--render", action="store_true")
            command.add_argument("--render-dir", type=Path)
        if name == "tune":
            command.add_argument("--samples", type=int, default=4)
        if name == "diagnose":
            command.add_argument(
                "--graph",
                action="store_true",
                help="Also save a DOT graph of one-step model predictions",
            )
    plot = commands.add_parser("plot")
    plot.add_argument("inputs", nargs="+", type=Path)
    plot.add_argument("--output", type=Path, required=True)
    plot.add_argument("--kind", choices=["box", "training"], default="box")
    plot.add_argument("--window", type=int, default=10)
    convert = commands.add_parser("convert-checkpoint")
    convert.add_argument("source", type=Path)
    convert.add_argument("destination", type=Path)
    convert.add_argument("--config", type=Path, required=True)
    convert.add_argument("--trust-legacy", action="store_true")
    database = commands.add_parser("import-database")
    database.add_argument("table")
    database.add_argument("--output", type=Path, required=True)
    return root


def resolve_config(args):
    from .checkpoints import load_checkpoint

    if args.config is not None:
        config = ExperimentConfig.load(args.config)
    elif args.checkpoint is not None:
        _, config = load_checkpoint(args.checkpoint, trust_legacy=args.trust_legacy)
    else:
        config = preset(args.game or "gridworld", args.method or "mcts")
    if args.game is not None and args.game != config.environment.name:
        config.environment = preset(args.game, config.decision.method).environment
    if args.method is not None:
        config.decision.method = args.method
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.runtime.device = args.device
    if getattr(args, "steps", None) is not None:
        config.training.steps = args.steps
    if getattr(args, "workers", None) is not None:
        config.runtime.workers = args.workers
    if args.command in {"train", "tune"} and args.output is not None:
        config.output_dir = str(args.output)
    config.validate()
    return config


def run(args):
    if args.command == "plot":
        from .analysis.results import plot_results

        return plot_results(args.inputs, args.output, kind=args.kind, window=args.window)
    if args.command == "import-database":
        from .analysis.database import import_table

        return str(import_table(args.table, args.output))
    if args.command == "convert-checkpoint":
        from .checkpoints import convert_checkpoint

        return str(
            convert_checkpoint(
                args.source,
                args.destination,
                ExperimentConfig.load(args.config),
                trust_legacy=args.trust_legacy,
            )
        )
    config = resolve_config(args)
    if args.command == "tune":
        from .tuning import tune_experiment

        return tune_experiment(config, samples=args.samples)
    from .experiment import Experiment

    if args.checkpoint:
        experiment = Experiment.from_checkpoint(
            args.checkpoint, config=config, trust_legacy=args.trust_legacy
        )
    else:
        experiment = Experiment(config)
    if args.command == "train":
        return experiment.train(tensorboard=not args.no_tensorboard)
    if args.command == "evaluate":
        return experiment.write_evaluation(
            args.output or Path(config.output_dir) / "evaluation.csv",
            args.episodes,
            render_mode="human" if args.render else None,
            render_dir=args.render_dir,
        )
    from .diagnostics import diagnose

    return diagnose(
        experiment, args.output or Path(config.output_dir) / "diagnosis.json", graph=args.graph
    )


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        result = run(args)
    except KeyboardInterrupt:
        print("Interrupted; the final training checkpoint was saved.")
        return 130
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, TimeoutError) as error:
        command_parser.exit(2, f"error: {error}\n")
    print(json.dumps(copy.deepcopy(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
