"""Versioned checkpoints and explicit, non-overwriting historical conversion."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .config import ExperimentConfig

SCHEMA_VERSION = 1


def code_version():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return revision + ("-dirty" if dirty else "")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return __version__


def metadata(config):
    package = Path(__file__).resolve().parent
    source_hash = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        source_hash.update(path.relative_to(package).as_posix().encode())
        source_hash.update(path.read_bytes())
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config.to_dict(),
        "environment_variant": config.environment.name,
        "observation_layout": config.environment.observation_layout,
        "code_version": code_version(),
        "package_source_sha256": source_hash.hexdigest(),
        "package_version": __version__,
        "historical": False,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "gymnasium", "minigrid", "ray")
        },
    }


def portable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(portable(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Checkpoint contains unsupported object {type(value).__name__}")


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(portable(checkpoint), temporary)
    os.replace(temporary, path)


def is_bundled(path):
    manifest = Path(__file__).resolve().parents[2] / "artifacts" / "manifest.json"
    if not manifest.exists():
        return False
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return any(item["sha256"] == digest for item in json.loads(manifest.read_text())["checkpoints"])


def load_checkpoint(path, *, legacy_config=None, trust_legacy=False):
    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        if not (trust_legacy or is_bundled(path)):
            raise ValueError(
                "Checkpoint cannot be read with weights_only=True; use explicit trusted legacy conversion for historical pickle files"
            ) from error
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "weights" not in checkpoint:
        raise ValueError("Checkpoint must contain MuZero weights")
    if "metadata" in checkpoint:
        if checkpoint["metadata"]["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint schema version")
        config = ExperimentConfig.from_dict(checkpoint["metadata"]["config"])
    elif legacy_config is not None:
        config = legacy_config
    else:
        raise ValueError("Historical checkpoints require an explicit legacy configuration")
    return checkpoint, config


def convert_checkpoint(source, destination, config, *, trust_legacy=False):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise FileExistsError(
            "Conversion must use a new destination and preserve the source checkpoint"
        )
    checkpoint, config = load_checkpoint(source, legacy_config=config, trust_legacy=trust_legacy)
    checkpoint["metadata"] = metadata(config)
    checkpoint["metadata"].update(
        {
            "historical": True,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_name": source.name,
            "conversion_changes_weights": False,
        }
    )
    save_checkpoint(destination, checkpoint)
    return destination
