"""Read both historical CSV/NPZ files and the common experiment output format."""

from pathlib import Path

import numpy as np


def read_results(paths):
    import pandas as pd

    frames = []
    for name in paths:
        path = Path(name)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                reward_key = "rewards" if "rewards" in arrays else "arr_0"
                timing_key = "timings" if "timings" in arrays else "arr_1"
                rewards = np.asarray(arrays[reward_key]).reshape(-1)
                if timing_key in arrays and arrays[timing_key].size != rewards.size:
                    raise ValueError(f"Reward and timing lengths differ in {path}")
                data = {"raw_return": rewards, "episode": np.arange(len(rewards))}
                if timing_key in arrays:
                    data["decision_seconds"] = arrays[timing_key].reshape(-1)
                frame = pd.DataFrame(data)
                if "metadata" in arrays:
                    import json

                    metadata = json.loads(str(arrays["metadata"]))
                    frame["method"] = metadata["config"]["decision"]["method"]
                    frame["environment"] = metadata["environment_variant"]
        elif path.suffix == ".csv":
            frame = pd.read_csv(path)
            frame = frame.rename(
                columns={"total_reward": "raw_return", "domain": "environment", "id": "run_id"}
            )
        else:
            raise ValueError(f"Expected a CSV or NPZ input, got {path}")
        if "raw_return" not in frame or frame.empty:
            raise ValueError(f"No episode returns found in {path}")
        if "method" not in frame:
            frame["method"] = path.stem
        if "environment" not in frame:
            frame["environment"] = "unspecified"
        if "run_id" not in frame:
            frame["run_id"] = path.stem
        if "episode" not in frame:
            frame["episode"] = (
                frame["training_step"]
                if "training_step" in frame
                else frame.groupby("run_id").cumcount()
            )
        frame["source"] = str(path)
        if not np.isfinite(frame["raw_return"].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite returns in {path}")
        frames.append(frame)
    if not frames:
        raise ValueError("At least one result input is required")
    return pd.concat(frames, ignore_index=True)


def summary(frame):
    from scipy.integrate import simpson

    rows = []
    for (method, environment, run), group in frame.groupby(
        ["method", "environment", "run_id"], sort=True
    ):
        group = group.sort_values("episode")
        rewards = group["raw_return"]
        row = {
            "method": method,
            "environment": environment,
            "run_id": run,
            "episodes": len(group),
            "mean": rewards.mean(),
            "median": rewards.median(),
            "std": rewards.std(ddof=1) if len(group) > 1 else 0.0,
            "min": rewards.min(),
            "max": rewards.max(),
            "integral": simpson(rewards.to_numpy(), x=group["episode"].to_numpy())
            if len(group) > 1
            else 0.0,
        }
        if "decision_seconds" in group:
            row["mean_decision_seconds"] = group["decision_seconds"].mean()
        rows.append(row)
    return rows


def plot_results(paths, output, *, kind="box", window=10):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = read_results(paths)
    if window < 1:
        raise ValueError("Rolling window must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    environments = list(frame["environment"].unique())
    figure, axes = plt.subplots(
        len(environments), 1, squeeze=False, figsize=(7, 4 * len(environments))
    )
    for axis, environment in zip(axes[:, 0], environments, strict=True):
        selected = frame[frame["environment"] == environment]
        if kind == "box":
            groups = list(selected.groupby("method", sort=True))
            axis.boxplot(
                [group["raw_return"].to_numpy() for _, group in groups],
                tick_labels=[method for method, _ in groups],
                showmeans=True,
            )
        else:
            for (method, run), group in selected.groupby(["method", "run_id"], sort=True):
                group = group.sort_values("episode")
                axis.plot(
                    group["episode"],
                    group["raw_return"].rolling(window, min_periods=1).mean(),
                    label=f"{method}: {run}",
                )
            axis.legend()
            axis.set_xlabel("Episode / training step")
        axis.set_title(environment)
        axis.set_ylabel("Raw episode return")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return summary(frame)
