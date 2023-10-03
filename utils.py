from pathlib import Path
import json
from importlib import import_module

import numpy as np
import pandas as pd


def get_path(type_, filename, subdir=""):
    """Generate a path from a type and a file name (and optional
    subdir).
    """

    paths = {
        "config": "./experiments/",
        "trial": "./",
        "dataset": "./datasets/",
        "result": "./results/",
        "plot": "./plots/",
    }

    return Path(paths[type_]) / subdir / filename


def load_config(name):
    """Load an experiment config."""

    filename = f"{name}.json"
    path = get_path("config", filename)

    with open(path, "r") as handle:
        config = json.load(handle)

    return config


def save_dataset(name, X, y):
    """Save a dataset."""

    filename = f"{name}.npz"
    path = get_path("dataset", filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(path, X=X, y=y)


def load_dataset(name):
    """Load a dataset."""

    filename = f"{name}.npz"
    path = get_path("dataset", filename)

    data = np.load(path)
    X, y = data["X"], data["y"]

    return X, y


def load_trial(name):
    """Load a trial."""

    filename = name
    path = get_path("trial", filename)

    trial_module = import_module(str(path))
    trial = trial_module.trial

    return trial


def save_result(name, df, suffix=""):
    """Save a result."""

    filename = f"{name}.csv{suffix}"
    path = get_path("result", filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)


def load_result(name, suffix=""):
    """Load a result."""

    filename = f"{name}.csv{suffix}"
    path = get_path("result", filename)

    df = pd.read_csv(path)

    return df


def remove_result(name, suffix=""):
    """Remove a result."""

    filename = f"{name}.csv{suffix}"
    path = get_path("result", filename)

    path.unlink()


def save_plot(name, fig, subdir=""):
    """Save a plot."""

    filename = f"{name}.pdf"
    path = get_path("plot", filename, subdir=subdir)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, bbox_inches="tight")
