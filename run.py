import os
import argparse
import functools
import subprocess
import contextlib

import numpy as np
import pandas as pd
from sklearn import model_selection
import joblib
from tqdm.auto import tqdm

import utils


class ProgressParallel(joblib.Parallel):
    """joblib.Parallel with tqdm progress bar.

    Source: https://stackoverflow.com/a/61900501
    """

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return super().__call__(*args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def preprocess_param_grid(obj):
    """Preprocess parameter grid.

    This parses ranges for convenience. Eg. an entry of the form
    *{"type_": "range", "start": 5, "stop": 10, "step": 2}* is converted
    to *np.arange(5, 10, 2)*.
    """

    if isinstance(obj, dict):
        type_ = obj.get("type_", None)
        if type_ is None:
            return {k: preprocess_param_grid(v) for k, v in obj.items()}
        elif type_ == "range":
            # If encountering a dict of 'type' range, convert to a numpy
            # range.
            if not "stop" in obj:
                raise ValueError("range without 'stop'")
            start = obj.get("start", 0)
            stop = obj["stop"]
            step = obj.get("step", 1)
            return np.arange(start, stop, step)
    elif isinstance(obj, list):
        return [preprocess_param_grid(o) for o in obj]
    else:
        return obj


def include_params(trial):
    """Decorate the trial function to output both parameters and
    results.
    """

    @functools.wraps(trial)
    def wrapped(params):
        result = trial(params)
        return {**params, **result}

    return wrapped


def run_joblib(args, trial, param_grid):
    """Run jobs locally. Optional multiprocessing uses joblib."""

    print(f"Running in 'joblib' backend ...")

    # Execute jobs using joblib with tqdm progress bar.
    jobs = (joblib.delayed(trial)(params) for params in param_grid)
    results = ProgressParallel(n_jobs=args.tasks, total=len(param_grid))(jobs)

    # Sort and save results.
    df = pd.DataFrame(results).set_index("_index").sort_index()
    utils.save_result(args.name, df)


def run_sge(args, trial, param_grid):
    """Automatically submit jobs on SGE and collect results."""

    print("Running in 'sge' backend ...")

    # The SGE_TASK_ID environment variable is set by SGE and starts at
    # value 1. We cannot have more tasks then there are parameter
    # configurations and if the number of tasks is -1, we do everything
    # at once.
    job_id = int(os.environ.get("SGE_TASK_ID", 0)) - 1
    n_tasks = min(args.tasks, len(param_grid))
    if n_tasks == -1:
        n_tasks = len(param_grid)

    if job_id == -1:
        # If the SGE_TASK_ID environment variable is not set, we are not
        # running on a compute node. Hence, we should submit the tasks.
        print("Submitting tasks ...")

        # Let the qsub command wait until all jobs are finished. If
        # there was an error, then quit.
        res = subprocess.run(
            [
                "qsub",
                "-N",
                args.name,
                "-t",
                f"1-{n_tasks}",
                "-sync",
                "y",
                *args.sge_args,
                "sge.sh",
                args.name,
                "sge",
                str(n_tasks),
            ]
        )
        if res.returncode != 0:
            print("One or more tasks failed! Quitting ...")
            return

        # Combine the partial results into one file.
        print("Combining temporary results ...")
        dfs = [utils.load_result(args.name, suffix=f"_part{i}") for i in range(n_tasks)]
        df = pd.concat(dfs).set_index("_index").sort_index()
        utils.save_result(args.name, df)

        # Clean up the partial results.
        for i in range(n_tasks):
            utils.remove_result(args.name, suffix=f"_part{i}")
    else:
        # The SGE_TASK_ID environment variable is set. Hence, we run on
        # a compute node and need to process a chunk of the experiment.
        print(f"Processing chunk {job_id+1}/{n_tasks} ...")

        # Select a chunk of the parameter grid and process it. This does
        # not distribute the amount of work equally between nodes but it
        # is good enough.
        params_chunk = param_grid[job_id::n_tasks]
        results = [trial(params) for params in params_chunk]

        # Save the partial results.
        df = pd.DataFrame(results)
        utils.save_result(args.name, df, suffix=f"_part{job_id}")


def main():
    BACKENDS = {
        "joblib": run_joblib,
        "sge": run_sge,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", dest="name", type=str, required=True)
    parser.add_argument(
        "--backend", dest="backend_name", choices=BACKENDS.keys(), default="joblib"
    )
    parser.add_argument("--tasks", dest="tasks", type=int, default=1)
    parser.add_argument("sge_args", nargs="*")
    args = parser.parse_args()

    assert args.tasks >= 1 or args.tasks == -1

    print(f"Loading experiment config ...")
    config = utils.load_config(args.name)

    # For the parameter grid we use sklearn's ParameterGrid class and
    # inject a unique index that we use for reproducibility.
    print(f"Building parameter grid ...")
    param_grid = config["param_grid"]
    param_grid = preprocess_param_grid(param_grid)
    param_grid = model_selection.ParameterGrid(param_grid)
    param_grid = [{"_index": i, **params} for i, params in enumerate(param_grid)]

    # Load the trail code and decorate it, so that it also returns the
    # parameters.
    print(f"Loading trial code ...")
    trial_name = config["trial"]
    trial = utils.load_trial(trial_name)
    trial = include_params(trial)

    # Run the experiment.
    backend = BACKENDS[args.backend_name]
    backend(args, trial, param_grid)


if __name__ == "__main__":
    main()
