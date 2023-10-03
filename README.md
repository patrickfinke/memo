# Memorization with neural nets: going beyond the worst case

This repository contains code to reproduce the results from the paper ["Memorization with neural nets: going beyond the worst case"](https://arxiv.org/abs/2310.00327).


## Repository structure

These are the core files of the repository:
- `data.py` generates datasets and writes them to `datasets/`.
- `run.py` runs experiments defined in `experiments/` and writes results to `results/`.
- `plot.py` generates plots and writes them to `plots/`.
- `algorithm.py` contains the implementation of the algorithm.

The rest are auxiliary files:
- `requirements.txt` contains a list of required packages.
- `utils.py` contains utility functions.
- `sge.sh` is a helper script for executing experiments on clusters running SGE.

All Python code is formatted with [black](https://github.com/psf/black) in its default configuration.


## Reproducing results

The code requires Python 3.8+ and the packages listed in requirements.txt. To reproduce the results from the paper, follow these steps:

1. Clone the repository, set up a virtual environment, and install the dependencies.

```bash
git clone https://github.com/patrickfinke/memo.git
cd memo

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Generate the datasets. This only needs to be done once.

```bash
python data.py
```

3. Run the experiments using `run.py`. Specify the name with the `--name` argument. A valid name is any filename in `experiments/` without the `.json` extension. Run the computation locally by specifying the backend `--backend joblib` (default) or on a compute cluster running Sun Grid Engine (SGE) with the backend `--backend sge`. (The file `sge.sh` might need some adjustments first.) The `--tasks` argument expects an integer specifying the number of threads (for `joblib`) or tasks in an array job (for `sge`). Setting this to `-1` uses all available resources, the default is `1`. For SGE, arguments after `--` will be passed to the `qsub` command. For example:

```bash
python run.py --name moons
python run.py --name mnist --backend sge --tasks 20 -- -q all.q
```

Alternatively, unzip the precomputed results:

```bash
unzip results.zip
```

4. Generate the plots using `plot.py`. Specify the name with the `--name` argument. Plots can be found in a subfolder of `plots/`. For example:

```bash
python plot.py --name moons
```


## Experiment configurations

Experiments are configured via JSON files inside `experiments/`. These contain a mapping with the following keys and values:

- `trial` maps to a filename (without `.py` extension) of a Python script that implements a method called `trial`. This method will be called for each set of parameters in the parameter grid to produce the results of the experiment.
- `param_grid` contains a mapping or list that is compatible with the `ParameterGrid` class from scikit-learn. See the existing files for examples.
- `plots` contains a list of mappings that each configures a plot. See the docstrings in `plot.py` for an explanation and the existing files for examples.
