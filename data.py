from functools import partial, wraps
import math

import numpy as np
from sklearn import preprocessing
from sklearn import datasets
from sklearn import model_selection

import utils


def pick_binary_from_multi(label_neg, label_pos):
    """Decorate a dataset to select a binary subproblem from a
    multiclass problem."""

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            X, y = f(*args, **kwargs)

            mask_neg = y == label_neg
            mask_pos = y == label_pos
            y[mask_neg] = 1
            y[mask_pos] = 0

            mask = mask_neg | mask_pos
            return X[mask], y[mask]

        return wrapper
    
    return decorator


def convert_dtype(f):
    """Decorate a dataset to convert data to float32, labels to uint8."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        X, y = f(*args, **kwargs)

        X = X.astype(np.float32)
        y = y.astype(np.uint8)
        return X, y

    return wrapper


def normalize(f):
    """Decorate a dataset to normalize the euclidean radius."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        X, y = f(*args, **kwargs)

        X /= np.linalg.norm(X, ord=2, axis=1).max()
        return X, y

    return wrapper


@convert_dtype
def moons(n_samples, noise=0.0, random_state=0):
    """The 'Two Moons' dataset."""

    X, y = datasets.make_moons(n_samples, noise=noise, random_state=random_state)
    return X, y


@convert_dtype
def mnist():
    """The 'MNIST' dataset."""

    X, y = datasets.fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )
    X = X.reshape((X.shape[0], -1)) / 255
    return X, y


@convert_dtype
def cifar10():
    """The 'CIFAR-10' dataset."""

    X, y = datasets.fetch_openml(
        "cifar_10", return_X_y=True, as_frame=False, parser="auto"
    )
    X = X.reshape((X.shape[0], -1)) / 255
    return X, y


@convert_dtype
def lines(n_samples, distance=1.0):
    """The 'Two parallel lines' dataset."""

    n, d = n_samples // 2, distance / 2
    t = 2 * np.arange(n) / (n - 1) - 1
    X1 = np.vstack((t, np.full(n, -d))).T
    X2 = np.vstack((t, np.full(n, d))).T
    X = np.concatenate((X1, X2))
    y = np.concatenate((np.full(n, 1), np.full(n, 0)))
    return X, y


def generate_sizes(X, y, sizes):
    datasets = []

    X_acc, X, y_acc, y = model_selection.train_test_split(X, y, train_size=sizes[0], stratify=y)
    datasets.append((X_acc, y_acc))

    increments = sizes[1:] - sizes[:-1]
    for increment in increments:
        if increment < X.shape[0]:
            X_incr, X, y_incr, y = model_selection.train_test_split(X, y, train_size=increment, stratify=y)
            X_acc = np.concatenate((X_acc, X_incr), axis=0)
            y_acc = np.concatenate((y_acc, y_incr))

            datasets.append((X_acc, y_acc))
        else:
            X_acc = np.concatenate((X_acc, X), axis=0)
            y_acc = np.concatenate((y_acc, y))

            datasets.append((X_acc, y_acc))
            break

    return datasets


def main():
    datasets = {}

    # moons
    datasets["moons"] = partial(normalize(moons), n_samples=1_000, noise=0.05)

    # moons size
    X, y = normalize(moons)(1_000_000, noise=0.05)
    datasets["moons_size_100"] = lambda X=X, y=y: (X[:100], y[:100])
    datasets["moons_size_1k"] = lambda X=X, y=y: (X[:1_000], y[:1_000])
    datasets["moons_size_5k"] = lambda X=X, y=y: (X[:5_000], y[:5_000])
    datasets["moons_size_10k"] = lambda X=X, y=y: (X[:10_000], y[:10_000])
    for i in range(100, 1100, 100):
        datasets[f"moons_size_{i}k"] = lambda X=X, y=y, i=i: (X[: i * 1_000], y[: i * 1_000])

    # mnist
    datasets["mnist"] = normalize(mnist)
    datasets["mnist_original"] = mnist

    # mnist 1 vs 9
    X, y = normalize(pick_binary_from_multi(1, 9)(mnist))()
    percentage = np.arange(10, 110, 10)
    sizes = (percentage / 100 * X.shape[0]).astype(int)
    for p, (X_partial, y_partial) in zip(percentage, generate_sizes(X, y, sizes)):
        datasets[f"mnist_1v9_size_{p}"] = lambda X_partial=X_partial, y_partial=y_partial: (X_partial, y_partial)

    # mnist 1 vs 8
    X, y = normalize(pick_binary_from_multi(1, 8)(mnist))()
    percentage = np.arange(10, 110, 10)
    sizes = (percentage / 100 * X.shape[0]).astype(int)
    for p, (X_partial, y_partial) in zip(percentage, generate_sizes(X, y, sizes)):
        datasets[f"mnist_1v8_size_{p}"] = lambda X_partial=X_partial, y_partial=y_partial: (X_partial, y_partial)

    # cifar10
    datasets["cifar10_original"] = cifar10
    datasets["cifar10"] = normalize(cifar10)

    # lines
    datasets["lines"] = partial(lines, n_samples=20, distance=0.1)


    print("Generating datasets ...")

    for name, generator_fn in datasets.items():
        print(f" -> {name}")

        X, y = generator_fn()
        utils.save_dataset(name, X, y)


if __name__ == "__main__":
    main()
