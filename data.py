from functools import partial, wraps

import numpy as np
from sklearn import datasets

import utils


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
def lines(n_samples, distance=1.0):
    """The 'Two parallel lines' dataset."""

    n, d = n_samples // 2, distance / 2
    t = 2 * np.arange(n) / (n - 1) - 1
    X1 = np.vstack((t, np.full(n, -d))).T
    X2 = np.vstack((t, np.full(n, d))).T
    X = np.concatenate((X1, X2))
    y = np.concatenate((np.full(n, 1), np.full(n, 0)))
    return X, y


def main():
    datasets = {
        "moons": partial(normalize(moons), n_samples=1_000, noise=0.05),
        "mnist": normalize(mnist),
        "mnist_original": mnist,
        "lines": partial(lines, n_samples=20, distance=0.1),
    }

    X, y = normalize(moons)(1_000_000, noise=0.05)
    datasets.update(
        {
            "moons_size_100": lambda X=X, y=y: (X[:100], y[:100]),
            "moons_size_1k": lambda X=X, y=y: (X[:1_000], y[:1_000]),
            "moons_size_10k": lambda X=X, y=y: (X[:10_000], y[:10_000]),
        }
    )
    datasets.update(
        {
            f"moons_size_{i}k": lambda X=X, y=y, i=i: (X[: i * 1_000], y[: i * 1_000])
            for i in range(100, 1100, 100)
        }
    )

    print("Generating datasets ...")

    for name, generator_fn in datasets.items():
        print(f" -> {name}")

        X, y = generator_fn()
        utils.save_dataset(name, X, y)


if __name__ == "__main__":
    main()
