import time
from functools import wraps

import numpy as np
from sklearn import base, pipeline, multiclass
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import utils


def timed(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        start = time.process_time()
        result = f(*args, **kwargs)
        stop = time.process_time()

        return stop - start, result

    return wrapped


class ReLU(base.TransformerMixin, base.BaseEstimator):
    """ReLU activation function."""

    def __init__(self, inplace=True):
        self.inplace = inplace

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = check_array(X)
        return np.maximum(X, 0, out=X if self.inplace else None)


class Threshold(base.TransformerMixin, base.BaseEstimator):
    """Threshold activation function."""

    def __init__(self, inplace=True):
        self.inplace = inplace

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = check_array(X)
        return np.heaviside(X, 0, dtype=X.dtype, out=X if self.inplace else None)


class FirstLayer(base.TransformerMixin, base.BaseEstimator):
    """First layer of the algorithm."""

    # [Line 1 from Algorithm 2 from the paper]

    def __init__(self, n_iter=500, max_bias=1.0, rng=None, dtype=np.float32):
        self.n_iter = n_iter
        self.max_bias = max_bias
        self.rng = rng or np.random.default_rng()
        self.dtype = dtype

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)

        self.dim_out_, self.dim_in_ = self.n_iter, X.shape[1]

        self.W_ = self.rng.standard_normal(
            size=(self.dim_out_, self.dim_in_), dtype=self.dtype
        )

        b = self.rng.random(size=self.dim_out_, dtype=self.dtype)
        self.b_ = (b * 2 - 1) * self.max_bias

        return self

    def transform(self, X):
        check_is_fitted(self, ("W_", "b_"))
        X = check_array(X)

        # calculate `out = X @ self.W.T + self.b` inplace
        out = X @ self.W_.T
        out += self.b_
        return out


class SecondLayer(base.TransformerMixin, base.BaseEstimator):
    """Second layer of the algorithm."""

    # [Lines 2-11 from Algorithm 2 from the paper]

    def __init__(self, rng=None, dtype=np.float32):
        self.rng = rng or np.random.default_rng()
        self.dtype = dtype

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        neg = y == 1
        pos = ~neg
        neg_idx = np.where(neg)[0]
        n_neg = np.count_nonzero(neg)

        # [Line 2]
        self.U_ = []
        self.m_ = []
        self._C = []  # This is used only for plots.

        candidates = np.full(n_neg, True, dtype=np.bool_)
        universe = np.full(n_neg, True, dtype=np.bool_)

        # Loop until there are no candidates left or every sample is
        # covered.
        while candidates.any() and universe.any():  # [Line 3]
            # [Line 4]
            # Draw a random candidate point and remove it from the list
            # of candidates.
            neg_i = self.rng.choice(np.where(candidates)[0])
            i = neg_idx[neg_i]
            candidates[neg_i] = False

            # [Line 5]
            # Calculate the mask U and the scalar products for both
            # classes at the same time.
            U = X[i] <= 0
            scalar_prod = X @ U.T
            m = scalar_prod[pos].min()

            # [Line 6]
            # Only proceed if at least the candidate point is covered.
            if m > 0:
                # [Line 7-8]
                # Remove points that are newly covered from `candidates`
                # and `universe`.
                not_covered = scalar_prod[neg] >= m
                candidates &= not_covered
                universe &= not_covered

                # Remember the neuron.
                self._C.append(neg_i)
                self.U_.append(U)
                self.m_.append(m)

        # If the layer is empty, add a dummy neuron.
        if not self.U_:
            self.U_.append(np.zeros(X.shape[1]))
            self.m_.append(0)

        # [Line 11]
        self.U_ = np.asarray(self.U_, dtype=np.uint8)
        self.m_ = np.asarray(self.m_, dtype=self.dtype)

        self.dim_out_, self.dim_in_ = self.U_.shape
        return self

    def transform(self, X):
        check_is_fitted(self, ("U_", "m_"))
        X = check_array(X)

        # calculate `out = -(X @ self.U.T) + self.m` inplace
        out = X @ self.U_.T
        out *= -1
        out += self.m_
        return out


class ThirdLayer(base.ClassifierMixin, base.BaseEstimator):
    """Third layer of the algorithm."""

    # [Part of Line 12 from Algorithm 2 from the paper.]

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.dim_out_, self.dim_in_ = 1, X.shape[1]
        self.label_dtype_ = y.dtype

        return self

    def predict(self, X):
        check_is_fitted(self, ("label_dtype_"))
        X = check_array(X)

        return X.any(axis=1).astype(self.label_dtype_)

    def predict_proba(self, X):
        pred = self.predict(X)
        return np.where((pred[:, np.newaxis] == 0), (1.0, 0.0), (0.0, 1.0))


def get_model(params, random_state=None):
    """Generate the model according to a set of parameters."""

    rng = np.random.default_rng(random_state)

    activation = {"thres": Threshold, "relu": ReLU}[params["activation"]]

    # [Line 12 from Algorithm 2 from the paper.]
    # We use `OneVsRestClassifier` to accomodate for the multiclass
    # case. Note that if there are only two labels, this still uses one
    # estimator only (and not two).
    model = pipeline.Pipeline(
        [
            ("firstlayer", FirstLayer(params["n_iter"], params["max_bias"], rng=rng)),
            ("activation-1", activation()),
            (
                "multi",
                multiclass.OneVsRestClassifier(
                    pipeline.Pipeline(
                        [
                            ("secondlayer", SecondLayer(rng=rng)),
                            ("activation-2", activation()),
                            ("thirdlayer", ThirdLayer()),
                        ]
                    )
                ),
            ),
        ]
    )

    return model


def get_model_metrics(model):
    """Collect and return metrics of a model."""

    metrics = {
        "first_dim_in": model["firstlayer"].dim_in_,
        "first_dim_out": model["firstlayer"].dim_out_,
    }

    estimators = model["multi"].estimators_
    for class_, estimator in zip(model["multi"].classes_, estimators):
        metrics.update(
            {
                f"second_dim_in_{class_}": estimator["secondlayer"].dim_in_,
                f"second_dim_out_{class_}": estimator["secondlayer"].dim_out_,
            }
        )

    metrics.update(
        {
            "second_dim_in": sum(
                estimator["secondlayer"].dim_in_ for estimator in estimators
            ),
            "second_dim_out": sum(
                estimator["secondlayer"].dim_out_ for estimator in estimators
            ),
        }
    )

    return metrics


def trial(params):
    """Execute one trial according to a set of parameters."""

    random_state = params["_index"]

    X, y = utils.load_dataset(params["dataset"])
    model = get_model(params, random_state)

    fit_time, _ = timed(model.fit)(X, y)
    score_time, score = timed(model.score)(X, y)

    result = {
        "score": score,
        "fit_time": fit_time,
        "score_time": score_time,
    }
    result.update(get_model_metrics(model))

    return result
