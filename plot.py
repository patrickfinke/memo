import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import base, pipeline

import utils


sns.set_style("whitegrid")

plt.rcParams["xtick.major.pad"] = "1.5"
plt.rcParams["ytick.major.pad"] = "1.5"

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc("font", size=SMALL_SIZE)  # default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def set_labels(ax, config):
    """Adjust axis labels.

    The *config* dict should contain:
    xlabel -- the x-axis label (default None)
    ylabel -- the y-axis label (default None)
    """

    if config.get("xlabel") is not None:
        ax.set_xlabel(config["xlabel"])
    if config.get("ylabel") is not None:
        ax.set_ylabel(config["ylabel"])


def set_ticks(ax, config):
    """Toggle tick labels.

    The *config* dict should contain:
    xticks -- enable x-axis ticks (default True)
    yticks -- enable y-axis ticks (default True)
    """

    if not config.get("xticks", True):
        ax.set_xticklabels([])
    if not config.get("yticks", True):
        ax.set_yticklabels([])


def set_value_limit(ax, config):
    """Set the y-axis limits.

    The same can be achieved with *set_ylim()*.

    The *config* dict should contain:
    vmin -- the minimal y-axis value (default automatic)
    vmax -- the maximal y-axis value (default automatic)
    """

    ax.dataLim.y0 = config.get("vmin", ax.dataLim.y0)
    ax.dataLim.y1 = config.get("vmax", ax.dataLim.y1)
    ax.autoscale_view()


def set_xlim(ax, config):
    """Set the x-axis limits.

    The *config* dict should contain:
    xlim -- a tuple of min and max x-axis values (default automatic)
    """

    x0, x1 = config.get("xlim", (ax.dataLim.x0, ax.dataLim.x1))
    ax.dataLim.x0, ax.dataLim.x1 = x0, x1
    ax.autoscale_view()


def set_ylim(ax, config):
    """Set the y-axis limits.

    The *config* dict should contain:
    ylim -- a tuple of min and max y-axis values (default automatic)
    """

    y0, y1 = config.get("ylim", (ax.dataLim.y0, ax.dataLim.y1))
    ax.dataLim.y0, ax.dataLim.y1 = y0, y1
    ax.autoscale_view()


def set_legend(ax, config):
    """Set the legend.

    The *config* dict should contain:
    legend -- If False do not show a legend, otherwise must be a dict
              containing the key 'title' and optionally 'labels'
              (default False)
    """

    if config.get("legend", False) is False:
        return

    handles, labels = ax.get_legend_handles_labels()

    config = config["legend"]
    ax.legend(
        handles=handles,
        title=config.get("title"),
        labels=config.get("labels", labels),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )


def set_outline(ax, config):
    """Toggle a highlighted outline.

    The *config* dict should contain:
    outline -- toggle the outline (default False)
    """

    if config.get("outline", False):
        ax.patch.set(edgecolor="black", linestyle="--", linewidth=3)


def set_axis_formatter(ax):
    def formatter(x):
        return f"{x.get_text()}"

    ax.set_yticklabels(label.get_text() for label in ax.get_yticklabels())
    ax.set_xticklabels(list(map(formatter, ax.get_xticklabels())))


def dataset_2d(name, config):
    """Plot a 2D dataset.

    The *config* dict should contain:
    dataset -- dataset name
    figsize -- matplotlib figsize (default automatic)

    It can also contain parameters for the legend, axis limits, ticks
    and axis labels.
    """

    X, y = utils.load_dataset(config["dataset"])
    neg, pos = X[y == 1], X[y == 0]

    fig, ax = plt.subplots(figsize=config.get("figsize"))
    ax.plot(pos[:, 0], pos[:, 1], "or", markersize=2, label="$\mathcal{X}^+$")
    ax.plot(neg[:, 0], neg[:, 1], "^b", markersize=2, label="$\mathcal{X}^-$")
    ax.set_aspect("equal")

    set_legend(ax, config)
    set_xlim(ax, config)
    set_ylim(ax, config)
    set_ticks(ax, config)
    set_labels(ax, config)

    return fig


def dataset_images(name, config):
    """Plot samples from a dataset containing images.

    The *config* dict should contain:
    dataset    -- dataset name
    rows       -- number of rows
    cols       -- number of columns
    classes    -- list of class labels
    img_width  -- width of images
    img_height -- height of images
    figsize    -- matplotlib figsize
    """

    X, y = utils.load_dataset(config["dataset"])

    figsize = config.get("figsize", (2.5, 1))
    fig, axs = plt.subplots(
        config["rows"],
        config["cols"],
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
        squeeze=True,
        figsize=figsize,
    )

    shape = (config["img_width"], config["img_height"])
    for n, ax in zip(config["classes"], axs.flatten()):
        idx = np.where(y == n)[0][0]
        ax.imshow(X[idx].reshape(shape), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    return fig


def _load_df(name):
    """Load a result and calculate additional metrics."""

    df = utils.load_result(name)

    if "score" in df:
        df["interpolated"] = df["score"] == 1.0

    if "dataset" in df:
        n_samples = {}
        for dataset_name in df["dataset"].unique():
            X, y = utils.load_dataset(dataset_name)
            n_samples[dataset_name] = X.shape[0]

        df["n_samples"] = df.apply(lambda row: n_samples[row["dataset"]], axis=1)

    return df


def _make_pivot(df, config):
    """Return a pivot table of the results (using pandas).

    The *config* dict should contain:
    idx   -- name of column to be used as index
    col   -- name of column to be used as column
    val   -- name of column to be used as values
    agg   -- aggregation function
    round -- round float type index/column (default no rounding)
    """

    index, columns, values = config["idx"], config["col"], config["val"]
    round_ = config.get("round")

    if round_ is not None:
        if pd.api.types.is_float_dtype(df[index]):
            df[index] = df[index].round(round_)
        if pd.api.types.is_float_dtype(df[columns]):
            df[columns] = df[columns].round(round_)

    pivot = df.pivot_table(
        index=index, columns=columns, values=values, aggfunc=config.get("agg")
    )

    return pivot


def add_contours(ax, pivot, config):
    """Plot contours onto an existing axis.

    The *config* dict should contain a sub dict 'contours' with:
    smoothing -- Gaussian smoothing (default 0.0)
    vmin      -- min value (default automatic)
    vmax      -- max value (default automatic)
    levels    -- number of levels (default automatic)
    """

    if "contours" not in config:
        return

    config = config["contours"]
    smoothing = config.get("smoothing", 0.0)
    smoothed = scipy.ndimage.gaussian_filter(pivot, smoothing)
    ax.contour(
        smoothed,
        vmin=config.get("vmin"),
        vmax=config.get("vmax"),
        levels=config.get("levels"),
        cmap="viridis",
        linewidths=0.75,
    )


def heatmap(name, config):
    """Plot a heatmap.

    The *config* dict should contain:
    pivot    -- a pivot config dict
    vmin     -- min value (default automatic)
    vmax     -- max value (default automatic)
    contours -- a contour data dict (default no contours)
    figsize  -- matplotlib figsize

    It can also contain parameters for ticks and axis labels.
    """

    df = _load_df(name)
    pivot = _make_pivot(df, config["pivot"])

    fig, ax = plt.subplots(figsize=config.get("figsize"))
    sns.heatmap(
        pivot, vmin=config.get("vmin"), vmax=config.get("vmax"), square=True, ax=ax
    )
    ax.tick_params(axis="x", rotation=90)
    ax.axes.invert_yaxis()
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    add_contours(ax, pivot, config)
    set_axis_formatter(ax)
    set_labels(ax, config)
    set_ticks(ax, config)

    # contour of the interpolation region
    interpolation_pivot = {**config["pivot"], **{"val": "interpolated"}}
    pivot = _make_pivot(df, interpolation_pivot)
    smoothed = scipy.ndimage.gaussian_filter(pivot, 1.0)
    ax.contour(
        smoothed,
        levels=[0.99],
        cmap=mpl.colors.ListedColormap(["blueviolet"]),
        linestyles="dashed",
        linewidths=1.5,
    )

    return fig


def slices(name, config):
    """Plot heatmap slices.

    The *config* dict should contain:
    pivot            -- a pivot config dict
    slices           -- list of slices to plot (from pivot columns)
    marker_threshold -- interpolation probability threshold for markers
                        (default no markers)
    figsize          -- matplotlib figsize

    It can also contain parameters for the legend, axis limits, ticks
    and axis labels.
    """

    fig, ax = plt.subplots(figsize=config.get("figsize", (6, 3)))

    # Plot lines and error bars.
    df = _load_df(name)
    mask = df[config["pivot"]["col"]].isin(config["slices"])
    df = df[mask]

    sns.lineplot(
        df,
        x=config["pivot"]["idx"],
        y=config["pivot"]["val"],
        hue=config["pivot"]["col"],
        style=config["pivot"]["col"],
        markers=False,
        dashes=False,
        palette="flare",
        seed=0,
        ax=ax,
    )

    # Plot markers if interpolation probability is high enough.
    pivot = _make_pivot(df, config["pivot"])

    pivot_config = config["pivot"].copy()
    pivot_config["val"] = "interpolated"
    pivot_interpolation = _make_pivot(df, pivot_config)

    markers = pivot_interpolation > config.get("marker_threshold", 2.0)
    markers = markers.iloc[:: config.get("mark_every", 1)]
    sns.scatterplot(
        pivot[markers].iloc[:, ::-1],
        palette="flare",
        s=15,
        legend=False,
        zorder=100,
        ax=ax,
    )

    set_value_limit(ax, config)
    set_xlim(ax, config)
    set_ylim(ax, config)

    set_legend(ax, config)
    set_ticks(ax, config)
    set_labels(ax, config)

    return fig


class Selector(base.TransformerMixin, base.BaseEstimator):
    def __init__(self, transformer, index):
        self.transformer = transformer
        self.index = index

    def transform(self, X):
        outputs = self.transformer.transform(X)
        return outputs[:, np.newaxis, self.index]


def _plot_decision_surface(
    ax, estimator, steps, color, alpha=1.0, fill=False, linestyle=None
):
    x, y = np.meshgrid(
        np.linspace(*ax.get_xlim(), num=steps), np.linspace(*ax.get_ylim(), num=steps)
    )
    z = estimator.predict(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)

    col = mpl.colors.to_rgba(color, alpha=alpha)
    cmap = mpl.colors.ListedColormap(["#00000000", col])
    if fill:
        ax.contourf(x, y, z, 1, cmap=cmap)
    else:
        ax.contour(x, y, z, 1, linestyles=linestyle, cmap=cmap)


def boundary_2d(name, config):
    """Plot the decision boundary on top of a 2d dataset.

    The *data* dict should contain:
    dataset -- dataset name
    model   -- a model config dict
    legend  -- toggle legend (default True)
    figsize -- matplotlib figsize

    It can also contain parameters for axis limits, ticks and axis
    labels.
    """

    dataset, model_args = config["dataset"], config["model"]
    step = 150

    X, y = utils.load_dataset(dataset)
    neg, pos = X[y == 1], X[y == 0]

    # Build and fit the model.
    from algorithm import get_model

    model = get_model(model_args, random_state=0)
    model.fit(X, y)

    # Plot the dataset to adjust the aspect ratio.
    fig, ax = plt.subplots(figsize=config.get("figsize"))
    ax.plot(pos[:, 0], pos[:, 1], "or", markersize=2, label="$\mathcal{X}^+$")
    ax.plot(neg[:, 0], neg[:, 1], "^b", markersize=2, label="$\mathcal{X}^-$")
    if config.get("legend", True):
        ax.legend()
    ax.set_aspect("equal")

    set_xlim(ax, config)
    set_ylim(ax, config)

    # For each of the neurons of the second layer, we restrict the model
    # to that neuron and draw a decision boundary.
    estimator = model["multi"].estimators_[0]
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for index, color in zip(range(estimator["secondlayer"].dim_out_), colors):
        restricted_model = pipeline.make_pipeline(
            model["firstlayer"],
            model["activation-1"],
            Selector(estimator["secondlayer"], index),
            estimator["activation-2"],
            estimator["thirdlayer"],
        )

        _plot_decision_surface(ax, restricted_model, step, color, alpha=0.5, fill=True)
        _plot_decision_surface(
            ax, restricted_model, step, color, alpha=1.0, fill=False, linestyle="solid"
        )

    _plot_decision_surface(ax, model, step, "black", fill=False, linestyle="dashed")

    # Draw the dataset on top of the decision boundaries.
    ax.plot(pos[:, 0], pos[:, 1], "or", markersize=2, label="$\mathcal{X}^+$")
    ax.plot(neg[:, 0], neg[:, 1], "^b", markersize=2, label="$\mathcal{X}^-$")

    # Draw markers for the points associated to each of the neurons.
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for index, color in zip(range(estimator["secondlayer"].dim_out_), colors):
        i = estimator["secondlayer"]._C[index]
        ax.plot(
            neg[i, 0],
            neg[i, 1],
            "*",
            markersize=12,
            markerfacecolor=color,
            markeredgewidth=1.0,
            markeredgecolor="black",
        )

    set_ticks(ax, config)
    set_labels(ax, config)
    set_outline(ax, config)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", dest="name", type=str, required=True)
    args = parser.parse_args()

    config = utils.load_config(args.name)
    plots = config["plots"]

    print("Generating plots ...")

    plot_fns = {
        "dataset_2d": dataset_2d,
        "dataset_images": dataset_images,
        "heatmap": heatmap,
        "slices": slices,
        "boundary_2d": boundary_2d,
    }

    for config in plots:
        type_, name = config["type"], config["name"]
        print(f" -> {name} ({type_})")

        plot_fn = plot_fns[type_]
        fig = plot_fn(args.name, config)
        utils.save_plot(name, fig, subdir=args.name)


if __name__ == "__main__":
    main()
