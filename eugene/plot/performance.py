import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.stats import gaussian_kde
from typing import Optional, Sequence, Union


def scatter(
    x,
    y,
    ax=None,
    density=False,
    c="#1f77b4",
    alpha=1,
    s=10,
    xlabel="Observed",
    ylabel="Predicted",
    metrics: Union[str, Sequence[str]] = ["mse", "pearsonr", "spearmanr"],
    figsize=(4, 4),
    save=None,
    add_reference_line=True,
    rasterized=False,
    return_axes=False,
):
    # Set up the axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Drop NA values if any
    x_nas = np.isnan(x)
    y_nas = np.isnan(y)
    x = x[~x_nas & ~y_nas]
    y = y[~x_nas & ~y_nas]

    if density:
        # Get point densities
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        c = z

    # Plot the points
    ax.scatter(x, y, c=c, s=s, rasterized=rasterized, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add scores outside the plot, not inside the plot
    if isinstance(metrics, str):
        metrics = [metrics]
    for i, metric in enumerate(metrics):
        if metric == "mse":
            mse = np.mean((x - y) ** 2)
            ax.annotate(
                f"MSE: {mse:.2f}",
                xy=(1.05, 0.95 - i * 0.05),
                xycoords="axes fraction",
            )
        elif metric == "pearsonr":
            pearson_r = pearsonr(x, y)
            ax.annotate(
                f"Pearson's r: {pearson_r[0]:.2f}",
                xy=(1.05, 0.95 - i * 0.05),
                xycoords="axes fraction",
            )
        elif metric == "spearmanr":
            spearman_r = spearmanr(x, y)
            ax.annotate(
                f"Spearman's r: {spearman_r[0]:.2f}",
                xy=(1.05, 0.95 - i * 0.05),
                xycoords="axes fraction",
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Add y=x line for reference but make the mins and maxes extend past the data
    if add_reference_line:
        min_val = min(min(x), min(y))
        max_val = max(max(x), max(y))
        ax.plot([min_val, max_val], [min_val, max_val], c="k", ls="--", lw=1)

    # Save and make sure it doesn't show
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        # Plt
        plt.tight_layout()
    if return_axes:
        return ax