import os
from os import PathLike
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from ._utils import (
    _plot_seaborn,
    many_logs2pandas,
    _save_fig,
)
from .. import settings


def metric_curve(
    log_path,
    metric: str = None,
    hue: str = "metric",
    title: str = None,
    xlab: str = "minibatch step",
    ylab: str = None,
    ax = None,
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots the loss curves from a PyTorch Lightning (PL) training run

    Parameters
    ----------
    log_path : str
        Path to tensorboard log directory.
    metric : str
        Metric to plot. Should be the string name of the metric used in PL
    title : str
        Title of plot.
    xlab : str
        Label for x-axis.
    ylab : str
        Label for y-axis.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
    If return_axes is True, returns the axes object.
    """
    ylab = metric if ylab is None else ylab
    tb_event_path = glob.glob(os.path.join(log_path, "events.out.tfevents.*"))
    dataframe = many_logs2pandas(tb_event_path)
    dataframe = dataframe[dataframe["metric"].str.contains(metric)]
    dataframe = dataframe[~dataframe["metric"].str.contains("step")]
    ax = _plot_seaborn(
        dataframe,
        "value",
        sns.lineplot,
        groupby="step",
        orient="v",
        title=title,
        xlab=xlab,
        ylab=ylab,
        hue=hue,
        ax=ax,
        **kwargs
    )
    if return_axes:
        return ax


def loss_curve(
    log_path: PathLike,
    title: str = None,
    xlab: str = "minibatch_step",
    ylab: str = "loss",
    ax = None,
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots the loss curves from a PyTorch Lightning (PL) training run. Wraps metrics_curve().

    Parameters
    ----------
    log_path : str
        Path to tensorboard log directory.
    title : str
        Title of plot.
    xlab : str
        Label for x-axis.
    ylab : str
        Label for y-axis.
    return_axes : bool
        If True, return the axes object.
    **kwargs
        Additional keyword arguments to pass to seaborn.

    Returns
    -------
    If return_axes is True, returns the axes object.
    """
    ax = metric_curve(
        log_path, 
        metric="loss", 
        title=title, 
        xlab=xlab, 
        ylab=ylab, 
        ax=ax,
        **kwargs
    )
    if return_axes:
        return ax


def training_summary(
    log_path: PathLike, 
    metric: str, 
    figsize=(12, 6),
    save: str = None, 
    return_axes: bool = False,
    **kwargs
) -> None:
    """
    Plots the training summary from a training run. Convenience function to plot loss and metric together.

    Parameters
    ----------
    log_path : str
        Path to tensorboard log directory.
    metrics : str or list of str
        Metrics to plot. Should be the string name of the metric used in PL

    Returns
    -------
    None
    """
    _, ax = plt.subplots(1, 2, figsize=figsize)
    loss_curve(log_path, ax=ax[0], **kwargs)
    metric_curve(log_path, metric=metric, ax=ax[1], **kwargs)
    if save is not None:
        _save_fig(save)
    if return_axes:
        return ax
