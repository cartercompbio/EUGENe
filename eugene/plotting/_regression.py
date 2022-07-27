import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Sequence
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
from ._utils import _create_matplotlib_axes, _label_plot, _plot_seaborn, tflog2pandas, many_logs2pandas


def _plot_performance_scatter(
    sdata, 
    target: str,
    prediction: str,
    metrics: Union[str, Sequence[str]] = ["r2", "mse", "spearmanr"],
    **kwargs
) -> None:
    """
    Plot a scatter plot of the performance of the model on a subset of the sequences.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target : str
        Name of the target variable.
    prediction : str
        Name of the prediction variable.
    metrics : str or list of str
        Metrics to plot. Should be the string name of the metric used in PL
    **kwargs

    Returns
    -------
    None
    """
    targets = sdata.seqs_annot[target]
    predictions = sdata.seqs_annot[prediction]

    r2 = r2_score(targets, predictions) if "r2" in metrics else None
    mse = mean_squared_error(targets, predictions) if "mse" in metrics else None
    spearr = spearmanr(targets, predictions).correlation if "spearmanr" in metrics else None

    plt.scatter(targets, predictions, **kwargs)
    plt.xlabel(target)
    plt.ylabel(prediction)
    plt.text(1.02, 0.95, f"$R^2$: {r2:.2f}", transform=plt.gca().transAxes, fontsize=16) if r2 is not None else None
    plt.text(1.02, 0.90, f"MSE: {mse:.2f}", transform=plt.gca().transAxes, fontsize=16) if mse is not None else None
    plt.text(1.02, 0.85, fr"Spearman $\rho$: {spearr:.2f}", transform=plt.gca().transAxes, fontsize=16) if spearr is not None else None
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")


def performance_scatter(
    sdata, 
    target: str, 
    prediction: str, 
    seq_idx = None,
    **kwargs
) -> None:
    """ 
    Plot a scatter plot of the performance of the model on a subset of the sequences.
    
    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target : str
        Name of the target variable.
    prediction : str
        Name of the prediction variable.
    seq_idx : list of int
        List of indices of sequences to plot.
    **kwargs

    Returns
    -------
    None
    """
    if seq_idx is not None:
        sdata = sdata[seq_idx]
    _plot_performance_scatter(sdata, target=target, prediction=prediction, **kwargs)
