import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_training_curves(
    path_log: pd.DataFrame, 
    alpha: float,
    ax=None, 
    save=None
) -> plt.Axes:
    """
    Plot training curves from a log file.

    Parameters
    ----------
    log_df : pd.DataFrame
       A bpnet-lite log file
    ax : matplotlib.pyplot.Axes, optional
        Axes to plot on, by default None

    Returns
    -------
    matplotlib.pyplot.Axes

    """
    # Read log file
    log_df = pd.read_csv(path_log, sep="\t")
    log_df["Training Loss"] = log_df["Training MNLL"] + alpha*log_df["Training Count MSE"]
    log_df["Validation Loss"] = log_df["Validation MNLL"] + alpha*log_df["Validation Count MSE"]
    log_df = log_df.set_index("Iteration")

    # Create figure and axes if not provided
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    # Line plot of "Training Loss", "Validation Loss"
    ax[0].plot(log_df["Training Loss"], label="Training Loss")
    ax[0].plot(log_df["Validation Loss"], label="Validation Val Loss")

    # Line plot of "Validation Count Pearson"
    ax[1].plot(log_df["Validation Count Pearson"], label="Validation Count Pearson")
    ax[1].plot(log_df["Validation Profile Pearson"], label="Validation Profile Pearson")

    # Plot last point in both where "Saved?" is True
    last_saved = log_df[log_df["Saved?"] == True].index[-1]
    ax[0].axvline(last_saved, color="black", linestyle="--", label="Checkpoint")
    ax[1].axvline(last_saved, color="black", linestyle="--", label="Checkpoint")

    # Set labels
    ax[0].set_xlabel("Iteration")
    ax[1].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss")
    ax[1].set_ylabel("Pearson Correlation Coefficient")
    ax[0].legend()
    ax[1].legend()

    # Add a little room to y-axis upper limit
    ax[0].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1] * 1.1)
    ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[1] * 1.1)

    plt.tight_layout()

    if save:    
        plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.close()        
        
    return ax