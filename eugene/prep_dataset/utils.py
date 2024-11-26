import yaml
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import mannwhitneyu, ttest_ind
from scipy.stats import kruskal, f_oneway
from statsmodels.stats.multitest import fdrcorrection


def correlate_cnt_with_continuous(
    cnt,
    covariate,
    normalize=False,
    method="pearson",
    nan_policy='omit',
):
    """Calculate the correlation between two variables.

    Parameters
    ----------
    cnt: np.ndarray
        The count variable as a 1D numpy array.
    covariate: np.ndarray
        The continuous variable as a 1D numpy array.
    normalize: bool
        Whether to normalize the count so that it sums to 1.
    method: str
        The method to use for correlation. Options are "pearson", "spearman", and "kendall".
    nan_policy: str
        The policy for handling NaN values. Options are "omit" and "raise".

    Returns
    -------
    (float, float)
        The correlation coefficient and p-value as a tuple.
    """
    if sum(cnt) == 0:
        return (np.nan, np.nan)
    
    if normalize:
        cnt = cnt / cnt.sum()

    if nan_policy == "omit":
        nan_msk = np.isnan(covariate) | np.isnan(cnt)
        cnt = cnt[~nan_msk]
        covariate = covariate[~nan_msk]
    if nan_policy == "raise":
        if np.isnan(covariate).any() or np.isnan(cnt).any():
            raise ValueError("NaN values found in the data")
    if method == "pearson":
        return pearsonr(cnt, covariate)
    elif method == "spearman":
        return spearmanr(cnt, covariate)
    elif method == "kendall":
        return kendalltau(cnt, covariate)
    else:
        raise ValueError(f"Unknown method: {method}")
    

def run_continuous_correlations(
    cnts: np.ndarray,
    covariate: np.ndarray,
    method: str = "pearson",
    nan_policy: str = "omit",
):
    """Calculate the correlation between each column of counts and a continuous covariate.

    Parameters
    ----------
    counts: np.ndarray
        The count variables as a 2D numpy array.
    covariate: np.ndarray
        The continuous variable as a 1D numpy array.
    method: str
        The method to use for correlation. Options are "pearson", "spearman", and "kendall".
    nan_policy: str
        The policy for handling NaN values. Options are "omit" and "raise".

    Returns
    -------
    (np.ndarray, np.ndarray)
        The correlation coefficients and p-values as numpy arrays.
    """
    n = cnts.shape[1]
    corrs = np.zeros(n)
    pvals = np.zeros(n)

    for i in range(n):
        corrs[i], pvals[i] = correlate_cnt_with_continuous(cnts[:, i], covariate, method=method, nan_policy=nan_policy)

    return corrs, pvals


def correlate_cnt_with_binary(
    cnt,
    binary,
    normalize=False,
    method="mannwhitneyu",
    nan_policy='omit',
):
    """Calculate the correlation between a count variable and a binary variable.

    Parameters
    ----------
    cnt: np.ndarray
        The count variable as a 1D numpy array.
    binary: np.ndarray
        The binary variable as a 1D numpy array.
    normalize: bool
        Whether to normalize the count so that it sums to 1.
    method: str
        The method to use for correlation. Options are "mannwhitneyu", "ttest_ind", "pearson", "spearman", and "kendall".

    Returns
    -------
    (float, float)
        The correlation coefficient and p-value as a tuple.
    """
    if sum(cnt) == 0:
        return (np.nan, np.nan)
    
    if normalize:
        cnt = cnt / cnt.sum()

    if method == "mannwhitneyu":
        return mannwhitneyu(cnt, binary, nan_policy=nan_policy)
    elif method == "ttest_ind":
        return ttest_ind(cnt, binary, nan_policy=nan_policy)
    elif method in ["pearson", "spearman", "kendall"]:
        return correlate_cnt_with_continuous(cnt, binary, normalize=normalize, method=method, nan_policy=nan_policy)
    

def run_binary_correlations(
    cnts: np.ndarray,
    binary: np.ndarray,
    method: str = "mannwhitneyu",
    nan_policy: str = "omit",
):
    """Calculate the correlation between each column of counts and a binary covariate.

    Parameters
    ----------
    counts: np.ndarray
        The count variables as a 2D numpy array.
    binary: np.ndarray
        The binary variable as a 1D numpy array.
    method: str
        The method to use for correlation. Options are "mannwhitneyu" and "ttest_ind".
    nan_policy: str
        The policy for handling NaN values. Options are "omit" and "raise".

    Returns
    -------
    (np.ndarray, np.ndarray)
        The correlation coefficients and p-values as numpy arrays.
    """
    n = cnts.shape[1]
    corrs = np.zeros(n)
    pvals = np.zeros(n)

    for i in range(n):
        corrs[i], pvals[i] = correlate_cnt_with_binary(cnts[:, i], binary, method=method, nan_policy=nan_policy)

    return corrs, pvals


def correlate_cnt_with_categorical(
    cnt,
    categorical,
    normalize=False,
    method="kruskal",
    nan_policy='omit',
):
    """Calculate the correlation between a count variable and a categorical variable.

    Parameters
    ----------
    cnt: np.ndarray
        The count variable as a 1D numpy array.
    categorical: np.ndarray
        The categorical variable as a 1D numpy array.
    normalize: bool
        Whether to normalize the count so that it sums to 1.
    method: str
        The method to use for correlation. Options are "kruskal" and "f_oneway".
    nan_policy: str
        The policy for handling NaN values. Options are "omit" and "raise".

    Returns
    -------
    (float, float)
        The correlation coefficient and p-value as a tuple.
    """
    if sum(cnt) == 0:
        return (np.nan, np.nan)
    
    if normalize:
        cnt = cnt / cnt.sum()

    if method == "kruskal":
        return kruskal(*[cnt[categorical == i] for i in np.unique(categorical)], nan_policy=nan_policy)
    elif method == "f_oneway":
        return f_oneway(*[cnt[categorical == i] for i in np.unique(categorical)], nan_policy=nan_policy)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    
def run_categorical_correlations(
    cnts: np.ndarray,
    categorical: np.ndarray,
    method: str = "kruskal",
    nan_policy: str = "omit",
):
    """Calculate the correlation between each column of counts and a categorical covariate.

    Parameters
    ----------
    counts: np.ndarray
        The count variables as a 2D numpy array.
    categorical: np.ndarray
        The categorical variable as a 1D numpy array.
    method: str
        The method to use for correlation. Options are "kruskal" and "f_oneway".
    nan_policy: str
        The policy for handling NaN values. Options are "omit" and "raise".

    Returns
    -------
    (np.ndarray, np.ndarray)
        The correlation coefficients and p-values as numpy arrays.
    """
    n = cnts.shape[1]
    corrs = np.zeros(n)
    pvals = np.zeros(n)

    for i in range(n):
        corrs[i], pvals[i] = correlate_cnt_with_categorical(cnts[:, i], categorical, method=method, nan_policy=nan_policy)

    return corrs, pvals
