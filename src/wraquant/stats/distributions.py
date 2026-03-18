"""Distribution fitting and tail analysis for financial data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def fit_distribution(data: pd.Series, dist: str = "norm") -> dict:
    """Fit a parametric distribution to data.

    Parameters:
        data: Data series to fit.
        dist: Name of a ``scipy.stats`` distribution (e.g., ``"norm"``,
            ``"t"``, ``"lognorm"``).

    Returns:
        Dictionary with ``params`` (tuple of fitted parameters),
        ``ks_statistic``, and ``ks_pvalue`` from a Kolmogorov-Smirnov
        goodness-of-fit test.

    Raises:
        AttributeError: If *dist* is not a valid scipy distribution.
    """
    clean = data.dropna().values
    distribution = getattr(sp_stats, dist)
    params = distribution.fit(clean)
    ks_stat, ks_p = sp_stats.kstest(clean, dist, args=params)
    return {
        "params": params,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


def tail_ratio(returns: pd.Series, quantile: float = 0.05) -> float:
    """Compute the tail ratio (right tail / left tail).

    A tail ratio > 1 indicates a fatter right tail (more extreme gains)
    relative to the left tail.

    Parameters:
        returns: Return series.
        quantile: Quantile for tail measurement (default 5%).

    Returns:
        Tail ratio as a float.
    """
    clean = returns.dropna()
    right = abs(clean.quantile(1 - quantile))
    left = abs(clean.quantile(quantile))
    if left == 0:
        return float("inf")
    return float(right / left)


def hurst_exponent(data: pd.Series) -> float:
    """Estimate the Hurst exponent via rescaled range (R/S) analysis.

    The Hurst exponent characterises the long-term memory of a series:

    - ``H < 0.5``: mean-reverting
    - ``H = 0.5``: random walk
    - ``H > 0.5``: trending / persistent

    Parameters:
        data: Time series (prices or returns).

    Returns:
        Estimated Hurst exponent as a float.
    """
    clean = data.dropna().values
    n = len(clean)

    max_k = max(2, n // 2)
    sizes = []
    rs_values = []

    size = 8
    while size <= max_k:
        sizes.append(size)
        n_chunks = n // size
        rs_list = []
        for i in range(n_chunks):
            chunk = clean[i * size : (i + 1) * size]
            mean = chunk.mean()
            deviations = chunk - mean
            cumdev = np.cumsum(deviations)
            r = cumdev.max() - cumdev.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append(np.mean(rs_list))
        size *= 2

    if len(sizes) < 2:
        return 0.5

    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)
    slope, _intercept = np.polyfit(log_sizes, log_rs, 1)
    return float(slope)
