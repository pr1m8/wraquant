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


# ---------------------------------------------------------------------------
# Extended distribution analysis
# ---------------------------------------------------------------------------


def fit_stable_distribution(data: pd.Series | np.ndarray) -> dict:
    """Fit a stable (Levy) distribution to data.

    Uses scipy's ``levy_stable`` distribution to estimate the four
    parameters: alpha (stability), beta (skewness), loc, and scale.

    Parameters:
        data: Data array or series.

    Returns:
        Dictionary with:
        - ``alpha``: stability parameter (0, 2].
        - ``beta``: skewness parameter [-1, 1].
        - ``loc``: location parameter.
        - ``scale``: scale parameter.
        - ``ks_statistic``: Kolmogorov-Smirnov statistic.
        - ``ks_pvalue``: KS test p-value.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]

    params = sp_stats.levy_stable.fit(clean)
    alpha, beta, loc, scale = params

    ks_stat, ks_p = sp_stats.kstest(
        clean, "levy_stable", args=(alpha, beta, loc, scale)
    )

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "loc": float(loc),
        "scale": float(scale),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


def tail_index(
    data: pd.Series | np.ndarray,
    method: str = "hill",
    threshold_quantile: float = 0.9,
) -> dict:
    """Estimate the tail index of a distribution.

    The tail index characterises the heaviness of distribution tails.
    A finite tail index indicates power-law tails (Pareto-like).

    Parameters:
        data: Data array or series.
        method: Estimation method.  One of ``"hill"`` (Hill estimator),
            ``"pickands"`` (Pickands estimator), or ``"moment"``
            (moment estimator of Dekkers-Einmahl-de Haan).
        threshold_quantile: Quantile above which tail observations are
            used (default 0.9, i.e. top 10%).

    Returns:
        Dictionary with:
        - ``tail_index``: estimated tail index (xi).
        - ``method``: method used.
        - ``n_tail``: number of observations in the tail.

    Raises:
        ValueError: If *method* is not one of the supported estimators.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    # Work with absolute values for tail analysis
    abs_data = np.abs(clean)
    threshold = float(np.quantile(abs_data, threshold_quantile))
    tail_obs = np.sort(abs_data[abs_data > threshold])[::-1]
    n_tail = len(tail_obs)

    if n_tail < 4:
        return {"tail_index": float("nan"), "method": method, "n_tail": n_tail}

    if method == "hill":
        # Hill estimator: 1/xi = (1/k) * sum(log(X_(i)) - log(X_(k+1)))
        # where X_(1) >= X_(2) >= ... are order statistics
        k = n_tail - 1
        log_ratio = np.log(tail_obs[:k]) - np.log(tail_obs[k])
        hill_est = float(np.mean(log_ratio))
        xi = hill_est  # gamma = 1/xi for Pareto, but xi itself is the tail index

    elif method == "pickands":
        # Pickands estimator using quartiles of tail
        m = n_tail // 4
        if m < 1:
            return {"tail_index": float("nan"), "method": method, "n_tail": n_tail}
        x1 = tail_obs[m - 1]
        x2 = tail_obs[2 * m - 1]
        x4 = tail_obs[4 * m - 1] if 4 * m <= n_tail else tail_obs[-1]
        denom = x2 - x4
        if denom <= 0:
            xi = 0.0
        else:
            xi = float(np.log((x1 - x2) / denom) / np.log(2.0))

    elif method == "moment":
        # Moment estimator (Dekkers-Einmahl-de Haan)
        k = n_tail - 1
        log_ratio = np.log(tail_obs[:k]) - np.log(tail_obs[k])
        m1 = float(np.mean(log_ratio))
        m2 = float(np.mean(log_ratio ** 2))
        xi = m1 + 1.0 - 0.5 / (1.0 - m1 ** 2 / m2) if m2 > 0 else m1

    else:
        msg = f"Unknown method '{method}'. Use 'hill', 'pickands', or 'moment'."
        raise ValueError(msg)

    return {
        "tail_index": float(xi),
        "method": method,
        "n_tail": n_tail,
    }


def qqplot_data(
    data: pd.Series | np.ndarray,
    dist: str = "norm",
) -> dict:
    """Generate quantile-quantile plot data.

    Computes theoretical and sample quantiles for constructing a Q-Q
    plot against a reference distribution.

    Parameters:
        data: Data array or series.
        dist: Name of a ``scipy.stats`` distribution to use as the
            theoretical reference (default ``"norm"``).

    Returns:
        Dictionary with:
        - ``theoretical_quantiles``: array of theoretical quantiles.
        - ``sample_quantiles``: array of ordered sample values.
        - ``slope``: slope of the best-fit line through the Q-Q plot.
        - ``intercept``: intercept of the best-fit line.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    clean = np.sort(clean)
    n = len(clean)

    distribution = getattr(sp_stats, dist)
    # Fit distribution to data for theoretical quantiles
    params = distribution.fit(clean)
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical = distribution.ppf(probabilities, *params)

    # Best-fit line
    slope, intercept = np.polyfit(theoretical, clean, 1)

    return {
        "theoretical_quantiles": theoretical,
        "sample_quantiles": clean,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def jarque_bera(data: pd.Series | np.ndarray) -> dict:
    """Perform the Jarque-Bera test for normality.

    Tests the null hypothesis that the data is normally distributed,
    based on sample skewness and kurtosis.

    Parameters:
        data: Data array or series.

    Returns:
        Dictionary with:
        - ``statistic``: Jarque-Bera test statistic.
        - ``p_value``: p-value of the test.
        - ``skewness``: sample skewness.
        - ``kurtosis``: sample excess kurtosis.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    n = len(clean)

    skew = float(sp_stats.skew(clean, bias=False))
    kurt = float(sp_stats.kurtosis(clean, bias=False))  # excess kurtosis

    jb_stat = (n / 6.0) * (skew ** 2 + (kurt ** 2) / 4.0)
    p_value = float(1.0 - sp_stats.chi2.cdf(jb_stat, df=2))

    return {
        "statistic": float(jb_stat),
        "p_value": p_value,
        "skewness": skew,
        "kurtosis": kurt,
    }


def kolmogorov_smirnov(
    data: pd.Series | np.ndarray,
    dist: str = "norm",
) -> dict:
    """Perform the Kolmogorov-Smirnov goodness-of-fit test.

    Tests the null hypothesis that *data* was drawn from the specified
    distribution.  The distribution parameters are first estimated via
    MLE.

    Parameters:
        data: Data array or series.
        dist: Name of a ``scipy.stats`` distribution (default ``"norm"``).

    Returns:
        Dictionary with:
        - ``statistic``: KS test statistic.
        - ``p_value``: p-value of the test.
        - ``dist``: distribution name tested.
        - ``params``: fitted distribution parameters.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]

    distribution = getattr(sp_stats, dist)
    params = distribution.fit(clean)
    ks_stat, ks_p = sp_stats.kstest(clean, dist, args=params)

    return {
        "statistic": float(ks_stat),
        "p_value": float(ks_p),
        "dist": dist,
        "params": params,
    }
