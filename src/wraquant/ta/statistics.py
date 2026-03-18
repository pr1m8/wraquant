"""Statistical technical analysis indicators.

This module provides rolling statistical measures commonly used in
quantitative analysis and systematic trading. All functions accept
``pd.Series`` inputs and return ``pd.Series``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "zscore",
    "percentile_rank",
    "mean_deviation",
    "median",
    "skewness",
    "kurtosis",
    "entropy",
    "hurst_exponent",
    "correlation",
    "beta",
    "r_squared",
    "information_coefficient",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_series(data: pd.Series, name: str = "data") -> pd.Series:
    if not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pd.Series, got {type(data).__name__}")
    return data


def _validate_period(period: int, name: str = "period") -> int:
    if period < 1:
        raise ValueError(f"{name} must be >= 1, got {period}")
    return period


# ---------------------------------------------------------------------------
# Z-Score
# ---------------------------------------------------------------------------


def zscore(data: pd.Series, period: int = 20) -> pd.Series:
    """Rolling z-score of price.

    Measures how many standard deviations the current value is from the
    rolling mean.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Rolling window length.

    Returns
    -------
    pd.Series
        Z-score values (unbounded).

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([10, 11, 12, 11, 10, 9, 10, 11, 12, 13])
    >>> zscore(close, period=5)  # doctest: +SKIP
    """
    _validate_series(data)
    _validate_period(period)

    rolling_mean = data.rolling(window=period, min_periods=period).mean()
    rolling_std = data.rolling(window=period, min_periods=period).std(ddof=1)
    result = (data - rolling_mean) / rolling_std
    result.name = "zscore"
    return result


# ---------------------------------------------------------------------------
# Percentile Rank
# ---------------------------------------------------------------------------


def percentile_rank(data: pd.Series, period: int = 20) -> pd.Series:
    """Rolling percentile rank.

    Computes the percentage of values within the rolling window that are
    less than or equal to the current value.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Rolling window length.

    Returns
    -------
    pd.Series
        Percentile rank in [0, 100].

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([10, 11, 12, 11, 10, 9, 10, 11, 12, 13])
    >>> percentile_rank(close, period=5)  # doctest: +SKIP
    """
    _validate_series(data)
    _validate_period(period)

    def _pct_rank(window: np.ndarray) -> float:
        current = window[-1]
        return np.sum(window <= current) / len(window) * 100.0

    result = data.rolling(window=period, min_periods=period).apply(
        _pct_rank, raw=True
    )
    result.name = "percentile_rank"
    return result


# ---------------------------------------------------------------------------
# Mean Deviation
# ---------------------------------------------------------------------------


def mean_deviation(data: pd.Series, period: int = 20) -> pd.Series:
    """Rolling mean absolute deviation.

    Computes the average of absolute deviations from the rolling mean.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Rolling window length.

    Returns
    -------
    pd.Series
        Mean absolute deviation values (>= 0).

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([10, 11, 12, 11, 10, 9, 10, 11, 12, 13])
    >>> mean_deviation(close, period=5)  # doctest: +SKIP
    """
    _validate_series(data)
    _validate_period(period)

    result = data.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    result.name = "mean_deviation"
    return result


# ---------------------------------------------------------------------------
# Median
# ---------------------------------------------------------------------------


def median(data: pd.Series, period: int = 20) -> pd.Series:
    """Rolling median.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Rolling window length.

    Returns
    -------
    pd.Series
        Rolling median values.

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([10, 11, 12, 11, 10, 9, 10, 11, 12, 13])
    >>> median(close, period=5)  # doctest: +SKIP
    """
    _validate_series(data)
    _validate_period(period)

    result = data.rolling(window=period, min_periods=period).median()
    result.name = "median"
    return result


# ---------------------------------------------------------------------------
# Skewness
# ---------------------------------------------------------------------------


def skewness(data: pd.Series, period: int = 20) -> pd.Series:
    """Rolling skewness.

    Measures the asymmetry of the distribution of values within the
    rolling window. Uses Fisher's definition (bias-corrected).

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Rolling window length (must be >= 3).

    Returns
    -------
    pd.Series
        Skewness values (unbounded).

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series(range(30), dtype=float)
    >>> skewness(close, period=20)  # doctest: +SKIP
    """
    _validate_series(data)
    _validate_period(period)

    result = data.rolling(window=period, min_periods=period).skew()
    result.name = "skewness"
    return result


# ---------------------------------------------------------------------------
# Kurtosis
# ---------------------------------------------------------------------------


def kurtosis(data: pd.Series, period: int = 20) -> pd.Series:
    """Rolling kurtosis (excess kurtosis, Fisher's definition).

    Measures the tailedness of the distribution of values within the
    rolling window. Normal distribution has excess kurtosis of 0.

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Rolling window length (must be >= 4).

    Returns
    -------
    pd.Series
        Excess kurtosis values (unbounded).

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series(range(30), dtype=float)
    >>> kurtosis(close, period=20)  # doctest: +SKIP
    """
    _validate_series(data)
    _validate_period(period)

    result = data.rolling(window=period, min_periods=period).kurt()
    result.name = "kurtosis"
    return result


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def entropy(data: pd.Series, period: int = 20, bins: int = 10) -> pd.Series:
    """Rolling Shannon entropy of binned price changes.

    Discretises the price changes within the rolling window into *bins*
    equal-width bins and computes Shannon entropy in nats (natural log).

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 20
        Rolling window length.
    bins : int, default 10
        Number of histogram bins for discretisation.

    Returns
    -------
    pd.Series
        Shannon entropy values (>= 0).

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series(range(30), dtype=float)
    >>> entropy(close, period=20, bins=5)  # doctest: +SKIP
    """
    _validate_series(data)
    _validate_period(period)

    changes = data.diff()

    def _shannon(window: np.ndarray) -> float:
        counts, _ = np.histogram(window, bins=bins)
        probs = counts / counts.sum()
        # Filter out zero-probability bins to avoid log(0)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    result = changes.rolling(window=period, min_periods=period).apply(
        _shannon, raw=True
    )
    result.name = "entropy"
    return result


# ---------------------------------------------------------------------------
# Hurst Exponent
# ---------------------------------------------------------------------------


def hurst_exponent(data: pd.Series, period: int = 100) -> pd.Series:
    """Rolling Hurst exponent via the rescaled range (R/S) method.

    * H < 0.5: mean-reverting
    * H = 0.5: random walk
    * H > 0.5: trending

    Parameters
    ----------
    data : pd.Series
        Price series.
    period : int, default 100
        Rolling window length (should be >= 20 for reliable estimates).

    Returns
    -------
    pd.Series
        Hurst exponent estimates in roughly [0, 1].

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> close = pd.Series(100 + np.cumsum(np.random.randn(200)))
    >>> hurst_exponent(close, period=100)  # doctest: +SKIP
    """
    _validate_series(data)
    _validate_period(period)

    def _rs_hurst(window: np.ndarray) -> float:
        n = len(window)
        if n < 20:
            return np.nan

        # Use log returns for stationarity
        returns = np.diff(np.log(window))
        if len(returns) == 0:
            return np.nan

        mean_ret = np.mean(returns)
        deviations = returns - mean_ret
        cumdev = np.cumsum(deviations)
        r = np.max(cumdev) - np.min(cumdev)
        s = np.std(returns, ddof=1)

        if s == 0 or r == 0:
            return np.nan

        # Use multiple sub-ranges for a more robust estimate
        rs_values = []
        lengths = []
        m = n - 1  # number of returns
        for div in [2, 4, 8]:
            sub_len = m // div
            if sub_len < 4:
                continue
            rs_sub = []
            for i in range(div):
                sub = returns[i * sub_len : (i + 1) * sub_len]
                sub_mean = np.mean(sub)
                sub_dev = np.cumsum(sub - sub_mean)
                sub_r = np.max(sub_dev) - np.min(sub_dev)
                sub_s = np.std(sub, ddof=1)
                if sub_s > 0:
                    rs_sub.append(sub_r / sub_s)
            if rs_sub:
                rs_values.append(np.log(np.mean(rs_sub)))
                lengths.append(np.log(sub_len))

        # Add full-range R/S
        rs_values.append(np.log(r / s))
        lengths.append(np.log(m))

        if len(rs_values) < 2:
            # Fallback: simple R/S estimate
            return float(np.log(r / s) / np.log(m))

        # Linear regression of log(R/S) vs log(n)
        coeffs = np.polyfit(lengths, rs_values, 1)
        return float(coeffs[0])

    result = data.rolling(window=period, min_periods=period).apply(
        _rs_hurst, raw=True
    )
    result.name = "hurst_exponent"
    return result


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def correlation(
    data: pd.Series,
    other: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Rolling Pearson correlation between two series.

    Parameters
    ----------
    data : pd.Series
        First price series.
    other : pd.Series
        Second price series.
    period : int, default 20
        Rolling window length.

    Returns
    -------
    pd.Series
        Correlation values in [-1, 1].

    Example
    -------
    >>> import pandas as pd
    >>> x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    >>> y = pd.Series([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=float)
    >>> correlation(x, y, period=5)  # doctest: +SKIP
    """
    _validate_series(data, "data")
    _validate_series(other, "other")
    _validate_period(period)

    result = data.rolling(window=period, min_periods=period).corr(other)
    result.name = "correlation"
    return result


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


def beta(
    data: pd.Series,
    benchmark: pd.Series,
    period: int = 60,
) -> pd.Series:
    """Rolling beta (OLS slope of data returns vs benchmark returns).

    Parameters
    ----------
    data : pd.Series
        Asset price series.
    benchmark : pd.Series
        Benchmark price series.
    period : int, default 60
        Rolling window length.

    Returns
    -------
    pd.Series
        Beta values (unbounded).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> stock = pd.Series(100 + np.cumsum(np.random.randn(100)))
    >>> market = pd.Series(100 + np.cumsum(np.random.randn(100)))
    >>> beta(stock, market, period=30)  # doctest: +SKIP
    """
    _validate_series(data, "data")
    _validate_series(benchmark, "benchmark")
    _validate_period(period)

    data_ret = data.pct_change()
    bench_ret = benchmark.pct_change()

    covar = data_ret.rolling(window=period, min_periods=period).cov(bench_ret)
    var_bench = bench_ret.rolling(window=period, min_periods=period).var(ddof=1)

    result = covar / var_bench
    result.name = "beta"
    return result


# ---------------------------------------------------------------------------
# R-Squared
# ---------------------------------------------------------------------------


def r_squared(
    data: pd.Series,
    benchmark: pd.Series,
    period: int = 60,
) -> pd.Series:
    """Rolling R-squared (coefficient of determination).

    Computed as the square of the rolling Pearson correlation of returns.

    Parameters
    ----------
    data : pd.Series
        Asset price series.
    benchmark : pd.Series
        Benchmark price series.
    period : int, default 60
        Rolling window length.

    Returns
    -------
    pd.Series
        R-squared values in [0, 1].

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(42)
    >>> stock = pd.Series(100 + np.cumsum(np.random.randn(100)))
    >>> market = pd.Series(100 + np.cumsum(np.random.randn(100)))
    >>> r_squared(stock, market, period=30)  # doctest: +SKIP
    """
    _validate_series(data, "data")
    _validate_series(benchmark, "benchmark")
    _validate_period(period)

    data_ret = data.pct_change()
    bench_ret = benchmark.pct_change()

    corr = data_ret.rolling(window=period, min_periods=period).corr(bench_ret)
    result = corr ** 2
    result.name = "r_squared"
    return result


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------


def information_coefficient(
    data: pd.Series,
    other: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Rolling information coefficient (Spearman rank correlation).

    Measures the rolling rank correlation between two series, commonly
    used to evaluate forecast skill.

    Parameters
    ----------
    data : pd.Series
        Forecast or signal series.
    other : pd.Series
        Realised outcome series.
    period : int, default 20
        Rolling window length.

    Returns
    -------
    pd.Series
        IC values in [-1, 1].

    Example
    -------
    >>> import pandas as pd
    >>> forecast = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    >>> actual = pd.Series([2, 1, 4, 3, 6, 5, 8, 7, 10, 9], dtype=float)
    >>> information_coefficient(forecast, actual, period=5)  # doctest: +SKIP
    """
    _validate_series(data, "data")
    _validate_series(other, "other")
    _validate_period(period)

    def _rank_corr(idx: int) -> float:
        start = idx - period + 1
        if start < 0:
            return np.nan
        x = data.iloc[start : idx + 1].values
        y = other.iloc[start : idx + 1].values
        # Compute Spearman rank correlation manually
        rx = _rankdata(x)
        ry = _rankdata(y)
        n = len(rx)
        d_sq = np.sum((rx - ry) ** 2)
        denom = n * (n * n - 1)
        if denom == 0:
            return np.nan
        return float(1.0 - 6.0 * d_sq / denom)

    # Use rolling apply with index-based access for Spearman
    indices = np.arange(len(data))
    idx_series = pd.Series(indices, index=data.index)

    def _spearman_window(window: np.ndarray) -> float:
        idx_end = int(window[-1])
        start = idx_end - period + 1
        if start < 0:
            return np.nan
        x = data.iloc[start : idx_end + 1].values
        y = other.iloc[start : idx_end + 1].values
        rx = _rankdata(x)
        ry = _rankdata(y)
        n = len(rx)
        d_sq = np.sum((rx - ry) ** 2)
        denom = n * (n * n - 1)
        if denom == 0:
            return np.nan
        return float(1.0 - 6.0 * d_sq / denom)

    result = idx_series.rolling(window=period, min_periods=period).apply(
        _spearman_window, raw=True
    )
    result.name = "information_coefficient"
    return result


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Assign ranks to data (average method for ties)."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # Handle ties with average rank
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[order[j + 1]] == arr[order[j]]:
            j += 1
        if j > i:
            avg_rank = np.mean(ranks[order[i : j + 1]])
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks
