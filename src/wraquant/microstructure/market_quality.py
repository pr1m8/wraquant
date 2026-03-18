"""Market quality metrics.

Provides bid-ask spread measures, market depth indicators, resilience
metrics, and variance ratio tests for assessing overall market quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def quoted_spread(
    bid: pd.Series | NDArray[np.floating],
    ask: pd.Series | NDArray[np.floating],
) -> pd.Series | NDArray[np.floating]:
    """Quoted bid-ask spread: ask - bid.

    Parameters:
        bid: Best bid prices.
        ask: Best ask prices.

    Returns:
        Absolute quoted spread.
    """
    return np.asarray(ask) - np.asarray(bid)


def relative_spread(
    bid: pd.Series | NDArray[np.floating],
    ask: pd.Series | NDArray[np.floating],
) -> pd.Series | NDArray[np.floating]:
    """Relative spread: (ask - bid) / midpoint.

    Parameters:
        bid: Best bid prices.
        ask: Best ask prices.

    Returns:
        Relative spread as a fraction of the midpoint.
    """
    bid_arr = np.asarray(bid, dtype=np.float64)
    ask_arr = np.asarray(ask, dtype=np.float64)
    mid = (bid_arr + ask_arr) / 2.0
    result = (ask_arr - bid_arr) / mid
    if isinstance(bid, pd.Series):
        return pd.Series(result, index=bid.index, name="relative_spread")
    return result


def depth(
    bid_volume: pd.DataFrame | NDArray[np.floating],
    ask_volume: pd.DataFrame | NDArray[np.floating],
    levels: int = 5,
) -> pd.Series | NDArray[np.floating]:
    """Market depth: total volume available at the top N price levels.

    Parameters:
        bid_volume: Volume at each bid level. Columns (or columns in the
            2-D array) represent successive price levels from best to worst.
        ask_volume: Volume at each ask level, same layout as *bid_volume*.
        levels: Number of price levels to include.

    Returns:
        Total depth (bid + ask) summed across the requested levels.
    """
    bid_arr = np.asarray(bid_volume, dtype=np.float64)
    ask_arr = np.asarray(ask_volume, dtype=np.float64)

    # Handle 1-D vs 2-D
    if bid_arr.ndim == 1:
        bid_sum = np.sum(bid_arr[:levels])
        ask_sum = np.sum(ask_arr[:levels])
        return bid_sum + ask_sum

    bid_sum = np.sum(bid_arr[:, :levels], axis=1)
    ask_sum = np.sum(ask_arr[:, :levels], axis=1)
    total = bid_sum + ask_sum

    if isinstance(bid_volume, pd.DataFrame):
        return pd.Series(total, index=bid_volume.index, name="depth")
    return total


def resiliency(
    spreads: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Spread resiliency: how quickly the spread recovers after a shock.

    Measured as the negative autocorrelation of spread changes. A higher
    value indicates a more resilient market (spreads revert faster).

    Parameters:
        spreads: Time series of quoted or effective spreads.
        window: Rolling window for estimating autocorrelation of
            spread changes.

    Returns:
        Rolling resiliency measure.
    """
    ds = spreads.diff()
    # Negative first-order autocorrelation of spread changes
    resilience = -ds.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan,
        raw=False,
    )
    resilience.name = "resiliency"
    return resilience


def variance_ratio(
    prices: pd.Series,
    short_period: int = 2,
    long_period: int = 10,
) -> dict[str, float]:
    """Lo-MacKinlay (1988) variance ratio test.

    Tests the random walk hypothesis by comparing the variance of
    *long_period* returns to *short_period* returns, scaled appropriately.
    Under a random walk, the ratio equals 1.

    Parameters:
        prices: Price series (levels, not returns).
        short_period: Short return horizon (default 2).
        long_period: Long return horizon (must be a multiple of
            *short_period* for a clean comparison, but this is not
            enforced).

    Returns:
        Dictionary with keys:

        - ``'vr'``: Variance ratio.
        - ``'z_stat'``: Asymptotic z-statistic under IID assumption.
        - ``'p_value'``: Two-sided p-value.
    """
    from scipy.stats import norm

    log_prices = np.log(prices).values
    n = len(log_prices)

    # Returns at two horizons (lagged differences, not n-th order diff)
    ret_short = log_prices[short_period:] - log_prices[:-short_period]
    ret_long = log_prices[long_period:] - log_prices[:-long_period]

    var_short = np.var(ret_short, ddof=1)
    var_long = np.var(ret_long, ddof=1)

    q = long_period / short_period
    vr = var_long / (q * var_short) if var_short > 0 else np.nan

    # Asymptotic z-statistic under IID
    nq = len(ret_short)
    se = np.sqrt(2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * nq))
    z_stat = (vr - 1.0) / se if se > 0 else np.nan
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan

    return {"vr": float(vr), "z_stat": float(z_stat), "p_value": float(p_value)}
