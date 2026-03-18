"""Performance and comparison indicators.

This module provides indicators that measure asset performance relative to
a benchmark or on an absolute basis: relative strength, alpha, tracking
error, drawdowns, and risk-adjusted return metrics. All functions accept
``pd.Series`` inputs and return ``pd.Series`` (or ``dict`` where noted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "relative_performance",
    "mansfield_rsi",
    "alpha",
    "tracking_error",
    "up_down_capture",
    "drawdown",
    "max_drawdown_rolling",
    "pain_index",
    "gain_loss_ratio",
    "profit_factor",
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
# Relative Performance
# ---------------------------------------------------------------------------


def relative_performance(
    asset: pd.Series,
    benchmark: pd.Series,
) -> pd.Series:
    """Relative Performance — ratio of asset to benchmark, normalized to 100.

    ``RP = (asset / benchmark) / (asset.iloc[0] / benchmark.iloc[0]) * 100``

    A rising line means the asset is outperforming the benchmark.

    Parameters
    ----------
    asset : pd.Series
        Asset price series.
    benchmark : pd.Series
        Benchmark price series.

    Returns
    -------
    pd.Series
        Relative performance, starting at 100.

    Example
    -------
    >>> asset = pd.Series([100, 105, 110])
    >>> bench = pd.Series([100, 102, 104])
    >>> relative_performance(asset, bench)
    """
    _validate_series(asset, "asset")
    _validate_series(benchmark, "benchmark")

    ratio = asset / benchmark.replace(0, np.nan)
    # Normalize so the first valid ratio equals 100
    first_valid = ratio.first_valid_index()
    if first_valid is not None:
        result = (ratio / ratio.loc[first_valid]) * 100.0
    else:
        result = ratio
    result.name = "relative_performance"
    return result


# ---------------------------------------------------------------------------
# Mansfield Relative Strength
# ---------------------------------------------------------------------------


def mansfield_rsi(
    asset: pd.Series,
    benchmark: pd.Series,
    period: int = 52,
) -> pd.Series:
    """Mansfield Relative Strength (not Wilder RSI).

    Compares the asset/benchmark ratio to its own simple moving average,
    expressing the result as a percentage deviation.

    ``MRS = ((asset / benchmark) / SMA(asset / benchmark, period) - 1) * 100``

    Parameters
    ----------
    asset : pd.Series
        Asset price series.
    benchmark : pd.Series
        Benchmark price series.
    period : int, default 52
        SMA look-back period.

    Returns
    -------
    pd.Series
        Mansfield RS values (percentage above/below zero line).

    Example
    -------
    >>> result = mansfield_rsi(asset, benchmark, period=52)
    """
    _validate_series(asset, "asset")
    _validate_series(benchmark, "benchmark")
    _validate_period(period)

    ratio = asset / benchmark.replace(0, np.nan)
    sma = ratio.rolling(window=period, min_periods=period).mean()
    result = ((ratio / sma) - 1.0) * 100.0
    result.name = "mansfield_rsi"
    return result


# ---------------------------------------------------------------------------
# Rolling Jensen's Alpha
# ---------------------------------------------------------------------------


def alpha(
    asset: pd.Series,
    benchmark: pd.Series,
    window: int = 60,
    risk_free: float = 0.0,
) -> pd.Series:
    """Rolling Jensen's Alpha vs. benchmark.

    Computes alpha as the intercept of the rolling OLS regression of
    excess asset returns on excess benchmark returns.

    Parameters
    ----------
    asset : pd.Series
        Asset price series.
    benchmark : pd.Series
        Benchmark price series.
    window : int, default 60
        Rolling window size (number of periods).
    risk_free : float, default 0.0
        Per-period risk-free rate.

    Returns
    -------
    pd.Series
        Rolling alpha values (annualization depends on input frequency).

    Example
    -------
    >>> result = alpha(asset, benchmark, window=60)
    """
    _validate_series(asset, "asset")
    _validate_series(benchmark, "benchmark")
    _validate_period(window, "window")

    asset_ret = asset.pct_change() - risk_free
    bench_ret = benchmark.pct_change() - risk_free

    def _ols_alpha(chunk: np.ndarray) -> float:
        """OLS alpha from a stacked [asset_ret, bench_ret] chunk."""
        n = len(chunk) // 2
        y = chunk[:n]
        x = chunk[n:]
        if np.any(np.isnan(y)) or np.any(np.isnan(x)):
            return np.nan
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx == 0:
            return np.nan
        beta = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
        return y_mean - beta * x_mean

    combined = pd.concat([asset_ret, bench_ret], axis=0, ignore_index=True)
    n = len(asset_ret)
    result_values = np.full(n, np.nan)

    asset_arr = asset_ret.values
    bench_arr = bench_ret.values

    for i in range(window, n):
        chunk = np.concatenate([asset_arr[i - window + 1 : i + 1],
                                bench_arr[i - window + 1 : i + 1]])
        result_values[i] = _ols_alpha(chunk)

    result = pd.Series(result_values, index=asset.index, name="alpha")
    return result


# ---------------------------------------------------------------------------
# Rolling Tracking Error
# ---------------------------------------------------------------------------


def tracking_error(
    asset: pd.Series,
    benchmark: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling Tracking Error vs. benchmark.

    Tracking error is the standard deviation of the difference in returns
    between the asset and the benchmark over a rolling window.

    Parameters
    ----------
    asset : pd.Series
        Asset price series.
    benchmark : pd.Series
        Benchmark price series.
    window : int, default 60
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling tracking error values.

    Example
    -------
    >>> result = tracking_error(asset, benchmark, window=60)
    """
    _validate_series(asset, "asset")
    _validate_series(benchmark, "benchmark")
    _validate_period(window, "window")

    asset_ret = asset.pct_change()
    bench_ret = benchmark.pct_change()
    diff = asset_ret - bench_ret
    result = diff.rolling(window=window, min_periods=window).std()
    result.name = "tracking_error"
    return result


# ---------------------------------------------------------------------------
# Up/Down Capture Ratio
# ---------------------------------------------------------------------------


def up_down_capture(
    asset: pd.Series,
    benchmark: pd.Series,
) -> dict[str, float]:
    """Up/Down Market Capture Ratio.

    Measures how much of the benchmark's up- and down-market returns the
    asset captures.

    Parameters
    ----------
    asset : pd.Series
        Asset price series.
    benchmark : pd.Series
        Benchmark price series.

    Returns
    -------
    dict[str, float]
        ``up_capture``, ``down_capture``, and ``capture_ratio`` (up/down).

    Example
    -------
    >>> result = up_down_capture(asset, benchmark)
    >>> result["up_capture"]
    """
    _validate_series(asset, "asset")
    _validate_series(benchmark, "benchmark")

    asset_ret = asset.pct_change().dropna()
    bench_ret = benchmark.pct_change().dropna()

    # Align
    common = asset_ret.index.intersection(bench_ret.index)
    asset_ret = asset_ret.loc[common]
    bench_ret = bench_ret.loc[common]

    up_mask = bench_ret > 0
    down_mask = bench_ret < 0

    if up_mask.sum() > 0:
        up_capture = asset_ret[up_mask].mean() / bench_ret[up_mask].mean() * 100.0
    else:
        up_capture = np.nan

    if down_mask.sum() > 0:
        down_capture = asset_ret[down_mask].mean() / bench_ret[down_mask].mean() * 100.0
    else:
        down_capture = np.nan

    if not np.isnan(down_capture) and down_capture != 0:
        capture_ratio = up_capture / down_capture
    else:
        capture_ratio = np.nan

    return {
        "up_capture": float(up_capture),
        "down_capture": float(down_capture),
        "capture_ratio": float(capture_ratio),
    }


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


def drawdown(data: pd.Series) -> pd.Series:
    """Drawdown from peak — current decline from running maximum.

    ``DD = (data - running_max) / running_max``

    Returns non-positive values (0 at peaks, negative during drawdowns).

    Parameters
    ----------
    data : pd.Series
        Price or equity curve.

    Returns
    -------
    pd.Series
        Drawdown values (non-positive fractions).

    Example
    -------
    >>> prices = pd.Series([100, 105, 102, 108, 103])
    >>> drawdown(prices)
    """
    _validate_series(data)

    running_max = data.cummax()
    result = (data - running_max) / running_max.replace(0, np.nan)
    result.name = "drawdown"
    return result


# ---------------------------------------------------------------------------
# Rolling Max Drawdown
# ---------------------------------------------------------------------------


def max_drawdown_rolling(
    data: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling maximum drawdown over a look-back window.

    For each point, computes the worst drawdown experienced within the
    trailing *window* periods.

    Parameters
    ----------
    data : pd.Series
        Price or equity curve.
    window : int, default 252
        Rolling look-back window.

    Returns
    -------
    pd.Series
        Rolling max drawdown values (non-positive fractions).

    Example
    -------
    >>> result = max_drawdown_rolling(prices, window=60)
    """
    _validate_series(data)
    _validate_period(window, "window")

    def _max_dd(chunk: np.ndarray) -> float:
        """Max drawdown within a price chunk."""
        peak = chunk[0]
        max_dd = 0.0
        for price in chunk:
            if price > peak:
                peak = price
            dd = (price - peak) / peak if peak != 0 else 0.0
            if dd < max_dd:
                max_dd = dd
        return max_dd

    result = data.rolling(window=window, min_periods=window).apply(_max_dd, raw=True)
    result.name = "max_drawdown_rolling"
    return result


# ---------------------------------------------------------------------------
# Pain Index
# ---------------------------------------------------------------------------


def pain_index(
    data: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Pain Index — mean of absolute drawdowns over a rolling window.

    The Pain Index averages the magnitude of drawdowns; a higher value
    indicates more sustained or deeper drawdowns.

    Parameters
    ----------
    data : pd.Series
        Price or equity curve.
    window : int, default 252
        Rolling look-back window.

    Returns
    -------
    pd.Series
        Pain Index values (non-negative).

    Example
    -------
    >>> result = pain_index(prices, window=60)
    """
    _validate_series(data)
    _validate_period(window, "window")

    dd = drawdown(data)
    result = dd.abs().rolling(window=window, min_periods=window).mean()
    result.name = "pain_index"
    return result


# ---------------------------------------------------------------------------
# Gain/Loss Ratio
# ---------------------------------------------------------------------------


def gain_loss_ratio(
    data: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Gain/Loss Ratio — average gain / average loss over a rolling window.

    Uses the per-period returns (percentage change) of the input series.
    Values above 1.0 indicate larger average gains than average losses.

    Parameters
    ----------
    data : pd.Series
        Price series.
    window : int, default 20
        Rolling look-back window.

    Returns
    -------
    pd.Series
        Gain/loss ratio values (NaN when no losses or no gains in window).

    Example
    -------
    >>> result = gain_loss_ratio(prices, window=20)
    """
    _validate_series(data)
    _validate_period(window, "window")

    returns = data.pct_change()
    gains = returns.clip(lower=0.0)
    losses = (-returns).clip(lower=0.0)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    result = avg_gain / avg_loss.replace(0, np.nan)
    result.name = "gain_loss_ratio"
    return result


# ---------------------------------------------------------------------------
# Profit Factor
# ---------------------------------------------------------------------------


def profit_factor(
    data: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Profit Factor — sum of gains / sum of losses over a rolling window.

    Uses the per-period returns (percentage change) of the input series.
    Values above 1.0 indicate total gains exceed total losses.

    Parameters
    ----------
    data : pd.Series
        Price series.
    window : int, default 20
        Rolling look-back window.

    Returns
    -------
    pd.Series
        Profit factor values (NaN when no losses in window).

    Example
    -------
    >>> result = profit_factor(prices, window=20)
    """
    _validate_series(data)
    _validate_period(window, "window")

    returns = data.pct_change()
    gains = returns.clip(lower=0.0)
    losses = (-returns).clip(lower=0.0)

    sum_gains = gains.rolling(window=window, min_periods=window).sum()
    sum_losses = losses.rolling(window=window, min_periods=window).sum()

    result = sum_gains / sum_losses.replace(0, np.nan)
    result.name = "profit_factor"
    return result
