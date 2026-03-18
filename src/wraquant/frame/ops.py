"""Common financial time series operations.

Backend-agnostic operations on price/return series. All functions accept
pandas Series/DataFrames as primary input and return the same type.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def returns(
    prices: pd.Series | pd.DataFrame,
    periods: int = 1,
) -> pd.Series | pd.DataFrame:
    """Calculate simple (arithmetic) returns.

    Parameters:
        prices: Price series or DataFrame.
        periods: Number of periods for return calculation.

    Returns:
        Simple returns: (P_t / P_{t-n}) - 1

    Example:
        >>> import pandas as pd
        >>> p = pd.Series([100, 102, 101, 105])
        >>> returns(p)
        0         NaN
        1    0.020000
        2   -0.009804
        3    0.039604
        dtype: float64
    """
    return prices.pct_change(periods=periods)


def log_returns(
    prices: pd.Series | pd.DataFrame,
    periods: int = 1,
) -> pd.Series | pd.DataFrame:
    """Calculate logarithmic returns.

    Parameters:
        prices: Price series or DataFrame.
        periods: Number of periods for return calculation.

    Returns:
        Log returns: ln(P_t / P_{t-n})
    """
    return np.log(prices / prices.shift(periods))


def cumulative_returns(
    simple_returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Calculate cumulative returns from simple returns.

    Parameters:
        simple_returns: Simple return series (not log returns).

    Returns:
        Cumulative return series starting from 0.
    """
    return (1 + simple_returns).cumprod() - 1


def drawdowns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate drawdown series from prices.

    Parameters:
        prices: Price series or DataFrame.

    Returns:
        Drawdown series (negative values representing decline from peak).
    """
    peak = prices.cummax()
    return (prices - peak) / peak


def rolling_mean(
    data: pd.Series | pd.DataFrame,
    window: int,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Calculate rolling mean.

    Parameters:
        data: Input series or DataFrame.
        window: Rolling window size.
        min_periods: Minimum observations required.

    Returns:
        Rolling mean series.
    """
    return data.rolling(window=window, min_periods=min_periods or window).mean()


def rolling_std(
    data: pd.Series | pd.DataFrame,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.Series | pd.DataFrame:
    """Calculate rolling standard deviation.

    Parameters:
        data: Input series or DataFrame.
        window: Rolling window size.
        min_periods: Minimum observations required.
        ddof: Delta degrees of freedom.

    Returns:
        Rolling standard deviation series.
    """
    return data.rolling(window=window, min_periods=min_periods or window).std(ddof=ddof)


def ewm_mean(
    data: pd.Series | pd.DataFrame,
    span: int | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
) -> pd.Series | pd.DataFrame:
    """Calculate exponentially weighted moving average.

    Parameters:
        data: Input series or DataFrame.
        span: EWM span parameter.
        halflife: EWM half-life parameter.
        alpha: EWM smoothing factor.

    Returns:
        EWMA series.
    """
    return data.ewm(span=span, halflife=halflife, alpha=alpha).mean()


def resample(
    data: pd.Series | pd.DataFrame,
    freq: str,
    agg: str = "last",
) -> pd.Series | pd.DataFrame:
    """Resample time series to a different frequency.

    Parameters:
        data: Input series or DataFrame with DatetimeIndex.
        freq: Target frequency string (e.g., 'W', 'M', 'Q').
        agg: Aggregation method ('last', 'first', 'mean', 'sum', 'ohlc').

    Returns:
        Resampled series.
    """
    resampler = data.resample(freq)
    if agg == "ohlc":
        return resampler.ohlc()
    return getattr(resampler, agg)()
