"""Data transformations for financial time series."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from wraquant.core._coerce import coerce_series


def to_returns(
    prices: pd.Series | pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
) -> pd.Series | pd.DataFrame:
    """Convert a price series to returns.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price series indexed by date.
    method : {'simple', 'log'}, default 'simple'
        ``'simple'`` computes arithmetic returns ``(P_t / P_{t-1}) - 1``.
        ``'log'`` computes logarithmic returns ``ln(P_t / P_{t-1})``.

    Returns
    -------
    pd.Series or pd.DataFrame
        Return series.  The first row will be ``NaN``.
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        prices = coerce_series(prices, name="prices")
    if method == "simple":
        return prices.pct_change()
    if method == "log":
        return np.log(prices / prices.shift(1))
    raise ValueError(f"Unknown method: {method!r}")


def to_prices(
    returns: pd.Series | pd.DataFrame,
    initial_price: float = 100.0,
    method: Literal["simple", "log"] = "simple",
) -> pd.Series | pd.DataFrame:
    """Convert a return series back to prices.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Return series (may contain a leading ``NaN``).
    initial_price : float, default 100.0
        Starting price level.
    method : {'simple', 'log'}, default 'simple'
        Must match the method used to compute the returns.

    Returns
    -------
    pd.Series or pd.DataFrame
        Reconstructed price series beginning at *initial_price*.
    """
    filled = returns.fillna(0.0)

    if method == "simple":
        cumulative = (1.0 + filled).cumprod()
    elif method == "log":
        cumulative = np.exp(filled.cumsum())
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return cumulative * initial_price


def to_excess_returns(
    returns: pd.Series,
    risk_free_rate: pd.Series | float,
) -> pd.Series:
    """Compute excess returns above a risk-free rate.

    Parameters
    ----------
    returns : pd.Series
        Asset return series.
    risk_free_rate : pd.Series or float
        Risk-free rate.  If a ``pd.Series``, it is aligned to *returns*
        by index.

    Returns
    -------
    pd.Series
        Excess return series.
    """
    returns = coerce_series(returns, name="returns")
    if isinstance(risk_free_rate, pd.Series):
        risk_free_rate = risk_free_rate.reindex(returns.index, method="ffill")
    return returns - risk_free_rate


def normalize_prices(
    prices: pd.Series | pd.DataFrame,
    base: float = 100.0,
) -> pd.Series | pd.DataFrame:
    """Rebase a price series so that it starts at *base*.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price series.
    base : float, default 100.0
        Desired starting value.

    Returns
    -------
    pd.Series or pd.DataFrame
        Rebased price series.
    """
    first = prices.iloc[0]
    return prices / first * base


def rank_transform(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Apply a cross-sectional rank transform.

    Values are replaced with their rank divided by the count of non-NaN
    values, producing output in the range ``(0, 1]``.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data.

    Returns
    -------
    pd.Series or pd.DataFrame
        Rank-transformed data.
    """
    if isinstance(data, pd.DataFrame):
        return data.rank(axis=1, pct=True)
    return data.rank(pct=True)


def percentile_rank(
    data: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Compute a rolling percentile rank.

    For each date the value is ranked within the preceding *window*
    observations and expressed as a percentile (0--1).

    Parameters
    ----------
    data : pd.Series
        Input time series.
    window : int, default 252
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling percentile ranks.
    """

    data = coerce_series(data, name="data")

    def _pct_rank(arr: np.ndarray) -> float:
        current = arr[-1]
        if np.isnan(current):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid <= current) / len(valid))

    return data.rolling(window, min_periods=1).apply(_pct_rank, raw=True)


def expanding_zscore(data: pd.Series) -> pd.Series:
    """Compute an expanding-window z-score.

    Parameters
    ----------
    data : pd.Series
        Input time series.

    Returns
    -------
    pd.Series
        Z-scores computed using all data up to and including each point.
    """
    data = coerce_series(data, name="data")
    expanding_mean = data.expanding(min_periods=2).mean()
    expanding_std = data.expanding(min_periods=2).std()
    return (data - expanding_mean) / expanding_std


def rolling_zscore(
    data: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Compute a rolling-window z-score.

    Parameters
    ----------
    data : pd.Series
        Input time series.
    window : int, default 252
        Rolling window size.

    Returns
    -------
    pd.Series
        Z-scores computed over the trailing *window* observations.
    """
    data = coerce_series(data, name="data")
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    return (data - rolling_mean) / rolling_std
