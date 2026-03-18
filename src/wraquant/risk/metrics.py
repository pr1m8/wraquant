"""Core risk and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio.

    Parameters:
        returns: Simple return series.
        risk_free: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        Annualized Sharpe ratio.
    """
    excess = returns - risk_free / periods_per_year
    mean_excess = excess.mean()
    std = excess.std()
    if std == 0:
        return 0.0
    return float(mean_excess / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (downside risk only).

    Parameters:
        returns: Simple return series.
        risk_free: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        Annualized Sortino ratio.
    """
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = np.sqrt((downside**2).mean())
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
) -> float:
    """Information ratio (active return / tracking error).

    Parameters:
        returns: Portfolio return series.
        benchmark: Benchmark return series.

    Returns:
        Information ratio (not annualized).
    """
    active = returns - benchmark
    te = active.std()
    if te == 0:
        return 0.0
    return float(active.mean() / te)


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown from a price series.

    Parameters:
        prices: Price series (not returns).

    Returns:
        Maximum drawdown as a negative float.
    """
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())


def hit_ratio(returns: pd.Series) -> float:
    """Fraction of positive return periods.

    Parameters:
        returns: Simple return series.

    Returns:
        Hit ratio between 0 and 1.
    """
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    return float((clean > 0).sum() / len(clean))
