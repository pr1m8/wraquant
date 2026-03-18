"""Descriptive statistics for financial return and price series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def summary_stats(returns: pd.Series) -> dict:
    """Compute summary statistics for a return series.

    Parameters:
        returns: Simple return series.

    Returns:
        Dictionary with mean, std, skew, kurtosis, min, max, and count.
    """
    return {
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "skew": float(sp_stats.skew(returns.dropna(), bias=False)),
        "kurtosis": float(sp_stats.kurtosis(returns.dropna(), bias=False)),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "count": int(returns.count()),
    }


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized return from a simple return series.

    Parameters:
        returns: Simple return series.
        periods_per_year: Number of periods per year (252 for daily).

    Returns:
        Annualized return as a float.
    """
    total = (1 + returns).prod()
    n = len(returns)
    return float(total ** (periods_per_year / n) - 1)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized volatility from a simple return series.

    Parameters:
        returns: Simple return series.
        periods_per_year: Number of periods per year (252 for daily).

    Returns:
        Annualized volatility as a float.
    """
    return float(returns.std() * np.sqrt(periods_per_year))


def max_drawdown(prices: pd.Series) -> float:
    """Compute maximum drawdown from a price series.

    Parameters:
        prices: Price series (not returns).

    Returns:
        Maximum drawdown as a negative float (e.g., -0.25 for 25% drawdown).
    """
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute the Calmar ratio (annualized return / max drawdown).

    Parameters:
        returns: Simple return series.
        periods_per_year: Number of periods per year.

    Returns:
        Calmar ratio as a float.
    """
    prices = (1 + returns).cumprod()
    mdd = max_drawdown(prices)
    if mdd == 0:
        return 0.0
    ann_ret = annualized_return(returns, periods_per_year)
    return float(ann_ret / abs(mdd))


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Compute the Omega ratio.

    The Omega ratio is the probability-weighted ratio of gains versus
    losses relative to a threshold.

    Parameters:
        returns: Simple return series.
        threshold: Return threshold (default 0).

    Returns:
        Omega ratio as a float.
    """
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess <= 0].sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)
