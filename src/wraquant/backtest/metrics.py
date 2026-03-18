"""Backtesting performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.risk.metrics import (
    max_drawdown as _risk_max_drawdown,
    sharpe_ratio as _risk_sharpe,
    sortino_ratio as _risk_sortino,
)


def performance_summary(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Calculate comprehensive performance metrics.

    Parameters:
        returns: Portfolio return series.
        risk_free: Annual risk-free rate.
        periods_per_year: Number of trading periods per year.

    Returns:
        Dict with performance metrics.
    """
    total_return = float((1 + returns).prod() - 1)
    n_periods = len(returns)
    ann_factor = periods_per_year / n_periods if n_periods > 0 else 1

    ann_return = float((1 + total_return) ** ann_factor - 1)
    ann_vol = float(returns.std() * np.sqrt(periods_per_year))

    sharpe = _risk_sharpe(returns, risk_free=risk_free, periods_per_year=periods_per_year)

    sortino = _risk_sortino(returns, risk_free=risk_free, periods_per_year=periods_per_year)

    # Max drawdown — risk.metrics.max_drawdown expects a price series
    cumulative = (1 + returns).cumprod()
    max_dd = _risk_max_drawdown(cumulative)

    # Calmar (no canonical implementation in risk module yet)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate
    n_positive = int((returns > 0).sum())
    win_rate = n_positive / n_periods if n_periods > 0 else 0.0

    # Profit factor
    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))
    profit_factor = gains / losses if losses > 0 else float("inf")

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_periods": n_periods,
    }
