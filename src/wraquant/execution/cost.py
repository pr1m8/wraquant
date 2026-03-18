"""Transaction cost analysis.

Provides slippage calculation, commission accounting, total execution
cost breakdown, and market impact models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def slippage(
    execution_price: float | NDArray[np.floating],
    benchmark_price: float | NDArray[np.floating],
    side: str = "buy",
) -> float | NDArray[np.floating]:
    """Per-trade slippage relative to a benchmark.

    For buys, slippage is positive when the execution price exceeds the
    benchmark.  For sells, it is positive when the execution price is
    below the benchmark.

    Parameters:
        execution_price: Actual fill price(s).
        benchmark_price: Reference price(s) (e.g., mid, arrival).
        side: ``'buy'`` or ``'sell'``.

    Returns:
        Signed slippage value(s).
    """
    ep = np.asarray(execution_price, dtype=np.float64)
    bp = np.asarray(benchmark_price, dtype=np.float64)

    if side == "buy":
        result = ep - bp
    elif side == "sell":
        result = bp - ep
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    return float(result) if np.ndim(result) == 0 else result


def commission_cost(
    qty: float | NDArray[np.floating],
    price: float | NDArray[np.floating],
    rate: float = 0.001,
) -> float | NDArray[np.floating]:
    """Commission cost calculation.

    Parameters:
        qty: Number of shares per trade.
        price: Price per share.
        rate: Commission rate as a fraction of notional value.

    Returns:
        Commission cost(s).
    """
    q = np.asarray(qty, dtype=np.float64)
    p = np.asarray(price, dtype=np.float64)
    result = np.abs(q) * p * rate
    return float(result) if np.ndim(result) == 0 else result


def total_cost(
    trades_df: pd.DataFrame,
    commission_rate: float = 0.001,
) -> dict[str, float]:
    """Total execution cost breakdown from a trades DataFrame.

    The DataFrame must contain columns ``'execution_price'``,
    ``'benchmark_price'``, ``'qty'``, and ``'side'``.

    Parameters:
        trades_df: DataFrame with one row per fill.
        commission_rate: Commission rate as a fraction of notional value.

    Returns:
        Dictionary with ``'total_slippage'``, ``'total_commission'``,
        ``'total_cost'``, ``'cost_bps'``, and ``'n_trades'``.
    """
    required = {"execution_price", "benchmark_price", "qty", "side"}
    missing = required - set(trades_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    total_slippage = 0.0
    total_commission = 0.0
    notional = 0.0

    for _, row in trades_df.iterrows():
        s = slippage(row["execution_price"], row["benchmark_price"], row["side"])
        c = commission_cost(row["qty"], row["execution_price"], commission_rate)
        n = abs(row["qty"]) * row["execution_price"]
        total_slippage += s * abs(row["qty"])
        total_commission += c
        notional += n

    cost = total_slippage + total_commission
    cost_bps = (cost / notional) * 10_000 if notional > 0 else 0.0

    return {
        "total_slippage": total_slippage,
        "total_commission": total_commission,
        "total_cost": cost,
        "cost_bps": cost_bps,
        "n_trades": len(trades_df),
    }


def market_impact_model(
    qty: float,
    avg_daily_volume: float,
    volatility: float,
    model: str = "sqrt",
) -> float:
    """Estimate market impact using a parametric model.

    Parameters:
        qty: Order quantity (shares).
        avg_daily_volume: Average daily volume.
        volatility: Daily price volatility (standard deviation of
            returns).
        model: Impact model to use.  ``'sqrt'`` for the square-root
            model (Barra / Kissell-Glantz), ``'linear'`` for a simple
            linear model.

    Returns:
        Estimated market impact in price units (same scale as
        *volatility*).
    """
    participation = abs(qty) / avg_daily_volume if avg_daily_volume > 0 else 0.0

    if model == "sqrt":
        # Square-root model: impact ~ sigma * sqrt(Q / ADV)
        return volatility * np.sqrt(participation)
    elif model == "linear":
        # Linear model: impact ~ sigma * (Q / ADV)
        return volatility * participation
    else:
        raise ValueError(f"model must be 'sqrt' or 'linear', got {model!r}")
