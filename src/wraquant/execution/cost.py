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

    Use slippage to measure the implicit cost of executing a trade at
    a worse price than the benchmark.  Slippage is always defined so
    that positive values represent a cost to the trader.

    For buys, slippage is positive when the execution price exceeds the
    benchmark.  For sells, it is positive when the execution price is
    below the benchmark.

    Parameters:
        execution_price: Actual fill price(s).
        benchmark_price: Reference price(s) (e.g., mid-quote, arrival
            price, or VWAP).
        side: ``'buy'`` or ``'sell'``.

    Returns:
        Signed slippage value(s).  Positive = cost, negative = improvement.

    Example:
        >>> slippage(100.05, 100.00, side='buy')
        0.05
        >>> slippage(99.95, 100.00, side='sell')
        0.05

    See Also:
        total_cost: Aggregate cost breakdown across multiple trades.
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

    Computes the explicit broker commission as a fraction of notional
    value (``|qty| * price * rate``).

    Parameters:
        qty: Number of shares per trade (sign is ignored).
        price: Price per share.
        rate: Commission rate as a fraction of notional value
            (default 0.001 = 10 bps).

    Returns:
        Commission cost(s).  Always non-negative.

    Example:
        >>> commission_cost(1000, 50.0, rate=0.001)
        50.0
        >>> commission_cost(1000, 50.0, rate=0.0005)  # 5 bps
        25.0

    See Also:
        slippage: Implicit execution cost.
        total_cost: Combined slippage + commission.
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

    Use ``total_cost`` to get a complete picture of execution quality
    across a batch of trades, decomposed into slippage (implicit market
    cost) and commissions (explicit broker cost).

    The DataFrame must contain columns ``'execution_price'``,
    ``'benchmark_price'``, ``'qty'``, and ``'side'``.

    Parameters:
        trades_df: DataFrame with one row per fill.  Required columns:
            ``'execution_price'``, ``'benchmark_price'``, ``'qty'``,
            ``'side'`` (``'buy'`` or ``'sell'``).
        commission_rate: Commission rate as a fraction of notional
            value (default 0.001 = 10 bps).

    Returns:
        Dictionary with:

        - ``'total_slippage'``: Sum of slippage * qty across all trades.
        - ``'total_commission'``: Sum of explicit commissions.
        - ``'total_cost'``: Slippage + commission.
        - ``'cost_bps'``: Total cost in basis points of total notional.
        - ``'n_trades'``: Number of fills.

    Example:
        >>> import pandas as pd
        >>> trades = pd.DataFrame({
        ...     'execution_price': [100.05, 50.10],
        ...     'benchmark_price': [100.00, 50.00],
        ...     'qty': [1000, 2000],
        ...     'side': ['buy', 'buy'],
        ... })
        >>> result = total_cost(trades, commission_rate=0.001)
        >>> result['n_trades']
        2
        >>> result['cost_bps'] > 0
        True

    See Also:
        slippage: Per-trade slippage calculation.
        commission_cost: Per-trade commission calculation.
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

    Use market impact models to pre-estimate how much your order will
    move the price, before you execute.  This is essential for
    pre-trade cost analysis and optimal execution sizing.

    Parameters:
        qty: Order quantity (shares).
        avg_daily_volume: Average daily volume.
        volatility: Daily price volatility (standard deviation of
            returns).
        model: Impact model to use.  ``'sqrt'`` for the square-root
            model (impact = sigma * sqrt(Q/ADV)), which is the
            industry standard.  ``'linear'`` for a simple linear
            model (impact = sigma * Q/ADV).

    Returns:
        Estimated market impact as a fraction of price (same scale as
        *volatility*).  Multiply by the stock price to get dollar impact.

    Example:
        >>> # 10,000 shares, ADV 1M, 2% daily vol
        >>> impact = market_impact_model(10_000, 1_000_000, 0.02, model='sqrt')
        >>> 0 < impact < 0.02  # less than full daily vol
        True

    Notes:
        The square-root model is empirically well-supported for
        equities: impact scales as the square root of participation
        rate, consistent with Kyle (1985) and Almgren et al. (2005).

    See Also:
        slippage: Measure actual post-trade slippage.
        wraquant.execution.optimal.almgren_chriss: Optimal execution trajectory.
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
