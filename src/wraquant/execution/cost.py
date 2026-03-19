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


def liquidity_adjusted_cost(
    price: float,
    quantity: float,
    bid: float | pd.Series,
    ask: float | pd.Series,
    volume: float | pd.Series,
    avg_daily_volume: float | None = None,
) -> dict[str, float]:
    """Estimate execution cost using microstructure liquidity measures.

    Combines spread cost from ``microstructure.liquidity`` with market
    impact from ``execution.cost`` for a complete cost estimate.  This
    bridges the microstructure and execution modules, giving a single
    function that incorporates both implicit (spread) and impact costs.

    The total cost has three components:

    1. **Spread cost**: half the effective spread times quantity. This
       is the immediate cost of crossing the bid-ask spread.
    2. **Market impact**: estimated price impact from the square-root
       model, scaled by quantity and price.
    3. **Total cost**: sum of spread cost and market impact cost.

    When to use:
        Use this for pre-trade cost estimation when you have both
        quote data (bid/ask) and volume data.  It gives a more
        realistic cost estimate than spread or impact alone because
        real execution costs include both components.

    Parameters:
        price: Current mid-price or last trade price.
        quantity: Order quantity (shares).
        bid: Best bid price(s).
        ask: Best ask price(s).
        volume: Recent trading volume.
        avg_daily_volume: Average daily volume for impact estimation.
            If *None*, uses the mean of *volume* (when *volume* is a
            Series) or *volume* itself (when scalar).

    Returns:
        Dictionary containing:

        - ``'spread_cost'`` (*float*) -- Cost from crossing the bid-ask
          spread (half-spread times quantity).
        - ``'market_impact_cost'`` (*float*) -- Estimated market impact
          cost in dollar terms.
        - ``'total_cost'`` (*float*) -- Sum of spread and impact costs.
        - ``'cost_bps'`` (*float*) -- Total cost in basis points of
          notional value.
        - ``'effective_spread'`` (*float*) -- Mean effective spread used
          in the calculation.
        - ``'amihud_illiquidity'`` (*float*) -- Amihud illiquidity ratio
          (higher = less liquid).

    Example:
        >>> result = liquidity_adjusted_cost(
        ...     price=100.0, quantity=5000,
        ...     bid=99.98, ask=100.02, volume=1_000_000,
        ... )
        >>> result['spread_cost'] > 0
        True
        >>> result['total_cost'] >= result['spread_cost']
        True

    See Also:
        wraquant.microstructure.liquidity.effective_spread: Raw spread.
        wraquant.microstructure.liquidity.amihud_illiquidity: Illiquidity.
        market_impact_model: Parametric impact estimation.
    """
    from wraquant.microstructure.liquidity import (
        amihud_illiquidity,
        effective_spread,
    )

    # Compute effective spread from bid/ask
    bid_arr = np.asarray(bid, dtype=np.float64)
    ask_arr = np.asarray(ask, dtype=np.float64)
    mid = (bid_arr + ask_arr) / 2.0

    # effective_spread returns 2 * |trade - mid|; for pre-trade estimate
    # use the quoted spread directly
    eff_spread_raw = effective_spread(
        trade_prices=np.atleast_1d(ask_arr),
        midpoints=np.atleast_1d(mid),
    )
    mean_eff_spread = float(np.nanmean(eff_spread_raw))

    # Spread cost = half-spread * quantity (you pay half the spread on entry)
    half_spread = mean_eff_spread / 2.0
    spread_cost = half_spread * abs(quantity)

    # Amihud illiquidity (scalar estimate)
    if isinstance(volume, pd.Series):
        vol_series = volume
        returns_proxy = pd.Series(np.zeros(len(volume)))
        returns_proxy.iloc[1:] = np.diff(np.log(np.maximum(mid, 1e-10))) if np.ndim(mid) > 0 else 0.0
        adv = float(vol_series.mean()) if avg_daily_volume is None else avg_daily_volume
        amihud = float(amihud_illiquidity(returns_proxy, vol_series))
    else:
        adv = float(volume) if avg_daily_volume is None else avg_daily_volume
        amihud = 0.0

    # Market impact via square-root model
    if adv > 0:
        # Estimate daily volatility from spread as a proxy if we don't have returns
        vol_proxy = mean_eff_spread / price if price > 0 else 0.01
        impact_frac = market_impact_model(
            qty=abs(quantity),
            avg_daily_volume=adv,
            volatility=vol_proxy,
            model="sqrt",
        )
        impact_cost = impact_frac * price * abs(quantity)
    else:
        impact_cost = 0.0

    total = spread_cost + impact_cost
    notional = abs(quantity) * price
    cost_bps = (total / notional) * 10_000 if notional > 0 else 0.0

    return {
        "spread_cost": float(spread_cost),
        "market_impact_cost": float(impact_cost),
        "total_cost": float(total),
        "cost_bps": float(cost_bps),
        "effective_spread": float(mean_eff_spread),
        "amihud_illiquidity": float(amihud),
    }


def expected_cost_model(
    quantity: float,
    price: float,
    adv: float,
    volatility: float,
    spread: float,
) -> dict[str, float]:
    """Comprehensive pre-trade expected cost model.

    Use this before executing an order to estimate total expected cost,
    broken down into three components: spread crossing cost, market
    impact, and timing risk.  This helps decide whether to execute
    aggressively (high urgency) or passively (low urgency).

    Components:
        1. **Spread cost**: immediate cost of crossing the bid-ask
           spread = 0.5 * spread * quantity.
        2. **Market impact**: price impact from trading = sigma *
           sqrt(Q / ADV) * price * quantity (square-root model).
        3. **Timing risk**: opportunity cost of slow execution =
           sigma * sqrt(T) * price * quantity, where T is estimated
           execution duration proportional to Q / ADV.

    Parameters:
        quantity: Order quantity (shares or units).
        price: Current market price.
        adv: Average daily volume (shares).
        volatility: Daily price volatility (standard deviation of
            returns, e.g., 0.02 = 2%).
        spread: Bid-ask spread in price terms (e.g., 0.02 for a
            $0.02 spread).

    Returns:
        Dictionary containing:

        - **spread_cost** (*float*) -- Half-spread cost in dollar terms.
        - **impact_cost** (*float*) -- Estimated market impact cost.
        - **timing_risk** (*float*) -- Estimated timing/opportunity risk.
        - **total_cost** (*float*) -- Sum of all components.
        - **cost_bps** (*float*) -- Total cost in basis points of
          notional value.

    Example:
        >>> result = expected_cost_model(
        ...     quantity=5000, price=100.0, adv=1_000_000,
        ...     volatility=0.02, spread=0.02,
        ... )
        >>> result['spread_cost']
        50.0
        >>> result['total_cost'] > result['spread_cost']
        True

    See Also:
        market_impact_model: Parametric impact estimation.
        liquidity_adjusted_cost: Spread + impact using microstructure data.
    """
    notional = abs(quantity) * price
    participation = abs(quantity) / adv if adv > 0 else 1.0

    # 1. Spread cost: half the spread per share
    s_cost = 0.5 * spread * abs(quantity)

    # 2. Market impact (square-root model)
    impact_frac = volatility * np.sqrt(participation)
    i_cost = impact_frac * notional

    # 3. Timing risk: proportional to execution duration
    # Execution duration ~ Q / (pov * ADV), assume pov = 0.1
    exec_duration = participation / 0.1  # in days
    t_risk = volatility * np.sqrt(max(exec_duration, 0.0)) * notional * 0.5

    total = s_cost + i_cost + t_risk
    cost_bps = (total / notional) * 10_000 if notional > 0 else 0.0

    return {
        "spread_cost": float(s_cost),
        "impact_cost": float(i_cost),
        "timing_risk": float(t_risk),
        "total_cost": float(total),
        "cost_bps": float(cost_bps),
    }


def transaction_cost_analysis(
    trades_df: pd.DataFrame,
    market_data_df: pd.DataFrame,
) -> pd.DataFrame:
    """Post-trade Transaction Cost Analysis (TCA).

    Use this after execution to evaluate the quality of each fill
    relative to multiple benchmarks.  TCA is the standard way
    institutional investors assess broker/algorithm performance.

    Compares each trade's execution price against:

    - **Arrival price**: mid-price when the order was submitted.
    - **VWAP**: volume-weighted average price during execution.
    - **Close**: closing price of the day.

    Parameters:
        trades_df: DataFrame with one row per fill.  Required columns:

            - ``'execution_price'`` -- actual fill price.
            - ``'qty'`` -- fill quantity (positive for buys).
            - ``'side'`` -- ``'buy'`` or ``'sell'``.
            - ``'timestamp'`` -- execution timestamp (optional, used
              for ordering).

        market_data_df: DataFrame with market reference data.  Required
            columns:

            - ``'arrival_price'`` -- mid-price at order arrival.
            - ``'vwap'`` -- VWAP during execution window.
            - ``'close'`` -- closing price.

            If *market_data_df* has a single row, the same benchmarks
            are applied to all trades.  If it has one row per trade,
            benchmarks are matched by index.

    Returns:
        DataFrame with original trade data plus additional columns:

        - ``'arrival_cost'`` -- execution price vs arrival price
          (positive = slippage cost for buys).
        - ``'arrival_cost_bps'`` -- arrival cost in basis points.
        - ``'vwap_cost'`` -- execution price vs VWAP.
        - ``'vwap_cost_bps'`` -- VWAP cost in basis points.
        - ``'close_cost'`` -- execution price vs close.
        - ``'close_cost_bps'`` -- close cost in basis points.

    Example:
        >>> import pandas as pd
        >>> trades = pd.DataFrame({
        ...     'execution_price': [100.05, 100.10, 100.08],
        ...     'qty': [1000, 2000, 1500],
        ...     'side': ['buy', 'buy', 'buy'],
        ... })
        >>> market = pd.DataFrame({
        ...     'arrival_price': [100.00],
        ...     'vwap': [100.06],
        ...     'close': [100.12],
        ... })
        >>> tca = transaction_cost_analysis(trades, market)
        >>> 'arrival_cost_bps' in tca.columns
        True

    See Also:
        total_cost: Aggregate cost breakdown.
        arrival_price_benchmark: Simpler arrival-price analysis.
    """
    result = trades_df.copy()

    # Expand market_data_df to match trades if single row
    if len(market_data_df) == 1:
        arrival = float(market_data_df["arrival_price"].iloc[0])
        vwap_price = float(market_data_df["vwap"].iloc[0])
        close_price = float(market_data_df["close"].iloc[0])
        arrivals = np.full(len(trades_df), arrival)
        vwaps = np.full(len(trades_df), vwap_price)
        closes = np.full(len(trades_df), close_price)
    else:
        arrivals = np.asarray(market_data_df["arrival_price"], dtype=np.float64)
        vwaps = np.asarray(market_data_df["vwap"], dtype=np.float64)
        closes = np.asarray(market_data_df["close"], dtype=np.float64)

    exec_prices = np.asarray(trades_df["execution_price"], dtype=np.float64)
    sides = trades_df["side"].values

    # Calculate signed costs (positive = cost for the trader)
    sign = np.where(sides == "buy", 1.0, -1.0)

    arrival_cost = sign * (exec_prices - arrivals)
    vwap_cost = sign * (exec_prices - vwaps)
    close_cost = sign * (exec_prices - closes)

    result["arrival_cost"] = arrival_cost
    result["arrival_cost_bps"] = np.where(
        arrivals != 0, (arrival_cost / arrivals) * 10_000, 0.0
    )
    result["vwap_cost"] = vwap_cost
    result["vwap_cost_bps"] = np.where(
        vwaps != 0, (vwap_cost / vwaps) * 10_000, 0.0
    )
    result["close_cost"] = close_cost
    result["close_cost_bps"] = np.where(
        closes != 0, (close_cost / closes) * 10_000, 0.0
    )

    return result
