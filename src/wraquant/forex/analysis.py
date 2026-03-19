"""Forex-specific calculations: pips, lot sizing, position sizing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from wraquant.forex.pairs import CurrencyPair


def pips(
    price_change: float | pd.Series,
    pair: CurrencyPair | None = None,
    is_jpy: bool = False,
) -> float | pd.Series:
    """Convert price change to pips.

    Use this to express price movements in the standard forex unit
    (pips) for consistent comparison across pairs.  One pip is 0.0001
    for most pairs and 0.01 for JPY pairs.

    Parameters:
        price_change: Price difference (e.g., 1.1050 - 1.1000 = 0.0050).
        pair: CurrencyPair (auto-detects JPY pairs).
        is_jpy: Whether pair involves JPY (if pair not provided).

    Returns:
        Number of pips (can be negative for downward moves).

    Example:
        >>> pips(0.0050)  # 50 pips for non-JPY pair
        50.0
        >>> pips(0.50, is_jpy=True)  # 50 pips for JPY pair
        50.0

    See Also:
        pip_value: Dollar value of one pip.
    """
    jpy = is_jpy or (pair is not None and pair.is_jpy_pair)
    pip_size = 0.01 if jpy else 0.0001
    return price_change / pip_size


def pip_value(
    pair: CurrencyPair | None = None,
    lot_size_units: float = 100_000,
    is_jpy: bool = False,
    exchange_rate: float = 1.0,
) -> float:
    """Calculate the value of one pip in account currency.

    Use pip value to determine the dollar (or account currency) impact
    of a one-pip move for a given position size.  This is fundamental
    to position sizing and risk management in forex.

    Formula: pip_value = (pip_size * lot_size_units) / exchange_rate

    Parameters:
        pair: CurrencyPair (auto-detects JPY pip size).
        lot_size_units: Position size in units (standard lot = 100,000,
            mini = 10,000, micro = 1,000).
        is_jpy: Whether pair involves JPY (if pair not provided).
        exchange_rate: Rate to convert to account currency.  Set to
            1.0 if account currency matches the quote currency.

    Returns:
        Value of one pip in account currency.

    Example:
        >>> pip_value(lot_size_units=100_000)  # Standard lot, non-JPY
        10.0
        >>> pip_value(lot_size_units=10_000)  # Mini lot
        1.0
        >>> pip_value(lot_size_units=100_000, is_jpy=True)  # JPY pair
        1000.0

    See Also:
        lot_size: Calculate position size from risk parameters.
        pips: Convert price change to pip count.
    """
    jpy = is_jpy or (pair is not None and pair.is_jpy_pair)
    ps = 0.01 if jpy else 0.0001
    return (ps * lot_size_units) / exchange_rate


def lot_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    pair: CurrencyPair | None = None,
    is_jpy: bool = False,
    exchange_rate: float = 1.0,
) -> float:
    """Calculate position size in lots based on risk management.

    Use lot size calculation to determine how large a position to take
    given your account size, risk tolerance, and stop-loss distance.
    This ensures that if the stop loss is hit, the loss is exactly
    the specified percentage of your account.

    Formula: lots = (account * risk%) / (stop_pips * pip_value_per_lot)

    Parameters:
        account_balance: Account balance in account currency.
        risk_percent: Risk per trade as percentage (e.g., 1.0 = 1%).
            Professional traders typically risk 0.5-2% per trade.
        stop_loss_pips: Stop loss distance in pips.  Wider stops
            require smaller positions to maintain the same risk.
        pair: CurrencyPair.
        is_jpy: Whether pair involves JPY.
        exchange_rate: Rate to convert to account currency.

    Returns:
        Position size in standard lots (1 lot = 100,000 units).

    Example:
        >>> lot_size(10_000, risk_percent=1.0, stop_loss_pips=50)
        0.2
        >>> lot_size(50_000, risk_percent=2.0, stop_loss_pips=100)
        1.0

    See Also:
        pip_value: Value of one pip for a given position size.
        pips: Convert price change to pip count.
    """
    risk_amount = account_balance * (risk_percent / 100)
    pv = pip_value(
        pair=pair, lot_size_units=100_000, is_jpy=is_jpy, exchange_rate=exchange_rate
    )
    risk_per_lot = stop_loss_pips * pv
    if risk_per_lot == 0:
        return 0.0
    return risk_amount / risk_per_lot


def spread_cost(
    spread_pips: float,
    lot_size_units: float = 100_000,
    pair: CurrencyPair | None = None,
    is_jpy: bool = False,
) -> float:
    """Calculate the cost of the spread for a position.

    The spread cost is an implicit transaction cost paid every time you
    enter or exit a position.  Use this to assess whether the spread
    makes a strategy unviable at a given position size.

    Parameters:
        spread_pips: Bid-ask spread in pips (e.g., 1.5 pips for EUR/USD).
        lot_size_units: Position size in units (default 100,000 = 1 lot).
        pair: CurrencyPair.
        is_jpy: Whether pair involves JPY.

    Returns:
        Spread cost in quote currency.

    Example:
        >>> spread_cost(1.5, lot_size_units=100_000)  # 1.5 pip spread, 1 lot
        15.0

    See Also:
        pip_value: Value of one pip.
    """
    pv = pip_value(pair=pair, lot_size_units=lot_size_units, is_jpy=is_jpy)
    return spread_pips * pv


def pip_distance(
    entry: float,
    exit: float,
    pair: CurrencyPair | str | None = None,
    is_jpy: bool = False,
) -> float:
    """Calculate the pip distance between two prices.

    Use this to measure the signed distance in pips between an entry and
    exit price.  Automatically detects JPY pairs (2-decimal pip size)
    versus standard pairs (4-decimal pip size).

    Formula: pip_distance = (exit - entry) / pip_size

    Parameters:
        entry: Entry (open) price.
        exit: Exit (close) price.
        pair: CurrencyPair instance or string like ``'USDJPY'`` for
            automatic JPY detection.  If *None*, uses *is_jpy* flag.
        is_jpy: Whether the pair involves JPY (only used when *pair*
            is not provided).

    Returns:
        Signed pip distance.  Positive means the price moved up from
        entry to exit (profit for a long position).

    Example:
        >>> pip_distance(1.1000, 1.1050)  # 50 pips up on EUR/USD
        50.0
        >>> pip_distance(110.00, 110.50, is_jpy=True)  # 50 pips on USD/JPY
        50.0
        >>> pip_distance(1.1050, 1.1000)  # 50 pips down
        -50.0

    See Also:
        pips: Convert a raw price change to pip count.
        risk_reward_ratio: Use pip distances for R:R analysis.
    """
    if isinstance(pair, str):
        pair = CurrencyPair.from_string(pair)
    jpy = is_jpy or (pair is not None and pair.is_jpy_pair)
    pip_size = 0.01 if jpy else 0.0001
    return (exit - entry) / pip_size


def position_value(
    lots: float,
    pip_val: float,
    pips_moved: float,
) -> float:
    """Calculate position P&L in account currency.

    Use this to compute the profit or loss of a forex position given
    the number of lots, the pip value per lot, and the number of pips
    the price has moved.

    Formula: P&L = lots * pip_value * pips

    Parameters:
        lots: Number of standard lots (1 lot = 100,000 units).
            Fractional lots are supported (e.g., 0.1 for a mini lot).
        pip_val: Value of one pip per lot in account currency.
            Use :func:`pip_value` to compute this.
        pips_moved: Number of pips the position has moved.  Positive
            for favourable moves (long profits / short losses),
            negative for adverse moves.

    Returns:
        Profit or loss in account currency.  Positive means profit.

    Example:
        >>> position_value(lots=1.0, pip_val=10.0, pips_moved=50)
        500.0
        >>> position_value(lots=0.5, pip_val=10.0, pips_moved=-30)
        -150.0

    See Also:
        pip_value: Calculate pip value per lot.
        lot_size: Calculate position size from risk parameters.
    """
    return lots * pip_val * pips_moved


def risk_reward_ratio(
    entry: float,
    stop: float,
    target: float,
    pair: CurrencyPair | str | None = None,
    is_jpy: bool = False,
) -> dict[str, float]:
    """Calculate the risk-reward ratio for a trade.

    Use this before entering a trade to evaluate whether the potential
    reward justifies the risk.  A ratio above 2.0 is generally considered
    favourable; below 1.0 means you risk more than you stand to gain.

    The function works for both long and short trades by comparing
    absolute distances from entry to stop and entry to target.

    Parameters:
        entry: Entry price.
        stop: Stop-loss price.
        target: Take-profit price.
        pair: CurrencyPair or string (e.g., ``'USDJPY'``) for automatic
            JPY detection.
        is_jpy: Whether the pair involves JPY.

    Returns:
        Dictionary containing:

        - **ratio** (*float*) -- Reward-to-risk ratio (target distance /
          stop distance).  Values above 2.0 are generally favourable.
        - **risk_pips** (*float*) -- Distance from entry to stop in pips
          (always positive).
        - **reward_pips** (*float*) -- Distance from entry to target in
          pips (always positive).

    Example:
        >>> result = risk_reward_ratio(1.1000, 1.0950, 1.1100)
        >>> result['ratio']
        2.0
        >>> result['risk_pips']
        50.0
        >>> result['reward_pips']
        100.0

    See Also:
        pip_distance: Raw pip distance between two prices.
        lot_size: Size position based on risk.
    """
    if isinstance(pair, str):
        pair = CurrencyPair.from_string(pair)
    jpy = is_jpy or (pair is not None and pair.is_jpy_pair)
    pip_size = 0.01 if jpy else 0.0001

    risk_pips = abs(entry - stop) / pip_size
    reward_pips = abs(target - entry) / pip_size
    ratio = reward_pips / risk_pips if risk_pips > 0 else float("inf")

    return {
        "ratio": ratio,
        "risk_pips": risk_pips,
        "reward_pips": reward_pips,
    }


def margin_call_price(
    entry: float,
    balance: float,
    margin_used: float,
    leverage: float,
    side: str = "long",
) -> float:
    """Calculate the price at which a margin call occurs.

    Use this to determine the maximum adverse move before a margin call
    is triggered.  A margin call occurs when the account equity falls to
    (or below) the margin required to maintain the position.

    For a long position, the margin call price is below entry; for a
    short position, it is above entry.

    Formula (long):
        margin_call_price = entry - (balance - margin_used) / (leverage * margin_used / entry)

    Simplified:
        margin_call_price = entry * (1 - (balance - margin_used) / (leverage * margin_used))

    Parameters:
        entry: Entry price of the position.
        balance: Account balance in account currency.
        margin_used: Margin (collateral) used for this position in
            account currency.
        leverage: Leverage ratio (e.g., 50.0 for 50:1 leverage).
        side: ``'long'`` or ``'short'``.

    Returns:
        Price at which a margin call is triggered.  For longs this is
        below entry; for shorts it is above entry.  If the margin call
        price is negative (for longs), returns 0.0 since prices cannot
        go negative.

    Example:
        >>> # $10,000 balance, $2,000 margin, 50:1 leverage, long at 1.1000
        >>> mc = margin_call_price(1.1000, 10_000, 2_000, 50.0)
        >>> mc < 1.1000  # margin call below entry for long
        True

    Notes:
        This is a simplified model.  Real margin calls depend on the
        broker's margin call level (e.g., 100% or 50% of margin),
        floating P&L on other positions, and swap costs.

    See Also:
        lot_size: Size positions to control risk.
    """
    # Position size in units = margin_used * leverage
    position_units = margin_used * leverage
    # Units per price unit
    units_per_price = position_units / entry if entry > 0 else 0.0

    if units_per_price == 0:
        return 0.0

    # Available equity above margin requirement
    available = balance - margin_used

    # Price movement that exhausts available equity
    price_move = available / units_per_price

    if side == "long":
        mc_price = entry - price_move
        return max(mc_price, 0.0)
    elif side == "short":
        mc_price = entry + price_move
        return mc_price
    else:
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
