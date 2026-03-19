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
