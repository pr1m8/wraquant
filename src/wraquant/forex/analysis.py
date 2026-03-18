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

    Parameters:
        price_change: Price difference.
        pair: CurrencyPair (auto-detects JPY pairs).
        is_jpy: Whether pair involves JPY (if pair not provided).

    Returns:
        Number of pips.

    Example:
        >>> pips(0.0050)  # 50 pips for non-JPY pair
        50.0
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

    Parameters:
        pair: CurrencyPair.
        lot_size_units: Position size in units (standard lot = 100,000).
        is_jpy: Whether pair involves JPY.
        exchange_rate: Rate to convert to account currency.

    Returns:
        Value of one pip in account currency.

    Example:
        >>> pip_value(lot_size_units=100_000)  # Standard lot, non-JPY
        10.0
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

    Parameters:
        account_balance: Account balance in account currency.
        risk_percent: Risk per trade as percentage (e.g., 1.0 = 1%).
        stop_loss_pips: Stop loss distance in pips.
        pair: CurrencyPair.
        is_jpy: Whether pair involves JPY.
        exchange_rate: Rate to convert to account currency.

    Returns:
        Position size in standard lots.

    Example:
        >>> lot_size(10_000, risk_percent=1.0, stop_loss_pips=50)
        0.2
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

    Parameters:
        spread_pips: Bid-ask spread in pips.
        lot_size_units: Position size in units.
        pair: CurrencyPair.
        is_jpy: Whether pair involves JPY.

    Returns:
        Spread cost in quote currency.
    """
    pv = pip_value(pair=pair, lot_size_units=lot_size_units, is_jpy=is_jpy)
    return spread_pips * pv
