"""Carry trade analysis and interest rate differential calculations."""

from __future__ import annotations

import pandas as pd


def interest_rate_differential(
    base_rate: float,
    quote_rate: float,
) -> float:
    """Calculate interest rate differential between two currencies.

    Parameters:
        base_rate: Annual interest rate of base currency (e.g., 0.05 = 5%).
        quote_rate: Annual interest rate of quote currency.

    Returns:
        Interest rate differential (base - quote).

    Example:
        >>> interest_rate_differential(0.05, 0.01)  # AUD vs JPY
        0.04
    """
    return base_rate - quote_rate


def carry_return(
    spot_change: pd.Series,
    base_rate: float,
    quote_rate: float,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate total carry trade return (spot + carry).

    Parameters:
        spot_change: Daily spot rate returns.
        base_rate: Annual interest rate of base currency.
        quote_rate: Annual interest rate of quote currency.
        periods_per_year: Periods per year.

    Returns:
        Total return series (spot return + daily carry).

    Example:
        >>> carry = carry_return(eurusd_returns, 0.04, 0.05)  # doctest: +SKIP
    """
    daily_carry = (base_rate - quote_rate) / periods_per_year
    return spot_change + daily_carry


def forward_premium(
    spot: float,
    base_rate: float,
    quote_rate: float,
    days: int = 365,
) -> float:
    """Calculate the forward premium/discount.

    Parameters:
        spot: Current spot rate.
        base_rate: Base currency annual rate.
        quote_rate: Quote currency annual rate.
        days: Forward period in days.

    Returns:
        Forward rate.
    """
    return spot * (1 + quote_rate * days / 365) / (1 + base_rate * days / 365)


def carry_attractiveness(
    rates: dict[str, float],
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Rank currency pairs by carry attractiveness.

    Parameters:
        rates: Dict mapping currency code to annual interest rate.
        pairs: Optional list of (base, quote) pairs to evaluate.
            If None, evaluates all combinations.

    Returns:
        DataFrame with pair, differential, and rank.
    """
    currencies = sorted(rates.keys())
    if pairs is None:
        pairs = [(b, q) for b in currencies for q in currencies if b != q]

    rows = []
    for base, quote in pairs:
        if base in rates and quote in rates:
            diff = rates[base] - rates[quote]
            rows.append(
                {
                    "pair": f"{base}{quote}",
                    "base_rate": rates[base],
                    "quote_rate": rates[quote],
                    "differential": diff,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("differential", ascending=False).reset_index(drop=True)
    return df
