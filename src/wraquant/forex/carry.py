"""Carry trade analysis and interest rate differential calculations."""

from __future__ import annotations

import pandas as pd


def interest_rate_differential(
    base_rate: float,
    quote_rate: float,
) -> float:
    """Calculate interest rate differential between two currencies.

    The interest rate differential is the foundation of carry trades.
    A positive differential means the base currency has a higher yield,
    so a long position earns positive carry.

    Parameters:
        base_rate: Annual interest rate of base currency (e.g., 0.05 = 5%).
        quote_rate: Annual interest rate of quote currency.

    Returns:
        Interest rate differential (base - quote).  Positive means
        long carry, negative means short carry.

    Example:
        >>> interest_rate_differential(0.05, 0.01)  # AUD vs JPY
        0.04
        >>> interest_rate_differential(0.01, 0.05)  # JPY vs AUD (negative carry)
        -0.04

    See Also:
        carry_return: Full carry trade return including spot moves.
        carry_attractiveness: Rank pairs by carry.
    """
    return base_rate - quote_rate


def carry_return(
    spot_change: pd.Series,
    base_rate: float,
    quote_rate: float,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate total carry trade return (spot + carry).

    Use this to evaluate the full P&L of a carry trade, which earns
    the interest rate differential daily but is exposed to spot rate
    movements.  A positive carry (base rate > quote rate) means you
    earn interest by holding the position, but adverse spot moves can
    overwhelm the carry.

    Total return = spot return + (base_rate - quote_rate) / periods_per_year

    Parameters:
        spot_change: Daily spot rate returns (log or simple).
        base_rate: Annual interest rate of base currency (e.g., 0.04 = 4%).
        quote_rate: Annual interest rate of quote currency.
        periods_per_year: Periods per year (default 252 for daily data).

    Returns:
        Total return series (spot return + daily carry).  Positive
        values indicate profit for a long carry position.

    Example:
        >>> import pandas as pd
        >>> spot_returns = pd.Series([0.001, -0.002, 0.0005, 0.001])
        >>> total = carry_return(spot_returns, base_rate=0.05, quote_rate=0.01)
        >>> total.iloc[0] > spot_returns.iloc[0]  # carry adds return
        True

    See Also:
        interest_rate_differential: Raw rate differential.
        carry_attractiveness: Rank pairs by carry.
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

    Use this to compute the theoretical forward exchange rate based on
    covered interest rate parity.  If the forward rate exceeds the spot
    rate, the base currency trades at a forward discount (its interest
    rate is higher than the quote currency's).

    Formula: F = S * (1 + r_quote * days/365) / (1 + r_base * days/365)

    Parameters:
        spot: Current spot rate.
        base_rate: Base currency annual rate.
        quote_rate: Quote currency annual rate.
        days: Forward period in days (default 365 for 1-year forward).

    Returns:
        Forward rate.  Compare to spot to determine premium (F > S)
        or discount (F < S).

    Example:
        >>> forward_premium(1.1000, base_rate=0.04, quote_rate=0.02, days=365)
        1.0788461538461539

    See Also:
        carry_return: Full carry trade P&L.
    """
    return spot * (1 + quote_rate * days / 365) / (1 + base_rate * days / 365)


def carry_attractiveness(
    rates: dict[str, float],
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Rank currency pairs by carry attractiveness.

    Use this to screen the forex universe for the best carry trade
    opportunities.  Pairs with the highest interest rate differential
    offer the most carry income, but also tend to have higher crash
    risk (carry trade unwinds).

    Parameters:
        rates: Dict mapping currency code to annual interest rate
            (e.g., ``{'USD': 0.05, 'JPY': 0.001, 'AUD': 0.04}``).
        pairs: Optional list of (base, quote) pairs to evaluate.
            If None, evaluates all combinations.

    Returns:
        DataFrame with columns ``pair``, ``base_rate``, ``quote_rate``,
        ``differential``, sorted by differential descending.  Top
        rows are the most attractive carry trades.

    Example:
        >>> rates = {'USD': 0.05, 'JPY': 0.001, 'EUR': 0.04}
        >>> df = carry_attractiveness(rates)
        >>> df.iloc[0]['pair']  # highest carry
        'USDJPY'

    See Also:
        carry_return: Full carry trade P&L including spot moves.
        forward_premium: Covered interest rate parity forward rate.
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
