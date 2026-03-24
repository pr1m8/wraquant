"""Carry trade analysis and interest rate differential calculations."""

from __future__ import annotations

import pandas as pd

from wraquant.core._coerce import coerce_series


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
    spot_change = coerce_series(spot_change, "spot_change")
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


def carry_portfolio(
    rates_dict: dict[str, float],
    weights: dict[str, float] | None = None,
    n_long: int = 3,
    n_short: int = 3,
) -> dict[str, object]:
    """Construct a carry trade portfolio: long high-yield, short low-yield.

    Use this to build a systematic carry strategy.  The portfolio goes
    long the *n_long* highest-yielding currencies and short the *n_short*
    lowest-yielding currencies.  If custom weights are provided, they
    override the automatic equal-weight allocation.

    This is the standard G10 carry trade approach used by institutional
    investors.  It earns the interest rate differential but is exposed
    to crash risk (carry unwinds).

    Parameters:
        rates_dict: Dictionary mapping currency codes to their annual
            interest rates (e.g., ``{'USD': 0.05, 'JPY': 0.001,
            'AUD': 0.04, 'EUR': 0.03, 'CHF': 0.015, 'NZD': 0.045}``).
        weights: Optional custom weight for each currency.  If *None*,
            equal-weights the long and short legs separately.
        n_long: Number of currencies in the long leg (default 3).
        n_short: Number of currencies in the short leg (default 3).

    Returns:
        Dictionary containing:

        - **weights** (*dict*) -- Portfolio weights per currency.
          Positive for long positions, negative for short positions.
          Weights sum to approximately zero (dollar-neutral).
        - **expected_carry** (*float*) -- Expected annualised carry
          return (weighted sum of rates for longs minus shorts).
        - **long_currencies** (*list*) -- Currencies in the long leg.
        - **short_currencies** (*list*) -- Currencies in the short leg.

    Example:
        >>> rates = {'USD': 0.05, 'JPY': 0.001, 'AUD': 0.04,
        ...          'EUR': 0.03, 'CHF': 0.015, 'NZD': 0.045}
        >>> result = carry_portfolio(rates, n_long=2, n_short=2)
        >>> result['expected_carry'] > 0
        True
        >>> len(result['long_currencies'])
        2

    See Also:
        carry_attractiveness: Rank all pairs by carry differential.
        carry_return: Full P&L including spot moves.
    """
    import numpy as np

    sorted_currencies = sorted(rates_dict.keys(), key=lambda c: rates_dict[c], reverse=True)

    long_ccys = sorted_currencies[:n_long]
    short_ccys = sorted_currencies[-n_short:]

    if weights is not None:
        port_weights = {c: weights.get(c, 0.0) for c in sorted_currencies}
    else:
        # Equal weight within each leg, dollar-neutral
        long_weight = 1.0 / n_long if n_long > 0 else 0.0
        short_weight = -1.0 / n_short if n_short > 0 else 0.0
        port_weights = {}
        for c in long_ccys:
            port_weights[c] = long_weight
        for c in short_ccys:
            port_weights[c] = short_weight

    # Expected carry = sum(weight_i * rate_i)
    expected_carry = sum(
        port_weights.get(c, 0.0) * rates_dict[c] for c in rates_dict
    )

    return {
        "weights": port_weights,
        "expected_carry": float(expected_carry),
        "long_currencies": long_ccys,
        "short_currencies": short_ccys,
    }


def uncovered_interest_parity(
    domestic_rate: float,
    foreign_rate: float,
    spot: float,
    maturity: float = 1.0,
) -> dict[str, float]:
    """Uncovered Interest Rate Parity (UIP) expected future spot rate.

    Use this to compute the expected future exchange rate implied by the
    interest rate differential under UIP.  UIP states that the expected
    depreciation of a currency equals the interest rate differential.

    While Covered Interest Parity (CIP) holds by arbitrage, UIP is an
    equilibrium condition that often fails empirically (the "forward
    premium puzzle"), which is why carry trades can be profitable.

    Formula:
        E[S_T] = S * (1 + r_domestic * T) / (1 + r_foreign * T)

    This is the same formula as Covered Interest Parity but interpreted
    as the *expected* future spot rate rather than the no-arbitrage
    forward rate.

    Parameters:
        domestic_rate: Annual interest rate of the domestic (base)
            currency.
        foreign_rate: Annual interest rate of the foreign (quote)
            currency.
        spot: Current spot exchange rate (domestic/foreign).
        maturity: Horizon in years (default 1.0).

    Returns:
        Dictionary containing:

        - **forward_rate** (*float*) -- UIP-implied expected future spot
          rate.  If domestic rate > foreign rate, the domestic currency
          is expected to depreciate (forward_rate > spot).
        - **forward_premium** (*float*) -- Forward premium as a
          percentage (``(forward - spot) / spot``).  Positive means the
          domestic currency trades at a forward premium (expected to
          depreciate).

    Example:
        >>> result = uncovered_interest_parity(0.05, 0.01, 1.1000, maturity=1.0)
        >>> result['forward_rate'] > 1.1000  # domestic rate higher -> depreciation expected
        True
        >>> abs(result['forward_premium'] - 0.0396) < 0.01
        True

    Notes:
        Reference: Fama (1984). "Forward and Spot Exchange Rates."
        *Journal of Monetary Economics*, 14, 319-338.

    See Also:
        forward_premium: CIP-based forward rate calculation.
        carry_return: Carry trade P&L (profits when UIP fails).
    """
    fwd = spot * (1 + domestic_rate * maturity) / (1 + foreign_rate * maturity)
    premium = (fwd - spot) / spot if spot != 0 else 0.0

    return {
        "forward_rate": float(fwd),
        "forward_premium": float(premium),
    }
