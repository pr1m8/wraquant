"""Forex-specific analysis and tools.

Provides a complete toolkit for foreign exchange analysis, covering
currency pair abstractions, pip and lot-size calculations, trading
session analytics, carry trade modeling, and forex-specific risk
management.  Designed for both discretionary FX traders and systematic
currency strategies.

Key sub-modules:

- **Pairs** (``pairs``) -- ``CurrencyPair`` dataclass for structured pair
  handling, ``cross_rate`` computation, ``major_pairs`` convenience list,
  ``correlation_matrix`` across pairs, ``currency_strength`` scoring,
  and ``volatility_by_session`` for session-level vol profiles.
- **Analysis** (``analysis``) -- Core FX calculations: ``pips`` (price to
  pip conversion), ``pip_value``, ``pip_distance``, ``lot_size`` for
  position sizing, ``spread_cost``, ``position_value``,
  ``risk_reward_ratio``, and ``margin_call_price``.
- **Sessions** (``session``) -- ``ForexSession`` enum (Tokyo, London,
  New York), ``current_session`` detection, and ``session_overlaps``
  for identifying high-liquidity windows.
- **Carry** (``carry``) -- Carry trade analytics: ``carry_return``,
  ``carry_attractiveness`` scoring, ``carry_portfolio`` construction,
  ``interest_rate_differential``, ``forward_premium``, and
  ``uncovered_interest_parity`` testing.
- **Risk** (``risk``) -- ``fx_portfolio_risk`` for multi-currency
  portfolio risk aggregation.

Example:
    >>> from wraquant.forex import CurrencyPair, pip_value, carry_return
    >>> pair = CurrencyPair("EUR", "USD")
    >>> pv = pip_value("EURUSD", lot_size=100_000)
    >>> cr = carry_return(spot=1.10, forward=1.0985, days=90)

Use ``wraquant.forex`` for FX-specific analytics.  For general portfolio
risk that includes currency exposure, combine with ``wraquant.risk``.
For macroeconomic data (interest rates, GDP) that feeds carry models,
use ``wraquant.data.fetch_macro``.
"""

from wraquant.forex.analysis import (
    lot_size,
    margin_call_price,
    pip_distance,
    pip_value,
    pips,
    position_value,
    risk_reward_ratio,
    spread_cost,
)
from wraquant.forex.carry import (
    carry_attractiveness,
    carry_portfolio,
    carry_return,
    forward_premium,
    interest_rate_differential,
    uncovered_interest_parity,
)
from wraquant.forex.pairs import (
    CurrencyPair,
    correlation_matrix,
    cross_rate,
    currency_strength,
    major_pairs,
    volatility_by_session,
)
from wraquant.forex.risk import fx_portfolio_risk
from wraquant.forex.session import ForexSession, current_session, session_overlaps

__all__ = [
    # Pairs
    "CurrencyPair",
    "cross_rate",
    "major_pairs",
    "correlation_matrix",
    "currency_strength",
    "volatility_by_session",
    # Analysis
    "pips",
    "pip_value",
    "pip_distance",
    "lot_size",
    "spread_cost",
    "position_value",
    "risk_reward_ratio",
    "margin_call_price",
    # Sessions
    "ForexSession",
    "current_session",
    "session_overlaps",
    # Carry
    "carry_return",
    "carry_attractiveness",
    "carry_portfolio",
    "interest_rate_differential",
    "forward_premium",
    "uncovered_interest_parity",
    # Risk
    "fx_portfolio_risk",
]
