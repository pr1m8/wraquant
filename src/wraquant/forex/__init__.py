"""Forex-specific analysis and tools.

Covers currency pair handling, pip calculations, lot sizing,
trading sessions, carry trade analysis, and forex risk management.
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
