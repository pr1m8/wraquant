"""Forex-specific analysis and tools.

Covers currency pair handling, pip calculations, lot sizing,
trading sessions, carry trade analysis, and forex risk management.
"""

from wraquant.forex.analysis import lot_size, pip_value, pips
from wraquant.forex.carry import carry_return, interest_rate_differential
from wraquant.forex.pairs import CurrencyPair, cross_rate, major_pairs
from wraquant.forex.session import ForexSession, current_session, session_overlaps

__all__ = [
    "CurrencyPair",
    "cross_rate",
    "major_pairs",
    "pips",
    "pip_value",
    "lot_size",
    "ForexSession",
    "current_session",
    "session_overlaps",
    "carry_return",
    "interest_rate_differential",
]
