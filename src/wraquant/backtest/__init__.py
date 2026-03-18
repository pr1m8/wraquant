"""Backtesting framework for trading strategies.

Provides both vectorized and event-driven backtesting engines,
strategy abstractions, execution models, and performance analysis.
"""

from wraquant.backtest.engine import Backtest
from wraquant.backtest.metrics import performance_summary
from wraquant.backtest.strategy import Strategy

__all__ = [
    "Backtest",
    "Strategy",
    "performance_summary",
]
