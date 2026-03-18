"""Backtesting framework for trading strategies.

Provides both vectorized and event-driven backtesting engines,
strategy abstractions, execution models, performance analysis,
event tracking, position sizing, and tearsheet generation.
"""

from wraquant.backtest.engine import Backtest
from wraquant.backtest.integrations import (
    empyrical_metrics,
    ffn_stats,
    pyfolio_tearsheet_data,
    quantstats_report,
    vectorbt_backtest,
)
from wraquant.backtest.events import (
    Event,
    EventTracker,
    EventType,
    detect_drawdown_events,
    detect_regime_changes,
)
from wraquant.backtest.metrics import performance_summary
from wraquant.backtest.position import (
    PositionSizer,
    clip_weights,
    invert_signal,
    rebalance_threshold,
)
from wraquant.backtest.strategy import Strategy
from wraquant.backtest.tearsheet import (
    drawdown_table,
    generate_tearsheet,
    monthly_returns_table,
    rolling_metrics_table,
    trade_analysis,
)

__all__ = [
    # engine
    "Backtest",
    # strategy
    "Strategy",
    # metrics
    "performance_summary",
    # events
    "Event",
    "EventTracker",
    "EventType",
    "detect_drawdown_events",
    "detect_regime_changes",
    # position
    "PositionSizer",
    "clip_weights",
    "invert_signal",
    "rebalance_threshold",
    # tearsheet
    "drawdown_table",
    "generate_tearsheet",
    "monthly_returns_table",
    "rolling_metrics_table",
    "trade_analysis",
    # integrations
    "vectorbt_backtest",
    "quantstats_report",
    "empyrical_metrics",
    "pyfolio_tearsheet_data",
    "ffn_stats",
]
