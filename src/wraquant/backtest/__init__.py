"""Backtesting framework for trading strategies.

Provides both vectorized and event-driven backtesting engines,
strategy abstractions, execution models, performance analysis,
event tracking, position sizing, and tearsheet generation.
"""

from wraquant.backtest.engine import Backtest, VectorizedBacktest, walk_forward_backtest
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
from wraquant.backtest.metrics import (
    burke_ratio,
    common_sense_ratio,
    expectancy,
    gain_to_pain_ratio,
    kappa_ratio,
    kelly_fraction,
    omega_ratio,
    payoff_ratio,
    performance_summary,
    profit_factor,
    rachev_ratio,
    recovery_factor,
    risk_of_ruin,
    system_quality_number,
    tail_ratio,
    ulcer_performance_index,
)
from wraquant.backtest.position import (
    PositionSizer,
    clip_weights,
    invert_signal,
    rebalance_threshold,
    regime_conditional_sizing,
    regime_signal_filter,
    risk_parity_position,
)
from wraquant.backtest.strategy import Strategy
from wraquant.backtest.tearsheet import (
    comprehensive_tearsheet,
    drawdown_table,
    generate_tearsheet,
    monthly_returns_table,
    rolling_metrics_table,
    strategy_comparison,
    trade_analysis,
)

__all__ = [
    # engine
    "Backtest",
    "VectorizedBacktest",
    "walk_forward_backtest",
    # strategy
    "Strategy",
    # metrics
    "performance_summary",
    "omega_ratio",
    "burke_ratio",
    "ulcer_performance_index",
    "kappa_ratio",
    "tail_ratio",
    "common_sense_ratio",
    "rachev_ratio",
    "gain_to_pain_ratio",
    "risk_of_ruin",
    "kelly_fraction",
    "expectancy",
    "profit_factor",
    "payoff_ratio",
    "recovery_factor",
    "system_quality_number",
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
    "risk_parity_position",
    "regime_conditional_sizing",
    "regime_signal_filter",
    # tearsheet
    "comprehensive_tearsheet",
    "drawdown_table",
    "generate_tearsheet",
    "monthly_returns_table",
    "rolling_metrics_table",
    "strategy_comparison",
    "trade_analysis",
    # integrations
    "vectorbt_backtest",
    "quantstats_report",
    "empyrical_metrics",
    "pyfolio_tearsheet_data",
    "ffn_stats",
]
