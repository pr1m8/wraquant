"""Backtesting framework for trading strategies.

Provides a full-featured backtesting infrastructure for evaluating
trading strategies on historical data.  Supports both fast vectorized
backtesting (for signal-based strategies) and event-driven simulation
(for complex order logic), with 30+ performance metrics, position sizing
models, regime-aware filtering, and publication-quality tearsheets.

Key sub-modules:

- **Engine** (``engine``) -- ``Backtest`` (event-driven engine with
  fill simulation), ``VectorizedBacktest`` (fast signal-based engine),
  and ``walk_forward_backtest`` (rolling out-of-sample evaluation).
- **Strategy** (``strategy``) -- ``Strategy`` base class for defining
  entry/exit logic, signal generation, and position management.
- **Metrics** (``metrics``) -- 30+ performance metrics computed from
  an equity curve: ``performance_summary`` (one-call overview),
  ``omega_ratio``, ``burke_ratio``, ``ulcer_performance_index``,
  ``kappa_ratio``, ``tail_ratio``, ``rachev_ratio``,
  ``gain_to_pain_ratio``, ``kelly_fraction`` (optimal bet sizing),
  ``risk_of_ruin``, ``profit_factor``, ``system_quality_number``,
  ``expectancy``, ``recovery_factor``, and more.
- **Position sizing** (``position``) -- ``PositionSizer`` framework,
  ``risk_parity_position``, ``regime_conditional_sizing`` (size based
  on detected regime), ``regime_signal_filter`` (suppress signals in
  unfavorable regimes), ``clip_weights``, ``rebalance_threshold``.
- **Events** (``events``) -- ``EventTracker`` for logging trades,
  rebalances, and drawdown events during simulation.
  ``detect_drawdown_events`` and ``detect_regime_changes`` identify
  key structural events in the equity curve.
- **Tearsheet** (``tearsheet``) -- ``generate_tearsheet`` and
  ``comprehensive_tearsheet`` produce multi-panel performance reports.
  ``monthly_returns_table``, ``drawdown_table``,
  ``rolling_metrics_table``, ``strategy_comparison``, and
  ``trade_analysis`` for detailed diagnostics.
- **Integrations** -- Wrappers for vectorbt, quantstats, empyrical,
  pyfolio, and ffn.

Example:
    >>> from wraquant.backtest import VectorizedBacktest, performance_summary
    >>> bt = VectorizedBacktest(signal_fn=my_signal)
    >>> result = bt.run(prices)
    >>> perf = performance_summary(result["equity_curve"])
    >>> print(f"Sharpe: {perf['sharpe']:.2f}, Max DD: {perf['max_drawdown']:.1%}")

Use ``wraquant.backtest`` for strategy evaluation and walk-forward
analysis.  For risk measurement on the resulting equity curve, see
``wraquant.risk``.  For parallel parameter sweeps, see
``wraquant.scale.parallel_backtest``.  For interactive tearsheet
visualization, see ``wraquant.viz.plot_backtest_tearsheet``.
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
