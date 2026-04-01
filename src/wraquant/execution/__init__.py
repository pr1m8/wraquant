"""Execution algorithms, optimal execution, and transaction cost analysis.

Provides a full suite of tools for splitting large orders into smaller
child orders to minimize market impact, plus analytical frameworks for
measuring and modeling transaction costs.  Bridges the gap between
portfolio construction (``wraquant.opt``) and realized performance by
quantifying the cost of turning target weights into actual positions.

Key sub-modules:

- **Algorithms** (``algorithms``) -- Standard execution schedules:
  ``twap_schedule`` (Time-Weighted Average Price -- uniform slicing),
  ``vwap_schedule`` (Volume-Weighted -- follows intraday volume profile),
  ``pov_schedule`` (Percentage of Volume), ``is_schedule``
  (Implementation Shortfall -- front-loads to minimize drift risk),
  ``adaptive_schedule`` (adjusts in real time), and
  ``arrival_price_benchmark``.
- **Optimal execution** (``optimal``) -- Analytical optimal trading
  trajectories: ``almgren_chriss`` (mean-variance optimal execution
  balancing urgency risk against market impact),
  ``bertsimas_lo`` (dynamic programming approach),
  ``optimal_execution_cost`` (expected cost of a given trajectory),
  and ``execution_frontier`` (cost-risk trade-off curve).
- **Transaction cost analysis** (``cost``) -- Post-trade TCA:
  ``slippage``, ``commission_cost``, ``market_impact_model`` (square-root
  or linear), ``liquidity_adjusted_cost``, ``total_cost`` aggregation,
  ``expected_cost_model`` (pre-trade cost estimation), and
  ``transaction_cost_analysis`` (full TCA report).

Example:
    >>> from wraquant.execution import vwap_schedule, almgren_chriss
    >>> schedule = vwap_schedule(total_shares=100_000, n_intervals=78)
    >>> trajectory = almgren_chriss(
    ...     total_shares=100_000, T=1.0, sigma=0.02, eta=0.01, gamma=0.001
    ... )

Use ``wraquant.execution`` when you need to model or plan how to execute
trades efficiently.  For measuring how microstructure affects your fills,
see ``wraquant.microstructure``.  For backtesting that incorporates
transaction costs, pass cost models to ``wraquant.backtest``.
"""

from __future__ import annotations

from wraquant.execution.algorithms import (
    adaptive_schedule,
    arrival_price_benchmark,
    close_auction_allocation,
    implementation_shortfall,
    is_schedule,
    participation_rate_schedule,
    pov_schedule,
    twap_schedule,
    vwap_schedule,
)
from wraquant.execution.cost import (
    commission_cost,
    expected_cost_model,
    liquidity_adjusted_cost,
    market_impact_model,
    slippage,
    total_cost,
    transaction_cost_analysis,
)
from wraquant.execution.optimal import (
    almgren_chriss,
    bertsimas_lo,
    execution_frontier,
    optimal_execution_cost,
)

__all__ = [
    # Algorithms
    "twap_schedule",
    "vwap_schedule",
    "implementation_shortfall",
    "participation_rate_schedule",
    "arrival_price_benchmark",
    "adaptive_schedule",
    "is_schedule",
    "pov_schedule",
    "close_auction_allocation",
    # Optimal execution
    "almgren_chriss",
    "bertsimas_lo",
    "optimal_execution_cost",
    "execution_frontier",
    # Cost
    "slippage",
    "commission_cost",
    "total_cost",
    "market_impact_model",
    "liquidity_adjusted_cost",
    "expected_cost_model",
    "transaction_cost_analysis",
]
