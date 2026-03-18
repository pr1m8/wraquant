"""Execution algorithms, optimal execution, and transaction cost analysis.

Provides scheduling algorithms (TWAP, VWAP, POV), the Almgren-Chriss
optimal execution framework, and transaction cost analytics.
"""

from __future__ import annotations

from wraquant.execution.algorithms import (
    arrival_price_benchmark,
    implementation_shortfall,
    participation_rate_schedule,
    twap_schedule,
    vwap_schedule,
)
from wraquant.execution.cost import (
    commission_cost,
    market_impact_model,
    slippage,
    total_cost,
)
from wraquant.execution.optimal import (
    almgren_chriss,
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
    # Optimal execution
    "almgren_chriss",
    "optimal_execution_cost",
    "execution_frontier",
    # Cost
    "slippage",
    "commission_cost",
    "total_cost",
    "market_impact_model",
]
