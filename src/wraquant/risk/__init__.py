"""Risk management and portfolio risk analytics.

Covers performance metrics, Value-at-Risk, portfolio risk decomposition,
and scenario analysis.
"""

from wraquant.risk.metrics import (
    hit_ratio,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from wraquant.risk.portfolio import (
    diversification_ratio,
    portfolio_volatility,
    risk_contribution,
)
from wraquant.risk.scenarios import monte_carlo_var, stress_test
from wraquant.risk.var import conditional_var, value_at_risk

__all__ = [
    # metrics
    "sharpe_ratio",
    "sortino_ratio",
    "information_ratio",
    "max_drawdown",
    "hit_ratio",
    # var
    "value_at_risk",
    "conditional_var",
    # portfolio
    "portfolio_volatility",
    "risk_contribution",
    "diversification_ratio",
    # scenarios
    "monte_carlo_var",
    "stress_test",
]
