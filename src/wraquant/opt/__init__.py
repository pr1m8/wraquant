"""Portfolio and mathematical optimization.

Provides portfolio optimization (MVO, risk parity, Black-Litterman),
convex optimization wrappers, and multi-objective optimization.
"""

from wraquant.opt.base import Constraint, Objective, OptimizationResult
from wraquant.opt.portfolio import (
    black_litterman,
    equal_weight,
    hierarchical_risk_parity,
    inverse_volatility,
    max_sharpe,
    mean_variance,
    min_volatility,
    risk_parity,
)

__all__ = [
    "Constraint",
    "Objective",
    "OptimizationResult",
    "mean_variance",
    "min_volatility",
    "max_sharpe",
    "risk_parity",
    "equal_weight",
    "inverse_volatility",
    "hierarchical_risk_parity",
    "black_litterman",
]
