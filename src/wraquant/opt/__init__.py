"""Portfolio and mathematical optimization.

Provides portfolio optimization (MVO, risk parity, Black-Litterman),
convex optimization wrappers, and multi-objective optimization.
"""

from wraquant.opt.base import Constraint, Objective, OptimizationResult
from wraquant.opt.convex import (
    minimize_quadratic,
    solve_qp,
    solve_sdp,
    solve_socp,
)
from wraquant.opt.linear import solve_lp, solve_milp, transportation_problem
from wraquant.opt.multi_objective import epsilon_constraint, nsga2, pareto_front
from wraquant.opt.nonlinear import global_minimize, minimize, root_find
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
from wraquant.opt.utils import (
    cardinality_constraint,
    sector_constraints,
    sum_to_one_constraint,
    turnover_constraint,
    weight_constraint,
)

__all__ = [
    # base
    "Constraint",
    "Objective",
    "OptimizationResult",
    # portfolio
    "mean_variance",
    "min_volatility",
    "max_sharpe",
    "risk_parity",
    "equal_weight",
    "inverse_volatility",
    "hierarchical_risk_parity",
    "black_litterman",
    # convex
    "minimize_quadratic",
    "solve_qp",
    "solve_socp",
    "solve_sdp",
    # linear
    "solve_lp",
    "solve_milp",
    "transportation_problem",
    # nonlinear
    "minimize",
    "global_minimize",
    "root_find",
    # multi-objective
    "pareto_front",
    "nsga2",
    "epsilon_constraint",
    # utils
    "weight_constraint",
    "sum_to_one_constraint",
    "sector_constraints",
    "turnover_constraint",
    "cardinality_constraint",
]
