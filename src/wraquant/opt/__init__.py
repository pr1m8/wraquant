"""Portfolio and mathematical optimization.

Provides a complete optimization toolkit for quantitative portfolio
construction, from classical mean-variance optimization through modern
hierarchical and Bayesian methods, plus general-purpose convex, linear,
nonlinear, and multi-objective solvers for custom formulations.

Key sub-modules:

- **Portfolio optimization** (``portfolio``) -- The core allocation
  methods:
  ``mean_variance`` (Markowitz MVO with optional constraints),
  ``min_volatility`` (global minimum variance portfolio),
  ``max_sharpe`` (tangency portfolio),
  ``risk_parity`` (equal risk contribution -- each asset contributes
  equally to portfolio volatility),
  ``equal_weight`` (1/N benchmark),
  ``inverse_volatility`` (weight inversely proportional to vol),
  ``hierarchical_risk_parity`` (Lopez de Prado's HRP -- uses
  hierarchical clustering to avoid inverting the covariance matrix),
  ``black_litterman`` (blend market equilibrium with investor views).
- **Convex optimization** (``convex``) -- ``minimize_quadratic``,
  ``solve_qp`` (quadratic program), ``solve_socp`` (second-order cone),
  ``solve_sdp`` (semidefinite program) via CVXPY.
- **Linear optimization** (``linear``) -- ``solve_lp``, ``solve_milp``
  (mixed-integer LP), and ``transportation_problem``.
- **Nonlinear optimization** (``nonlinear``) -- ``minimize`` (local),
  ``global_minimize`` (basin-hopping, differential evolution), and
  ``root_find``.
- **Multi-objective** (``multi_objective``) -- ``pareto_front``,
  ``nsga2`` (evolutionary multi-objective), and ``epsilon_constraint``.
- **Constraint utilities** (``utils``) -- ``weight_constraint``,
  ``sum_to_one_constraint``, ``sector_constraints``,
  ``turnover_constraint``, and ``cardinality_constraint`` for building
  realistic constraint sets.
- **Result types** (``base``) -- ``OptimizationResult``, ``Objective``,
  and ``Constraint`` dataclasses for structured output.

Example:
    >>> from wraquant.opt import max_sharpe, risk_parity
    >>> result = max_sharpe(returns, risk_free_rate=0.04)
    >>> print(result.weights, result.sharpe_ratio)
    >>> rp = risk_parity(returns)

Use ``wraquant.opt`` for portfolio allocation decisions.  For risk
measurement and decomposition of the resulting portfolio, see
``wraquant.risk``.  For parallel optimization sweeps across constraint
sets, see ``wraquant.scale.parallel_optimize``.
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
