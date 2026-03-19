"""Linear programming solvers.

All functions use scipy's built-in LP/MILP solvers and require no optional
dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import optimize


def solve_lp(
    c: npt.NDArray[np.floating],
    A_ub: npt.NDArray[np.floating] | None = None,
    b_ub: npt.NDArray[np.floating] | None = None,
    A_eq: npt.NDArray[np.floating] | None = None,
    b_eq: npt.NDArray[np.floating] | None = None,
    bounds: list[tuple[float | None, float | None]] | None = None,
    method: str = "highs",
) -> dict[str, Any]:
    """Solve a linear program via :func:`scipy.optimize.linprog`.

    Use LP for problems with a linear objective and linear constraints,
    such as transaction cost minimisation, portfolio rebalancing with
    turnover constraints, or resource allocation.

    Solves::

        min  c' x
        s.t. A_ub x <= b_ub
             A_eq x == b_eq
             lb <= x <= ub

    Parameters:
        c: Objective coefficient vector ``(n,)``.
        A_ub: Inequality constraint matrix ``(m_ub, n)``.
        b_ub: Inequality constraint RHS ``(m_ub,)``.
        A_eq: Equality constraint matrix ``(m_eq, n)``.
        b_eq: Equality constraint RHS ``(m_eq,)``.
        bounds: Variable bounds ``[(lb, ub), ...]``.  Use ``None`` for
            unbounded sides.
        method: LP solver method (default ``'highs'``).

    Returns:
        Dict with keys ``x`` (optimal solution vector), ``objective``
        (optimal value of c'x), ``status`` (``'optimal'`` or error),
        and ``success`` (bool).

    Example:
        >>> import numpy as np
        >>> # Minimise -x1 - 2*x2 s.t. x1 + x2 <= 4, x1, x2 >= 0
        >>> c = np.array([-1, -2], dtype=float)
        >>> A_ub = np.array([[1, 1]], dtype=float)
        >>> b_ub = np.array([4.0])
        >>> result = solve_lp(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)])
        >>> result['success']
        True
        >>> np.isclose(result['objective'], -8.0)
        True

    See Also:
        solve_milp: Mixed-integer LP (handles integer constraints).
        wraquant.opt.convex.solve_qp: Quadratic programming.
    """
    c = np.asarray(c, dtype=float)

    result = optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
    )

    return {
        "x": result.x,
        "objective": float(result.fun) if result.success else float("inf"),
        "status": "optimal" if result.success else result.message,
        "success": bool(result.success),
    }


def solve_milp(
    c: npt.NDArray[np.floating],
    A_ub: npt.NDArray[np.floating] | None = None,
    b_ub: npt.NDArray[np.floating] | None = None,
    A_eq: npt.NDArray[np.floating] | None = None,
    b_eq: npt.NDArray[np.floating] | None = None,
    bounds: list[tuple[float | None, float | None]] | None = None,
    integrality: npt.NDArray[np.integer] | list[int] | None = None,
) -> dict[str, Any]:
    """Solve a mixed-integer linear program via :func:`scipy.optimize.milp`.

    Use MILP when some decision variables must be integers, such as
    selecting a discrete number of assets to hold, binary buy/sell
    decisions, or lot-size-constrained portfolio construction.

    Parameters:
        c: Objective coefficient vector ``(n,)``.
        A_ub: Inequality constraint matrix ``(m_ub, n)``.
        b_ub: Inequality constraint RHS ``(m_ub,)``.
        A_eq: Equality constraint matrix ``(m_eq, n)``.
        b_eq: Equality constraint RHS ``(m_eq,)``.
        bounds: Variable bounds ``[(lb, ub), ...]``.
        integrality: Per-variable integrality indicator.
            ``0`` = continuous, ``1`` = integer.  If ``None`` all variables
            are continuous (equivalent to an LP).

    Returns:
        Dict with keys ``x`` (solution -- integer variables will be
        rounded), ``objective``, ``status``, and ``success``.

    Example:
        >>> import numpy as np
        >>> # Select 2 assets from 4 (binary selection)
        >>> c = np.array([-3, -5, -2, -4], dtype=float)
        >>> A_eq = np.ones((1, 4))
        >>> b_eq = np.array([2.0])
        >>> result = solve_milp(c, A_eq=A_eq, b_eq=b_eq,
        ...                    bounds=[(0, 1)] * 4,
        ...                    integrality=[1, 1, 1, 1])
        >>> result['success']
        True
        >>> int(sum(result['x']))  # exactly 2 assets selected
        2

    See Also:
        solve_lp: Continuous LP (faster, no integer constraints).
    """
    c = np.asarray(c, dtype=float)

    constraints: list[optimize.LinearConstraint] = []

    if A_ub is not None and b_ub is not None:
        A_ub = np.asarray(A_ub, dtype=float)
        b_ub = np.asarray(b_ub, dtype=float)
        constraints.append(optimize.LinearConstraint(A_ub, ub=b_ub))

    if A_eq is not None and b_eq is not None:
        A_eq = np.asarray(A_eq, dtype=float)
        b_eq = np.asarray(b_eq, dtype=float)
        constraints.append(optimize.LinearConstraint(A_eq, lb=b_eq, ub=b_eq))

    scipy_bounds = None
    if bounds is not None:
        lb = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
        ub = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
        scipy_bounds = optimize.Bounds(lb, ub)

    integrality_arr = (
        np.asarray(integrality, dtype=int) if integrality is not None else None
    )

    result = optimize.milp(
        c,
        constraints=constraints if constraints else None,
        integrality=integrality_arr,
        bounds=scipy_bounds,
    )

    return {
        "x": result.x,
        "objective": float(result.fun) if result.success else float("inf"),
        "status": "optimal" if result.success else result.message,
        "success": bool(result.success),
    }


def transportation_problem(
    costs: npt.NDArray[np.floating],
    supply: npt.NDArray[np.floating],
    demand: npt.NDArray[np.floating],
) -> dict[str, Any]:
    """Solve the transportation / assignment problem as an LP.

    Given *m* supply nodes and *n* demand nodes, find the shipment matrix
    ``X`` of shape ``(m, n)`` that minimises total cost.

    Parameters:
        costs: Cost matrix ``(m, n)`` where ``costs[i, j]`` is the
            per-unit shipping cost from supply *i* to demand *j*.
        supply: Supply capacities ``(m,)``.
        demand: Demand requirements ``(n,)``.

    Returns:
        Dict with keys ``x`` (shipment matrix), ``objective`` (total cost),
        ``status``, and ``success``.

    Raises:
        ValueError: If total supply does not equal total demand.
    """
    costs = np.asarray(costs, dtype=float)
    supply = np.asarray(supply, dtype=float)
    demand = np.asarray(demand, dtype=float)

    m, n = costs.shape

    if not np.isclose(supply.sum(), demand.sum()):
        raise ValueError(
            f"Total supply ({supply.sum()}) must equal total demand "
            f"({demand.sum()}) for a balanced transportation problem."
        )

    # Decision variables: x_{ij} flattened row-major
    c_flat = costs.flatten()
    num_vars = m * n

    # Supply constraints: sum over j of x_{ij} == supply[i]
    A_eq_rows: list[npt.NDArray] = []
    b_eq_parts: list[float] = []

    for i in range(m):
        row = np.zeros(num_vars)
        row[i * n : (i + 1) * n] = 1.0
        A_eq_rows.append(row)
        b_eq_parts.append(float(supply[i]))

    # Demand constraints: sum over i of x_{ij} == demand[j]
    for j in range(n):
        row = np.zeros(num_vars)
        for i in range(m):
            row[i * n + j] = 1.0
        A_eq_rows.append(row)
        b_eq_parts.append(float(demand[j]))

    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq_parts)
    bounds_list: list[tuple[float | None, float | None]] = [(0.0, None)] * num_vars

    result = solve_lp(c_flat, A_eq=A_eq, b_eq=b_eq, bounds=bounds_list)

    if result["x"] is not None:
        result["x"] = result["x"].reshape(m, n)

    return result
