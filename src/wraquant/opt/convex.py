"""Convex optimization wrappers.

Core functions use pure scipy; cvxpy-based solvers are gated behind the
``optimization`` optional-dependency group.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import optimize

from wraquant.core.decorators import requires_extra


def minimize_quadratic(
    Q: npt.NDArray[np.floating],
    c: npt.NDArray[np.floating],
    A_eq: npt.NDArray[np.floating] | None = None,
    b_eq: npt.NDArray[np.floating] | None = None,
    A_ub: npt.NDArray[np.floating] | None = None,
    b_ub: npt.NDArray[np.floating] | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Solve min 0.5 * x'Qx + c'x subject to linear constraints via SLSQP.

    Use this for portfolio optimisation (where Q is the covariance matrix),
    regularised regression, and any convex quadratic problem.  The SLSQP
    solver handles moderate problem sizes (n < 1000) well; for larger
    problems, use ``solve_qp`` with the ``'osqp'`` or ``'cvxpy'`` backend.

    Parameters:
        Q: Positive semi-definite matrix of shape ``(n, n)``.
        c: Linear cost vector of shape ``(n,)``.
        A_eq: Equality constraint matrix, ``A_eq @ x == b_eq``.
        b_eq: Equality constraint right-hand side.
        A_ub: Inequality constraint matrix, ``A_ub @ x <= b_ub``.
        b_ub: Inequality constraint right-hand side.
        bounds: Variable bounds as ``[(lb, ub), ...]``.

    Returns:
        Dict with keys ``x`` (solution vector), ``objective`` (optimal
        value), ``status`` (``'optimal'`` or error message), and
        ``success`` (bool).

    Example:
        >>> import numpy as np
        >>> Q = np.array([[2, 0], [0, 2]], dtype=float)
        >>> c = np.array([-4, -6], dtype=float)
        >>> result = minimize_quadratic(Q, c, bounds=[(0, 10), (0, 10)])
        >>> result['success']
        True
        >>> np.allclose(result['x'], [2, 3], atol=0.1)
        True

    See Also:
        solve_qp: Multi-backend QP solver (scipy, OSQP, cvxpy).
    """
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float)
    n = Q.shape[0]

    def objective(x: npt.NDArray) -> float:
        return float(0.5 * x @ Q @ x + c @ x)

    def gradient(x: npt.NDArray) -> npt.NDArray:
        return Q @ x + c

    constraints: list[dict[str, Any]] = []

    if A_eq is not None and b_eq is not None:
        A_eq = np.asarray(A_eq, dtype=float)
        b_eq = np.asarray(b_eq, dtype=float)
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x, _A=A_eq, _b=b_eq: _A @ x - _b,
                "jac": lambda x, _A=A_eq: _A,
            }
        )

    if A_ub is not None and b_ub is not None:
        A_ub = np.asarray(A_ub, dtype=float)
        b_ub = np.asarray(b_ub, dtype=float)
        # scipy expects ineq constraint as fun(x) >= 0, so: b_ub - A_ub @ x >= 0
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, _A=A_ub, _b=b_ub: _b - _A @ x,
                "jac": lambda x, _A=A_ub: -_A,
            }
        )

    x0 = np.zeros(n)

    result = optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
    )

    return {
        "x": result.x,
        "objective": float(result.fun),
        "status": "optimal" if result.success else result.message,
        "success": bool(result.success),
    }


def solve_qp(
    Q: npt.NDArray[np.floating],
    c: npt.NDArray[np.floating],
    A_eq: npt.NDArray[np.floating] | None = None,
    b_eq: npt.NDArray[np.floating] | None = None,
    A_ub: npt.NDArray[np.floating] | None = None,
    b_ub: npt.NDArray[np.floating] | None = None,
    bounds: list[tuple[float, float]] | None = None,
    solver: str = "scipy",
) -> dict[str, Any]:
    """Quadratic program solver with backend dispatch.

    Use ``solve_qp`` as the primary entry point for quadratic
    programming.  It dispatches to scipy (no extra deps), OSQP (fast
    first-order solver for large sparse QPs), or cvxpy (general-purpose
    convex solver).

    Solves::

        min  0.5 * x' Q x + c' x
        s.t. A_eq x  = b_eq
             A_ub x <= b_ub
             lb <= x <= ub

    Parameters:
        Q: Positive semi-definite matrix ``(n, n)``.
        c: Linear cost vector ``(n,)``.
        A_eq: Equality constraint matrix.
        b_eq: Equality constraint RHS.
        A_ub: Inequality constraint matrix.
        b_ub: Inequality constraint RHS.
        bounds: Variable bounds ``[(lb, ub), ...]``.
        solver: Backend -- ``'scipy'`` (default, no extra deps),
            ``'osqp'`` (fast for large sparse problems, requires
            ``optimization`` extra), or ``'cvxpy'`` (most flexible,
            requires ``optimization`` extra).

    Returns:
        Dict with keys ``x`` (solution), ``objective`` (optimal value),
        ``status`` (solver status), and ``success`` (bool).

    Raises:
        ValueError: If *solver* is not recognised.

    Example:
        >>> import numpy as np
        >>> Q = np.eye(3) * 2
        >>> c = np.array([-2, -3, -1], dtype=float)
        >>> A_eq = np.ones((1, 3))
        >>> b_eq = np.array([1.0])
        >>> result = solve_qp(Q, c, A_eq=A_eq, b_eq=b_eq,
        ...                   bounds=[(0, 1)] * 3)
        >>> result['success']
        True

    See Also:
        minimize_quadratic: Scipy-only QP solver.
        wraquant.opt.portfolio.mean_variance: Portfolio-specific QP.
    """
    solver = solver.lower()

    if solver == "scipy":
        return minimize_quadratic(Q, c, A_eq, b_eq, A_ub, b_ub, bounds)

    if solver == "osqp":
        return _solve_qp_osqp(Q, c, A_eq, b_eq, A_ub, b_ub, bounds)

    if solver == "cvxpy":
        return _solve_qp_cvxpy(Q, c, A_eq, b_eq, A_ub, b_ub, bounds)

    raise ValueError(
        f"Unknown solver '{solver}'. Choose from 'scipy', 'osqp', 'cvxpy'."
    )


@requires_extra("optimization")
def _solve_qp_osqp(
    Q: npt.NDArray[np.floating],
    c: npt.NDArray[np.floating],
    A_eq: npt.NDArray[np.floating] | None = None,
    b_eq: npt.NDArray[np.floating] | None = None,
    A_ub: npt.NDArray[np.floating] | None = None,
    b_ub: npt.NDArray[np.floating] | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Solve QP via OSQP (requires ``optimization`` extra)."""
    import osqp
    from scipy import sparse

    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float)
    n = Q.shape[0]

    # Build constraint matrices for OSQP: l <= Ax <= u
    A_rows: list[npt.NDArray] = []
    l_parts: list[npt.NDArray] = []
    u_parts: list[npt.NDArray] = []

    if A_eq is not None and b_eq is not None:
        A_eq = np.asarray(A_eq, dtype=float)
        b_eq = np.asarray(b_eq, dtype=float)
        A_rows.append(A_eq)
        l_parts.append(b_eq)
        u_parts.append(b_eq)

    if A_ub is not None and b_ub is not None:
        A_ub = np.asarray(A_ub, dtype=float)
        b_ub = np.asarray(b_ub, dtype=float)
        A_rows.append(A_ub)
        l_parts.append(np.full(len(b_ub), -np.inf))
        u_parts.append(b_ub)

    if bounds is not None:
        eye = np.eye(n)
        A_rows.append(eye)
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        l_parts.append(lb)
        u_parts.append(ub)

    if A_rows:
        A_combined = sparse.csc_matrix(np.vstack(A_rows))
        l_combined = np.concatenate(l_parts)
        u_combined = np.concatenate(u_parts)
    else:
        A_combined = sparse.csc_matrix(np.eye(n))
        l_combined = np.full(n, -np.inf)
        u_combined = np.full(n, np.inf)

    P = sparse.csc_matrix(Q)
    solver = osqp.OSQP()
    solver.setup(P, c, A_combined, l_combined, u_combined, verbose=False)
    res = solver.solve()

    success = res.info.status == "solved"
    return {
        "x": res.x,
        "objective": float(res.info.obj_val) if success else float("inf"),
        "status": res.info.status,
        "success": success,
    }


@requires_extra("optimization")
def _solve_qp_cvxpy(
    Q: npt.NDArray[np.floating],
    c: npt.NDArray[np.floating],
    A_eq: npt.NDArray[np.floating] | None = None,
    b_eq: npt.NDArray[np.floating] | None = None,
    A_ub: npt.NDArray[np.floating] | None = None,
    b_ub: npt.NDArray[np.floating] | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Solve QP via cvxpy (requires ``optimization`` extra)."""
    import cvxpy as cp

    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float)
    n = Q.shape[0]

    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(Q)) + c @ x)

    constraints: list[Any] = []
    if A_eq is not None and b_eq is not None:
        constraints.append(np.asarray(A_eq) @ x == np.asarray(b_eq))
    if A_ub is not None and b_ub is not None:
        constraints.append(np.asarray(A_ub) @ x <= np.asarray(b_ub))
    if bounds is not None:
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        constraints.append(x >= lb)
        constraints.append(x <= ub)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    success = prob.status == "optimal"
    return {
        "x": np.array(x.value).flatten() if success else np.full(n, np.nan),
        "objective": float(prob.value) if success else float("inf"),
        "status": prob.status,
        "success": success,
    }


@requires_extra("optimization")
def solve_socp(
    c: npt.NDArray[np.floating],
    A: npt.NDArray[np.floating],
    b: npt.NDArray[np.floating],
    cone_constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Solve a Second-Order Cone Program via cvxpy.

    Solves::

        min  c' x
        s.t. A x == b
             ||A_i x + b_i|| <= c_i' x + d_i   (for each cone constraint)

    Each element of *cone_constraints* is a dict with keys ``A_cone``,
    ``b_cone``, ``c_cone``, and ``d_cone``.

    Parameters:
        c: Linear objective vector ``(n,)``.
        A: Equality constraint matrix ``(m, n)``.
        b: Equality constraint RHS ``(m,)``.
        cone_constraints: List of SOC constraint dicts.  Each dict must
            contain ``A_cone`` (matrix), ``b_cone`` (vector),
            ``c_cone`` (vector), and ``d_cone`` (scalar).

    Returns:
        Dict with keys ``x``, ``objective``, ``status``, and ``success``.
    """
    import cvxpy as cp

    c_vec = np.asarray(c, dtype=float)
    A_mat = np.asarray(A, dtype=float)
    b_vec = np.asarray(b, dtype=float)
    n = c_vec.shape[0]

    x = cp.Variable(n)
    objective = cp.Minimize(c_vec @ x)

    constraints: list[Any] = [A_mat @ x == b_vec]

    if cone_constraints is not None:
        for cone in cone_constraints:
            A_c = np.asarray(cone["A_cone"], dtype=float)
            b_c = np.asarray(cone["b_cone"], dtype=float)
            c_c = np.asarray(cone["c_cone"], dtype=float)
            d_c = float(cone["d_cone"])
            constraints.append(cp.SOC(c_c @ x + d_c, A_c @ x + b_c))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    success = prob.status == "optimal"
    return {
        "x": np.array(x.value).flatten() if success else np.full(n, np.nan),
        "objective": float(prob.value) if success else float("inf"),
        "status": prob.status,
        "success": success,
    }


@requires_extra("optimization")
def solve_sdp(
    C: npt.NDArray[np.floating],
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Solve a Semidefinite Program via cvxpy.

    Solves::

        min  tr(C X)
        s.t. tr(A_i X) == b_i   for each constraint
             X >> 0              (positive semidefinite)

    Parameters:
        C: Symmetric cost matrix ``(n, n)``.
        constraints: List of constraint dicts, each with ``A`` (matrix) and
            ``b`` (scalar) specifying ``tr(A @ X) == b``.

    Returns:
        Dict with keys ``X`` (optimal matrix), ``objective``,
        ``status``, and ``success``.
    """
    import cvxpy as cp

    C_mat = np.asarray(C, dtype=float)
    n = C_mat.shape[0]

    X = cp.Variable((n, n), symmetric=True)
    objective = cp.Minimize(cp.trace(C_mat @ X))

    cvx_constraints: list[Any] = [X >> 0]
    for con in constraints:
        A_i = np.asarray(con["A"], dtype=float)
        b_i = float(con["b"])
        cvx_constraints.append(cp.trace(A_i @ X) == b_i)

    prob = cp.Problem(objective, cvx_constraints)
    prob.solve()

    success = prob.status == "optimal"
    return {
        "X": np.array(X.value) if success else np.full((n, n), np.nan),
        "objective": float(prob.value) if success else float("inf"),
        "status": prob.status,
        "success": success,
    }
