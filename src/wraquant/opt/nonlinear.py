"""Nonlinear optimization wrappers.

Thin convenience layer over :mod:`scipy.optimize` for general-purpose
nonlinear programming, global optimization, and root-finding.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy import optimize


def minimize(
    fun: Callable[..., float],
    x0: npt.NDArray[np.floating],
    method: str = "SLSQP",
    bounds: list[tuple[float, float]] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    jac: Callable[..., npt.NDArray] | str | None = None,
    hess: Callable[..., npt.NDArray] | str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """General nonlinear program solver.

    Use this for any smooth optimization problem with nonlinear
    objectives or constraints, such as fitting option pricing models,
    calibrating yield curves, or custom portfolio objectives with
    non-convex penalties.

    Wrapper around :func:`scipy.optimize.minimize` that returns a plain
    dict instead of an ``OptimizeResult``.

    Parameters:
        fun: Scalar objective function ``f(x) -> float``.
        x0: Initial guess ``(n,)``.  A good starting point is critical
            for nonlinear problems.
        method: Optimisation algorithm (e.g. ``'SLSQP'``, ``'L-BFGS-B'``,
            ``'trust-constr'``).
        bounds: Variable bounds ``[(lb, ub), ...]``.
        constraints: List of constraint dicts accepted by scipy.
        jac: Jacobian of *fun* or a string method (``'2-point'``,
            ``'3-point'``, ``'cs'``).  Providing an analytical Jacobian
            significantly improves speed and convergence.
        hess: Hessian of *fun* or a string method.
        options: Solver-specific options dict (e.g.
            ``{'maxiter': 1000}``).

    Returns:
        Dict with keys ``x`` (solution), ``fun`` (objective value at
        solution), ``success`` (bool), and ``n_iter`` (iteration count).

    Example:
        >>> import numpy as np
        >>> # Minimize Rosenbrock function
        >>> def rosenbrock(x):
        ...     return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        >>> result = minimize(rosenbrock, np.array([0.0, 0.0]))
        >>> result['success']
        True
        >>> np.allclose(result['x'], [1, 1], atol=1e-3)
        True

    See Also:
        global_minimize: Global optimization for multimodal problems.
        wraquant.opt.convex.solve_qp: Quadratic programming (convex).
    """
    x0 = np.asarray(x0, dtype=float)

    result = optimize.minimize(
        fun,
        x0,
        method=method,
        bounds=bounds,
        constraints=constraints,
        jac=jac,
        hess=hess,
        options=options,
    )

    return {
        "x": result.x,
        "fun": float(result.fun),
        "success": bool(result.success),
        "n_iter": int(result.nit) if hasattr(result, "nit") else 0,
    }


def global_minimize(
    fun: Callable[..., float],
    bounds: list[tuple[float, float]],
    method: str = "differential_evolution",
    seed: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Global optimisation via stochastic search methods.

    Use global optimization when the objective has multiple local minima
    (e.g., calibrating stochastic volatility models, fitting regime
    switching parameters, or optimising non-convex trading strategies).
    Local solvers get trapped; global methods explore the search space
    before converging.

    Supports ``differential_evolution``, ``dual_annealing``, and
    ``basinhopping``.

    Parameters:
        fun: Scalar objective function ``f(x) -> float``.
        bounds: Search bounds ``[(lb, ub), ...]`` for each variable.
        method: Algorithm name -- ``'differential_evolution'``
            (population-based, most robust), ``'dual_annealing'``
            (simulated annealing + local search), or
            ``'basinhopping'`` (random restarts + local optimization).
        seed: Random seed for reproducibility.
        **kwargs: Additional keyword arguments forwarded to the chosen
            scipy solver.

    Returns:
        Dict with keys ``x`` (best solution found), ``fun`` (objective
        at solution), ``success`` (bool), and ``n_iter``.

    Raises:
        ValueError: If *method* is not recognised.

    Example:
        >>> import numpy as np
        >>> # Rastrigin function (many local minima, global min at origin)
        >>> def rastrigin(x):
        ...     return 20 + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)
        >>> result = global_minimize(rastrigin, [(-5, 5), (-5, 5)], seed=42)
        >>> result['fun'] < 1.0  # near global minimum of 0
        True

    See Also:
        minimize: Local nonlinear solver (faster but gets trapped).
    """
    method = method.lower()

    if method == "differential_evolution":
        result = optimize.differential_evolution(fun, bounds, seed=seed, **kwargs)
    elif method == "dual_annealing":
        result = optimize.dual_annealing(fun, bounds, seed=seed, **kwargs)
    elif method == "basinhopping":
        # basinhopping needs x0 and doesn't accept bounds the same way
        rng = np.random.default_rng(seed)
        x0 = np.array([(lb + ub) / 2 for lb, ub in bounds])
        minimizer_kwargs = kwargs.pop("minimizer_kwargs", {})
        minimizer_kwargs.setdefault("bounds", bounds)
        result = optimize.basinhopping(
            fun,
            x0,
            seed=rng.integers(0, 2**31),
            minimizer_kwargs=minimizer_kwargs,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from "
            "'differential_evolution', 'dual_annealing', 'basinhopping'."
        )

    return {
        "x": result.x,
        "fun": float(result.fun),
        "success": bool(getattr(result, "success", True)),
        "n_iter": int(getattr(result, "nit", 0)),
    }


def root_find(
    fun: Callable[..., npt.NDArray | float],
    x0: npt.NDArray[np.floating],
    method: str = "hybr",
    jac: Callable[..., npt.NDArray] | str | bool | None = None,
) -> dict[str, Any]:
    """Find roots of a nonlinear system.

    Use root finding for implied volatility calculation (solve
    BS(sigma) - market_price = 0), yield-to-maturity computation,
    or any problem where you need to find x such that f(x) = 0.

    Wrapper around :func:`scipy.optimize.root`.

    Parameters:
        fun: Vector function ``f(x) -> array`` whose root is sought.
        x0: Initial guess ``(n,)``.
        method: Solver algorithm (e.g. ``'hybr'`` (default, hybrid
            Powell), ``'lm'`` (Levenberg-Marquardt), ``'broyden1'``).
        jac: Jacobian or a flag/string for finite-difference
            approximation.

    Returns:
        Dict with keys ``x`` (root), ``fun`` (residual at root --
        should be near zero), ``success`` (bool), and ``n_iter``
        (function evaluations or iteration count).

    Example:
        >>> import numpy as np
        >>> # Find x such that x^2 - 4 = 0
        >>> result = root_find(lambda x: x**2 - 4, np.array([1.0]))
        >>> result['success']
        True
        >>> np.isclose(result['x'][0], 2.0)
        True

    See Also:
        minimize: Find minimum of a function (not root).
    """
    x0 = np.asarray(x0, dtype=float)

    result = optimize.root(fun, x0, method=method, jac=jac)

    return {
        "x": result.x,
        "fun": result.fun,
        "success": bool(result.success),
        "n_iter": int(getattr(result, "nfev", getattr(result, "nit", 0))),
    }
