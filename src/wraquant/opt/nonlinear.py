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

    Wrapper around :func:`scipy.optimize.minimize` that returns a plain
    dict instead of an ``OptimizeResult``.

    Parameters:
        fun: Scalar objective function ``f(x) -> float``.
        x0: Initial guess ``(n,)``.
        method: Optimisation algorithm (e.g. ``'SLSQP'``, ``'L-BFGS-B'``,
            ``'trust-constr'``).
        bounds: Variable bounds ``[(lb, ub), ...]``.
        constraints: List of constraint dicts accepted by scipy.
        jac: Jacobian of *fun* or a string method (``'2-point'``,
            ``'3-point'``, ``'cs'``).
        hess: Hessian of *fun* or a string method.
        options: Solver-specific options dict.

    Returns:
        Dict with keys ``x`` (solution), ``fun`` (objective value),
        ``success`` (bool), and ``n_iter`` (iteration count).
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

    Supports ``differential_evolution``, ``dual_annealing``, and
    ``basinhopping``.

    Parameters:
        fun: Scalar objective function ``f(x) -> float``.
        bounds: Search bounds ``[(lb, ub), ...]`` for each variable.
        method: Algorithm name — ``'differential_evolution'``,
            ``'dual_annealing'``, or ``'basinhopping'``.
        seed: Random seed for reproducibility.
        **kwargs: Additional keyword arguments forwarded to the chosen
            scipy solver.

    Returns:
        Dict with keys ``x``, ``fun``, ``success``, and ``n_iter``.

    Raises:
        ValueError: If *method* is not recognised.
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

    Wrapper around :func:`scipy.optimize.root`.

    Parameters:
        fun: Vector function ``f(x) -> array`` whose root is sought.
        x0: Initial guess ``(n,)``.
        method: Solver algorithm (e.g. ``'hybr'``, ``'lm'``, ``'broyden1'``).
        jac: Jacobian or a flag/string for finite-difference approximation.

    Returns:
        Dict with keys ``x`` (root), ``fun`` (residual at root),
        ``success`` (bool), and ``n_iter`` (function evaluations or
        iteration count).
    """
    x0 = np.asarray(x0, dtype=float)

    result = optimize.root(fun, x0, method=method, jac=jac)

    return {
        "x": result.x,
        "fun": result.fun,
        "success": bool(result.success),
        "n_iter": int(getattr(result, "nfev", getattr(result, "nit", 0))),
    }
