"""Multi-objective optimization.

Core routines (weighted-sum Pareto approximation and epsilon-constraint)
use pure scipy. NSGA-II is gated behind the ``optimization`` extra
(pymoo).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy import optimize

from wraquant.core.decorators import requires_extra


def pareto_front(
    objectives: list[Callable[..., float]],
    constraints: list[dict[str, Any]] | None = None,
    n_points: int = 50,
    bounds: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Approximate the Pareto frontier via the weighted-sum method.

    Use the weighted-sum approach for a quick Pareto front approximation
    when you have a convex bi-objective problem (e.g., risk vs. return,
    cost vs. tracking error).  For non-convex problems or more than 2
    objectives, use ``nsga2`` instead.

    For each of *n_points* evenly-spaced weight vectors the scalar
    weighted-sum of the objectives is minimised using SLSQP.

    Parameters:
        objectives: List of scalar objective functions ``f(x) -> float``
            to be minimised simultaneously.
        constraints: Scipy-compatible constraint dicts applied to every
            sub-problem.
        n_points: Number of Pareto-front samples (default 50).
        bounds: Variable bounds ``[(lb, ub), ...]``.

    Returns:
        Dict with keys ``points`` (non-dominated decision vectors, shape
        ``(k, n)``), ``objectives`` (corresponding objective values,
        shape ``(k, m)``), and ``n_points`` (number of non-dominated
        points found).

    Example:
        >>> import numpy as np
        >>> f1 = lambda x: float(x[0] ** 2)
        >>> f2 = lambda x: float((x[0] - 1) ** 2)
        >>> result = pareto_front([f1, f2], bounds=[(0, 1)], n_points=20)
        >>> result['n_points'] > 0
        True

    See Also:
        nsga2: Evolutionary multi-objective optimizer (handles non-convex).
        epsilon_constraint: Epsilon-constraint Pareto method.
    """
    n_obj = len(objectives)
    if n_obj < 2:
        raise ValueError("Need at least two objectives for a Pareto front.")

    # Generate weight vectors (simplex grid for 2 objectives,
    # uniform for >2)
    if n_obj == 2:
        alphas = np.linspace(0.0, 1.0, n_points)
        weight_vectors = np.column_stack([alphas, 1.0 - alphas])
    else:
        rng = np.random.default_rng(0)
        raw = rng.dirichlet(np.ones(n_obj), size=n_points)
        weight_vectors = raw

    if bounds is not None:
        n_vars = len(bounds)
    else:
        # infer from constraints or default
        n_vars = 2

    points: list[npt.NDArray] = []
    obj_values: list[npt.NDArray] = []

    for w in weight_vectors:

        def weighted_obj(x: npt.NDArray, _w: npt.NDArray = w) -> float:
            return float(sum(_w[i] * objectives[i](x) for i in range(n_obj)))

        x0 = np.zeros(n_vars)
        if bounds is not None:
            x0 = np.array([(lb + ub) / 2 for lb, ub in bounds])

        result = optimize.minimize(
            weighted_obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            points.append(result.x)
            obj_values.append(np.array([objectives[i](result.x) for i in range(n_obj)]))

    if not points:
        return {
            "points": np.empty((0, n_vars)),
            "objectives": np.empty((0, n_obj)),
            "n_points": 0,
        }

    pts = np.array(points)
    objs = np.array(obj_values)

    # Filter dominated solutions
    mask = _non_dominated_mask(objs)
    pts = pts[mask]
    objs = objs[mask]

    return {
        "points": pts,
        "objectives": objs,
        "n_points": int(pts.shape[0]),
    }


def _non_dominated_mask(objectives: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    """Return boolean mask of non-dominated rows (all objectives minimised)."""
    n = objectives.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(i + 1, n):
            if not mask[j]:
                continue
            # Check if j dominates i
            if np.all(objectives[j] <= objectives[i]) and np.any(
                objectives[j] < objectives[i]
            ):
                mask[i] = False
                break
            # Check if i dominates j
            if np.all(objectives[i] <= objectives[j]) and np.any(
                objectives[i] < objectives[j]
            ):
                mask[j] = False
    return mask


@requires_extra("optimization")
def nsga2(
    objectives: list[Callable[..., float]],
    bounds: list[tuple[float, float]],
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int | None = None,
) -> dict[str, Any]:
    """NSGA-II multi-objective optimisation via pymoo.

    Use NSGA-II when you have two or more competing objectives and want
    to find the Pareto-optimal frontier.  In finance, this is used for
    risk-return trade-offs, cost-tracking trade-offs in execution, and
    multi-factor portfolio construction.  NSGA-II is an evolutionary
    algorithm that maintains diversity across the Pareto front.

    Parameters:
        objectives: List of scalar objective functions ``f(x) -> float``,
            each to be minimised.
        bounds: Variable bounds ``[(lb, ub), ...]``.
        pop_size: Population size (default 100).  Larger populations
            improve frontier coverage but increase computation.
        n_gen: Number of generations (default 200).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:

        - ``pareto_front``: np.ndarray of decision vectors on the
          Pareto front (shape ``(n_points, n_var)``).
        - ``pareto_objectives``: np.ndarray of objective values (shape
          ``(n_points, n_obj)``).
        - ``n_points``: number of Pareto-optimal solutions found.

    Example:
        >>> import numpy as np
        >>> # Minimise x^2 and (x-2)^2 simultaneously
        >>> f1 = lambda x: float(x[0] ** 2)
        >>> f2 = lambda x: float((x[0] - 2) ** 2)
        >>> result = nsga2([f1, f2], bounds=[(0, 3)], pop_size=50, n_gen=100, seed=42)
        >>> result['n_points'] > 0
        True

    References:
        - Deb et al. (2002), "A Fast and Elitist Multiobjective Genetic
          Algorithm: NSGA-II"

    See Also:
        pareto_front: Weighted-sum Pareto approximation (no pymoo needed).
        epsilon_constraint: Epsilon-constraint method.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination

    n_var = len(bounds)
    n_obj = len(objectives)
    xl = np.array([b[0] for b in bounds])
    xu = np.array([b[1] for b in bounds])

    class _Problem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        def _evaluate(
            self, x: npt.NDArray, out: dict, *args: Any, **kwargs: Any
        ) -> None:
            out["F"] = np.array([obj(x) for obj in objectives])

    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)

    result = pymoo_minimize(
        _Problem(),
        algorithm,
        termination,
        seed=seed,
        verbose=False,
    )

    if result.F is not None:
        return {
            "pareto_front": result.X,
            "pareto_objectives": result.F,
            "n_points": int(result.F.shape[0]),
        }

    return {
        "pareto_front": np.empty((0, n_var)),
        "pareto_objectives": np.empty((0, n_obj)),
        "n_points": 0,
    }


def epsilon_constraint(
    primary_obj: Callable[..., float],
    secondary_objs: list[Callable[..., float]],
    epsilon_values: list[npt.NDArray[np.floating]],
    bounds: list[tuple[float, float]] | None = None,
    constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Epsilon-constraint method for multi-objective optimization.

    Minimises the *primary_obj* while constraining each secondary
    objective to be at most the corresponding epsilon value.

    Parameters:
        primary_obj: Primary objective function to minimise.
        secondary_objs: Secondary objective functions.
        epsilon_values: List of 1-D arrays.  For each secondary objective
            ``k``, ``epsilon_values[k]`` gives the set of upper-bound
            values to iterate over.  The Cartesian product of all epsilon
            arrays defines the grid of sub-problems.
        bounds: Variable bounds ``[(lb, ub), ...]``.
        constraints: Additional scipy constraint dicts.

    Returns:
        Dict with keys ``points`` (decision vectors), ``primary_values``
        (primary objective at each point), ``secondary_values`` (secondary
        objective values), and ``n_points``.
    """
    if len(secondary_objs) != len(epsilon_values):
        raise ValueError("Length of secondary_objs and epsilon_values must match.")

    if bounds is not None:
        n_vars = len(bounds)
    else:
        n_vars = 2

    # Build epsilon grid via Cartesian product
    eps_arrays = [np.asarray(ev, dtype=float) for ev in epsilon_values]
    grids = np.meshgrid(*eps_arrays, indexing="ij")
    eps_grid = np.column_stack([g.ravel() for g in grids])

    points: list[npt.NDArray] = []
    primary_vals: list[float] = []
    secondary_vals: list[npt.NDArray] = []

    base_constraints = list(constraints) if constraints else []

    for eps_row in eps_grid:
        sub_constraints = list(base_constraints)
        for k, sec_obj in enumerate(secondary_objs):
            eps_k = float(eps_row[k])
            sub_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, _f=sec_obj, _e=eps_k: _e - _f(x),
                }
            )

        x0 = np.zeros(n_vars)
        if bounds is not None:
            x0 = np.array([(lb + ub) / 2 for lb, ub in bounds])

        result = optimize.minimize(
            primary_obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=sub_constraints,
        )

        if result.success:
            points.append(result.x)
            primary_vals.append(float(result.fun))
            secondary_vals.append(np.array([sec(result.x) for sec in secondary_objs]))

    if not points:
        n_sec = len(secondary_objs)
        return {
            "points": np.empty((0, n_vars)),
            "primary_values": np.empty(0),
            "secondary_values": np.empty((0, n_sec)),
            "n_points": 0,
        }

    return {
        "points": np.array(points),
        "primary_values": np.array(primary_vals),
        "secondary_values": np.array(secondary_vals),
        "n_points": len(points),
    }
