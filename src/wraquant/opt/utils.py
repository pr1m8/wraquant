"""Optimization constraint and utility helpers.

Convenience functions for building constraints commonly used in
portfolio and general optimisation problems.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def weight_constraint(
    n_assets: int,
    lb: float = 0.0,
    ub: float = 1.0,
) -> dict[str, Any]:
    """Generate uniform bounds for portfolio weights.

    Parameters:
        n_assets: Number of assets.
        lb: Lower bound for each weight.
        ub: Upper bound for each weight.

    Returns:
        Dict with key ``bounds`` — a list of ``(lb, ub)`` tuples, one per
        asset.
    """
    return {"bounds": [(lb, ub)] * n_assets}


def sum_to_one_constraint(n_assets: int) -> dict[str, Any]:
    """Equality constraint requiring weights to sum to one.

    Compatible with :func:`scipy.optimize.minimize` constraint format.

    Parameters:
        n_assets: Number of assets (used only for documentation /
            introspection; the constraint function works for any length).

    Returns:
        Scipy-style constraint dict ``{'type': 'eq', 'fun': ...}``.
    """
    return {
        "type": "eq",
        "fun": lambda w: float(np.sum(w) - 1.0),
        "n_assets": n_assets,
    }


def sector_constraints(
    n_assets: int,
    sectors: dict[str, list[int]],
    sector_limits: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    """Build inequality constraints for sector / group weight limits.

    Parameters:
        n_assets: Total number of assets.
        sectors: Mapping of sector name to list of asset indices belonging
            to that sector.
        sector_limits: Mapping of sector name to ``(min_weight, max_weight)``
            tuple for the aggregate sector weight.

    Returns:
        List of scipy-style constraint dicts (``'ineq'`` type).  Two
        constraints per sector — one for the lower bound and one for the
        upper bound.

    Example:
        >>> cons = sector_constraints(
        ...     5,
        ...     sectors={"tech": [0, 1], "energy": [2, 3, 4]},
        ...     sector_limits={"tech": (0.1, 0.5), "energy": (0.2, 0.6)},
        ... )  # doctest: +SKIP
    """
    result: list[dict[str, Any]] = []

    for sector_name, indices in sectors.items():
        if sector_name not in sector_limits:
            continue
        lb, ub = sector_limits[sector_name]
        idx = list(indices)

        # sum(w[idx]) >= lb  =>  sum(w[idx]) - lb >= 0
        result.append(
            {
                "type": "ineq",
                "fun": lambda w, _idx=idx, _lb=lb: float(np.sum(w[_idx]) - _lb),
                "sector": sector_name,
                "bound": "lower",
            }
        )

        # sum(w[idx]) <= ub  =>  ub - sum(w[idx]) >= 0
        result.append(
            {
                "type": "ineq",
                "fun": lambda w, _idx=idx, _ub=ub: float(_ub - np.sum(w[_idx])),
                "sector": sector_name,
                "bound": "upper",
            }
        )

    return result


def turnover_constraint(
    current_weights: npt.NDArray[np.floating],
    max_turnover: float,
) -> dict[str, Any]:
    """Constraint limiting portfolio turnover.

    Turnover is defined as ``0.5 * sum(|w_new - w_old|)``.

    Parameters:
        current_weights: Current portfolio weights ``(n,)``.
        max_turnover: Maximum allowed one-way turnover (e.g. ``0.20``
            for 20 %).

    Returns:
        Scipy-style inequality constraint dict.
    """
    w_old = np.asarray(current_weights, dtype=float)

    return {
        "type": "ineq",
        "fun": lambda w, _w_old=w_old, _mt=max_turnover: float(
            _mt - 0.5 * np.sum(np.abs(np.asarray(w) - _w_old))
        ),
    }


def cardinality_constraint(
    n_assets: int,
    max_holdings: int,
) -> dict[str, Any]:
    """Information dict describing a cardinality constraint.

    Cardinality constraints (limiting the number of non-zero weights)
    are not directly expressible as smooth inequality constraints for
    gradient-based solvers.  This helper returns a description dict
    that can be consumed by MILP-based or heuristic optimisers.

    Parameters:
        n_assets: Total number of assets.
        max_holdings: Maximum number of assets with non-zero weight.

    Returns:
        Dict with keys ``n_assets``, ``max_holdings``, and
        ``description``.
    """
    return {
        "n_assets": n_assets,
        "max_holdings": max_holdings,
        "description": (
            f"At most {max_holdings} of {n_assets} assets may have " "non-zero weight."
        ),
    }
