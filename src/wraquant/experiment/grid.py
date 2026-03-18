"""Parameter grid search for strategy optimization.

Provides tools for exhaustive and random parameter search over strategy
configurations, returning ranked results with the best-performing parameter
combinations.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable

import numpy as np


class ParameterGrid:
    """Generate all parameter combinations from a dict of lists.

    Parameters
    ----------
    param_dict : dict[str, list]
        Mapping of parameter names to lists of values to try.
        Example: ``{'fast_ma': [5, 10, 20], 'slow_ma': [50, 100, 200]}``.
    """

    def __init__(self, param_dict: dict[str, list]) -> None:
        self._param_dict = param_dict
        self._keys = sorted(param_dict.keys())
        self._values = [param_dict[k] for k in self._keys]

    def __iter__(self):
        """Yield dicts of parameter combinations.

        Yields
        ------
        dict[str, Any]
            A single parameter combination.
        """
        for combo in itertools.product(*self._values):
            yield dict(zip(self._keys, combo, strict=False))

    def __len__(self) -> int:
        """Return total number of parameter combinations.

        Returns
        -------
        int
            Product of the lengths of all parameter value lists.
        """
        length = 1
        for v in self._values:
            length *= len(v)
        return length

    def __repr__(self) -> str:
        return f"ParameterGrid({self._param_dict!r})"


def grid_search(
    objective_fn: Callable[..., float],
    param_grid: ParameterGrid | dict[str, list],
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Run objective function for each parameter combination.

    Parameters
    ----------
    objective_fn : Callable[..., float]
        Function that accepts keyword arguments matching the parameter names
        and returns a scalar score (higher is better).
    param_grid : ParameterGrid | dict[str, list]
        Parameter grid to search over. If a plain dict is provided it is
        wrapped in a ``ParameterGrid``.
    n_jobs : int, optional
        Number of parallel jobs. Currently only ``n_jobs=1`` is supported
        (sequential execution). Default is ``1``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``best_params`` (dict): Parameters that achieved the highest score.
        - ``best_score`` (float): The highest objective value found.
        - ``all_results`` (list[dict]): All evaluated combinations sorted by
          score in descending order.  Each entry has ``params`` and ``score``.
    """
    if isinstance(param_grid, dict):
        param_grid = ParameterGrid(param_grid)

    results: list[dict[str, Any]] = []
    for params in param_grid:
        score = objective_fn(**params)
        results.append({"params": params, "score": float(score)})

    results.sort(key=lambda r: r["score"], reverse=True)

    return {
        "best_params": results[0]["params"],
        "best_score": results[0]["score"],
        "all_results": results,
    }


def random_search(
    objective_fn: Callable[..., float],
    param_distributions: dict[str, Any],
    n_iter: int = 100,
    seed: int | None = None,
) -> dict[str, Any]:
    """Random parameter sampling from distributions.

    Parameters
    ----------
    objective_fn : Callable[..., float]
        Function that accepts keyword arguments matching the parameter names
        and returns a scalar score (higher is better).
    param_distributions : dict[str, Any]
        Mapping of parameter names to distributions.  Supported distribution
        specifications:

        - A **list** — a random element is chosen uniformly.
        - A **tuple** ``(low, high)`` — uniform sampling on ``[low, high)``.
        - A **dict** with key ``'type'``:

          - ``{'type': 'uniform', 'low': ..., 'high': ...}``
          - ``{'type': 'log-uniform', 'low': ..., 'high': ...}`` — sample
            ``exp(uniform(log(low), log(high)))``.
          - ``{'type': 'choice', 'values': [...]}`` — uniform random choice.
    n_iter : int, optional
        Number of random samples to draw. Default is ``100``.
    seed : int | None, optional
        Random seed for reproducibility. Default is ``None``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``best_params`` (dict): Parameters that achieved the highest score.
        - ``best_score`` (float): The highest objective value found.
        - ``all_results`` (list[dict]): All evaluated combinations sorted by
          score in descending order.
    """
    rng = np.random.default_rng(seed)

    def _sample(spec: Any) -> Any:
        if isinstance(spec, list):
            return spec[rng.integers(len(spec))]
        if isinstance(spec, tuple) and len(spec) == 2:
            low, high = spec
            return rng.uniform(low, high)
        if isinstance(spec, dict):
            dist_type = spec.get("type", "uniform")
            if dist_type == "uniform":
                return rng.uniform(spec["low"], spec["high"])
            if dist_type == "log-uniform":
                log_low = np.log(spec["low"])
                log_high = np.log(spec["high"])
                return float(np.exp(rng.uniform(log_low, log_high)))
            if dist_type == "choice":
                values = spec["values"]
                return values[rng.integers(len(values))]
            raise ValueError(f"Unknown distribution type: {dist_type!r}")
        raise TypeError(
            f"Unsupported distribution spec: {spec!r}. "
            "Use a list, tuple (low, high), or dict with 'type' key."
        )

    results: list[dict[str, Any]] = []
    keys = sorted(param_distributions.keys())

    for _ in range(n_iter):
        params = {k: _sample(param_distributions[k]) for k in keys}
        score = objective_fn(**params)
        results.append({"params": params, "score": float(score)})

    results.sort(key=lambda r: r["score"], reverse=True)

    return {
        "best_params": results[0]["params"],
        "best_score": results[0]["score"],
        "all_results": results,
    }
