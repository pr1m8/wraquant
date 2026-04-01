"""Sensitivity analysis for strategy parameters.

Tools for understanding how sensitive a strategy's performance is to its
parameter values, including one-at-a-time sweeps, two-dimensional heatmaps,
walk-forward stability scoring, and Monte-Carlo robustness checks.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from wraquant.core._coerce import coerce_array  # noqa: F401


def parameter_sensitivity(
    objective_fn: Callable[..., float],
    base_params: dict[str, Any],
    param_name: str,
    values: list[Any],
) -> pd.DataFrame:
    """Vary one parameter while holding others fixed.

    Parameters
    ----------
    objective_fn : Callable[..., float]
        Objective function accepting keyword arguments.
    base_params : dict[str, Any]
        Baseline parameter values.  Only ``param_name`` is varied.
    param_name : str
        The name of the parameter to sweep.
    values : list[Any]
        The values to test for ``param_name``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``param_name`` (the swept values) and
        ``score`` (the corresponding objective values).
    """
    scores: list[float] = []
    for v in values:
        params = {**base_params, param_name: v}
        scores.append(float(objective_fn(**params)))

    return pd.DataFrame({param_name: values, "score": scores})


def parameter_heatmap(
    objective_fn: Callable[..., float],
    base_params: dict[str, Any],
    param1_name: str,
    param1_values: list[Any],
    param2_name: str,
    param2_values: list[Any],
) -> pd.DataFrame:
    """Two-dimensional parameter sweep.

    Parameters
    ----------
    objective_fn : Callable[..., float]
        Objective function accepting keyword arguments.
    base_params : dict[str, Any]
        Baseline parameter values.  Only ``param1_name`` and
        ``param2_name`` are varied.
    param1_name : str
        Name of the first parameter (used as the DataFrame index).
    param1_values : list[Any]
        Values to test for the first parameter.
    param2_name : str
        Name of the second parameter (used as the DataFrame columns).
    param2_values : list[Any]
        Values to test for the second parameter.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``param1_values`` as the index and
        ``param2_values`` as the columns; cell values are objective
        scores.
    """
    results = np.empty((len(param1_values), len(param2_values)))

    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            params = {**base_params, param1_name: v1, param2_name: v2}
            results[i, j] = objective_fn(**params)

    return pd.DataFrame(
        results,
        index=pd.Index(param1_values, name=param1_name),
        columns=pd.Index(param2_values, name=param2_name),
    )


def stability_score(walk_forward_results: dict[str, Any]) -> float:
    """Measure parameter stability across walk-forward folds.

    Computes the fraction of consecutive folds in which the optimal
    parameter set remains unchanged.

    Parameters
    ----------
    walk_forward_results : dict[str, Any]
        Output of
        :func:`wraquant.experiment.walk_forward.walk_forward_optimize`.

    Returns
    -------
    float
        Stability score in ``[0, 1]``.  ``1.0`` means the same
        parameters were optimal in every fold.
    """
    fold_results = walk_forward_results["fold_results"]
    if len(fold_results) <= 1:
        return 1.0

    param_tuples = [tuple(sorted(f["best_params"].items())) for f in fold_results]

    same = sum(
        1 for i in range(1, len(param_tuples)) if param_tuples[i] == param_tuples[i - 1]
    )

    return same / (len(param_tuples) - 1)


def robustness_check(
    objective_fn: Callable[..., float],
    params: dict[str, Any],
    n_perturbations: int = 100,
    noise_std: float = 0.1,
    seed: int | None = None,
) -> dict[str, Any]:
    """Perturb parameters randomly and measure objective stability.

    Each numeric parameter is perturbed by additive Gaussian noise
    scaled by ``noise_std * abs(value)`` (or ``noise_std`` when the base
    value is zero).  Non-numeric parameters are left unchanged.

    Parameters
    ----------
    objective_fn : Callable[..., float]
        Objective function accepting keyword arguments.
    params : dict[str, Any]
        Baseline parameter values.
    n_perturbations : int, optional
        Number of random perturbations.  Default is ``100``.
    noise_std : float, optional
        Relative standard deviation of the perturbation noise.  Default
        is ``0.1`` (10%).
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``base_score`` (float): Objective at the unperturbed params.
        - ``mean_score`` (float): Mean objective across perturbations.
        - ``std_score`` (float): Standard deviation of perturbed scores.
        - ``min_score`` (float): Minimum perturbed score.
        - ``max_score`` (float): Maximum perturbed score.
        - ``scores`` (numpy.ndarray): All perturbed scores.
    """
    rng = np.random.default_rng(seed)
    base_score = float(objective_fn(**params))

    numeric_keys = [
        k for k, v in params.items() if isinstance(v, (int, float, np.number))
    ]

    scores = np.empty(n_perturbations)

    for i in range(n_perturbations):
        perturbed = dict(params)
        for k in numeric_keys:
            base_val = params[k]
            scale = noise_std * abs(base_val) if base_val != 0 else noise_std
            noise = rng.normal(0, scale)
            perturbed_val = base_val + noise
            # Preserve int type when the original is int
            if isinstance(base_val, (int, np.integer)):
                perturbed_val = int(round(perturbed_val))
            perturbed[k] = perturbed_val

        scores[i] = objective_fn(**perturbed)

    return {
        "base_score": base_score,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "scores": scores,
    }
