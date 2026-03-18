"""Walk-forward optimization.

Provides rolling and expanding window splits along with a walk-forward
optimization driver that trains on each in-sample window, selects the best
parameters, then evaluates on the out-of-sample window.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from wraquant.experiment.grid import ParameterGrid


def rolling_window_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate rolling (fixed-length) window splits.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    train_size : int
        Number of samples in each training window.
    test_size : int
        Number of samples in each test window.
    step_size : int | None, optional
        Number of samples to advance between folds.  Defaults to
        ``test_size`` when ``None``.

    Returns
    -------
    list[tuple[numpy.ndarray, numpy.ndarray]]
        List of ``(train_indices, test_indices)`` tuples.  Indices are
        integer ``numpy`` arrays suitable for slicing.
    """
    if step_size is None:
        step_size = test_size

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0

    while start + train_size + test_size <= n_samples:
        train_idx = np.arange(start, start + train_size)
        test_start = start + train_size
        test_idx = np.arange(test_start, test_start + test_size)
        splits.append((train_idx, test_idx))
        start += step_size

    return splits


def expanding_window_splits(
    n_samples: int,
    min_train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding (anchored) window splits.

    The training window always starts at index 0 and grows with each fold.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    min_train_size : int
        Minimum number of samples in the first training window.
    test_size : int
        Number of samples in each test window.
    step_size : int | None, optional
        Number of samples to advance between folds.  Defaults to
        ``test_size`` when ``None``.

    Returns
    -------
    list[tuple[numpy.ndarray, numpy.ndarray]]
        List of ``(train_indices, test_indices)`` tuples with expanding
        training sets.
    """
    if step_size is None:
        step_size = test_size

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train_size

    while train_end + test_size <= n_samples:
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, train_end + test_size)
        splits.append((train_idx, test_idx))
        train_end += step_size

    return splits


def walk_forward_optimize(
    objective_fn: Callable[..., float],
    param_grid: ParameterGrid | dict[str, list],
    data: np.ndarray,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    anchored: bool = False,
) -> dict[str, Any]:
    """Walk-forward optimization over rolling or expanding windows.

    For each fold the best parameters are found on the training window using
    an exhaustive grid search, and then the objective function is evaluated
    on the test window with those parameters.

    The ``objective_fn`` must accept a ``data`` keyword argument (the
    window slice) in addition to the strategy parameters defined in
    ``param_grid``.

    Parameters
    ----------
    objective_fn : Callable[..., float]
        Objective function.  Called as
        ``objective_fn(data=window, **params)`` and must return a scalar
        score (higher is better).
    param_grid : ParameterGrid | dict[str, list]
        Parameter grid for optimization.
    data : numpy.ndarray
        The full dataset.  Individual windows are sliced from this array.
    train_size : int
        Training window size (or minimum training size when
        ``anchored=True``).
    test_size : int
        Test window size.
    step_size : int | None, optional
        Number of samples to advance between folds.  Defaults to
        ``test_size``.
    anchored : bool, optional
        If ``True``, use expanding windows (training always starts at
        index 0).  Default is ``False`` (rolling windows).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``fold_results`` (list[dict]): Per-fold information including
          ``train_indices``, ``test_indices``, ``best_params``,
          ``train_score``, and ``test_score``.
        - ``aggregate_metrics`` (dict): ``mean_test_score``,
          ``std_test_score``, ``mean_train_score``, ``std_train_score``.
        - ``stability`` (dict): ``unique_param_sets`` (number of distinct
          best-param dicts across folds), ``param_change_rate`` (fraction
          of folds where best params differ from the previous fold).
    """
    if isinstance(param_grid, dict):
        param_grid = ParameterGrid(param_grid)

    n_samples = len(data)

    if anchored:
        splits = expanding_window_splits(n_samples, train_size, test_size, step_size)
    else:
        splits = rolling_window_splits(n_samples, train_size, test_size, step_size)

    fold_results: list[dict[str, Any]] = []

    for train_idx, test_idx in splits:
        train_data = data[train_idx]
        test_data = data[test_idx]

        # Grid search on training data
        best_score = -np.inf
        best_params: dict[str, Any] = {}
        for params in param_grid:
            score = objective_fn(data=train_data, **params)
            if score > best_score:
                best_score = score
                best_params = params

        # Evaluate on test data
        test_score = objective_fn(data=test_data, **best_params)

        fold_results.append(
            {
                "train_indices": train_idx,
                "test_indices": test_idx,
                "best_params": best_params,
                "train_score": float(best_score),
                "test_score": float(test_score),
            }
        )

    # Aggregate metrics
    train_scores = np.array([f["train_score"] for f in fold_results])
    test_scores = np.array([f["test_score"] for f in fold_results])

    aggregate_metrics = {
        "mean_test_score": float(np.mean(test_scores)),
        "std_test_score": float(np.std(test_scores)),
        "mean_train_score": float(np.mean(train_scores)),
        "std_train_score": float(np.std(train_scores)),
    }

    # Stability analysis
    param_tuples = [tuple(sorted(f["best_params"].items())) for f in fold_results]
    unique_param_sets = len(set(param_tuples))

    changes = 0
    for i in range(1, len(param_tuples)):
        if param_tuples[i] != param_tuples[i - 1]:
            changes += 1
    param_change_rate = changes / max(len(param_tuples) - 1, 1)

    stability = {
        "unique_param_sets": unique_param_sets,
        "param_change_rate": float(param_change_rate),
    }

    return {
        "fold_results": fold_results,
        "aggregate_metrics": aggregate_metrics,
        "stability": stability,
    }
