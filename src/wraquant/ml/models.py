"""Model wrappers for financial machine-learning workflows.

Functions that require scikit-learn are guarded by the
``@requires_extra('ml')`` decorator so that the rest of the package can
be imported without it.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "walk_forward_train",
    "ensemble_predict",
    "feature_importance_mdi",
    "feature_importance_mda",
    "sequential_feature_selection",
]


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------


@requires_extra("ml")
def walk_forward_train(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    train_size: int = 252,
    test_size: int = 21,
    step_size: int = 21,
) -> dict[str, Any]:
    """Walk-forward (expanding or rolling window) analysis.

    At each step the model is cloned (via scikit-learn's ``clone``),
    fitted on the training window, and used to predict the test window.

    Parameters
    ----------
    model : estimator
        A scikit-learn-compatible estimator that implements ``fit`` and
        ``predict``.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector.
    train_size : int
        Number of training observations in the first window.
    test_size : int
        Number of test observations per fold.
    step_size : int
        Number of observations to step forward between folds.

    Returns
    -------
    dict
        ``predictions``: np.ndarray of concatenated predictions,
        ``actuals``: np.ndarray of corresponding true values,
        ``test_indices``: np.ndarray of test-set indices,
        ``n_folds``: number of walk-forward folds.
    """
    from sklearn.base import clone

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n = len(X_arr)

    all_preds: list[np.ndarray] = []
    all_actuals: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    n_folds = 0

    start = 0
    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = min(train_end + test_size, n)

        X_train = X_arr[start:train_end]
        y_train = y_arr[start:train_end]
        X_test = X_arr[train_end:test_end]
        y_test = y_arr[train_end:test_end]

        m = clone(model)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

        all_preds.append(np.asarray(preds))
        all_actuals.append(np.asarray(y_test))
        all_indices.append(np.arange(train_end, test_end))
        n_folds += 1

        start += step_size

    return {
        "predictions": np.concatenate(all_preds) if all_preds else np.array([]),
        "actuals": np.concatenate(all_actuals) if all_actuals else np.array([]),
        "test_indices": (
            np.concatenate(all_indices) if all_indices else np.array([], dtype=int)
        ),
        "n_folds": n_folds,
    }


# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------


def ensemble_predict(
    models: Sequence[Any],
    X: pd.DataFrame | np.ndarray,
    method: Literal["mean", "median", "vote"] = "mean",
) -> np.ndarray:
    """Generate ensemble predictions from multiple fitted models.

    Parameters
    ----------
    models : Sequence
        Fitted scikit-learn-compatible estimators.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    method : {'mean', 'median', 'vote'}
        Aggregation method.  ``'mean'`` and ``'median'`` average the raw
        predictions; ``'vote'`` takes the mode (majority vote).

    Returns
    -------
    np.ndarray
        Aggregated predictions.
    """
    X_arr = np.asarray(X)
    preds = np.column_stack([np.asarray(m.predict(X_arr)) for m in models])

    if method == "mean":
        return preds.mean(axis=1)
    if method == "median":
        return np.median(preds, axis=1)
    if method == "vote":
        from scipy.stats import mode as _mode

        result = _mode(preds, axis=1, keepdims=False)
        return np.asarray(result.mode).ravel()

    raise ValueError(f"Unknown method '{method}'; use 'mean', 'median', or 'vote'.")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def feature_importance_mdi(
    model: Any,
    feature_names: Sequence[str],
) -> pd.Series:
    """Mean Decrease Impurity (MDI) feature importance.

    Reads ``model.feature_importances_`` (available on tree-based
    estimators after fitting) and returns a sorted ``pd.Series``.

    Parameters
    ----------
    model : estimator
        A fitted tree-based estimator with a ``feature_importances_``
        attribute (e.g. ``RandomForestClassifier``).
    feature_names : Sequence[str]
        Feature names corresponding to the columns of the training data.

    Returns
    -------
    pd.Series
        Importance values indexed by feature name, sorted descending.
    """
    importances = np.asarray(model.feature_importances_)
    series = pd.Series(importances, index=list(feature_names), name="mdi_importance")
    return series.sort_values(ascending=False)


@requires_extra("ml")
def feature_importance_mda(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int = 10,
) -> pd.Series:
    """Mean Decrease Accuracy (permutation importance).

    Repeatedly permutes each feature and measures the decrease in the
    model's score.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn-compatible estimator.
    X : pd.DataFrame or np.ndarray
        Feature matrix (test or validation set).
    y : pd.Series or np.ndarray
        True labels.
    feature_names : Sequence[str]
        Feature names corresponding to columns of *X*.
    n_repeats : int
        Number of permutation repeats per feature.

    Returns
    -------
    pd.Series
        Mean importance values indexed by feature name, sorted
        descending.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        np.asarray(X),
        np.asarray(y),
        n_repeats=n_repeats,
        random_state=42,
    )
    series = pd.Series(
        result.importances_mean,
        index=list(feature_names),
        name="mda_importance",
    )
    return series.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Sequential feature selection
# ---------------------------------------------------------------------------


@requires_extra("ml")
def sequential_feature_selection(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_features: int = 5,
    direction: Literal["forward", "backward"] = "forward",
    cv: int = 5,
) -> list[str | int]:
    """Sequential (forward / backward) feature selection.

    Parameters
    ----------
    model : estimator
        A scikit-learn-compatible estimator.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector.
    n_features : int
        Number of features to select.
    direction : {'forward', 'backward'}
        Selection direction.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    list[str | int]
        Selected feature names (if *X* is a DataFrame) or column
        indices.
    """
    from sklearn.feature_selection import SequentialFeatureSelector

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction=direction,
        cv=cv,
    )
    sfs.fit(np.asarray(X), np.asarray(y))
    support = sfs.get_support()

    if isinstance(X, pd.DataFrame):
        return list(X.columns[support])
    return list(np.where(support)[0])
