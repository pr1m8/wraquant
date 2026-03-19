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

    Use walk-forward analysis to evaluate a model under realistic conditions
    where only past data is available for training at each step.  This is
    the standard time-series cross-validation approach in quantitative
    finance, avoiding the look-ahead bias inherent in random K-fold splits.

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
        Number of training observations in the first window (default 252,
        approximately one trading year).
    test_size : int
        Number of test observations per fold (default 21, approximately
        one trading month).
    step_size : int
        Number of observations to step forward between folds.

    Returns
    -------
    dict
        ``predictions`` : np.ndarray
            Concatenated out-of-sample predictions across all folds.
        ``actuals`` : np.ndarray
            Corresponding true values.  Compare with predictions to
            measure forecast accuracy.
        ``test_indices`` : np.ndarray
            Original row indices for each prediction, useful for
            aligning results back to a DatetimeIndex.
        ``n_folds`` : int
            Number of walk-forward folds executed.

    Example
    -------
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np, pandas as pd
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.randn(500, 3), columns=['mom', 'vol', 'size'])
    >>> y = X['mom'] * 0.5 + np.random.randn(500) * 0.1
    >>> result = walk_forward_train(Ridge(), X, y, train_size=252, test_size=21)
    >>> result['n_folds'] > 0
    True
    >>> len(result['predictions']) == len(result['actuals'])
    True

    Notes
    -----
    The window is *expanding* (all data from the start up to the current
    train end is used).  For a rolling window, see
    ``wraquant.ml.pipeline.walk_forward_backtest`` which supports both modes.

    See Also
    --------
    wraquant.ml.pipeline.walk_forward_backtest : Full walk-forward backtest with PnL.
    wraquant.ml.preprocessing.purged_kfold : Purged K-fold cross-validation.
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

    Use ensemble prediction to combine several models (e.g., Ridge, Random
    Forest, Gradient Boosting) into a single, more robust forecast.  Ensembles
    reduce variance and are standard practice in alpha research and
    competition-winning pipelines.

    Parameters
    ----------
    models : Sequence
        Fitted scikit-learn-compatible estimators.  Each must implement
        ``predict(X)``.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    method : {'mean', 'median', 'vote'}
        Aggregation method.  ``'mean'`` and ``'median'`` average the raw
        predictions (best for regression); ``'vote'`` takes the mode
        (majority vote, best for classification).

    Returns
    -------
    np.ndarray
        Aggregated predictions.  For ``'mean'``/``'median'``, the values
        are continuous.  For ``'vote'``, the values are discrete class
        labels.

    Example
    -------
    >>> from sklearn.linear_model import Ridge, Lasso
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X_train = np.random.randn(200, 3)
    >>> y_train = X_train @ [1, 0.5, 0] + np.random.randn(200) * 0.1
    >>> m1 = Ridge().fit(X_train, y_train)
    >>> m2 = Lasso(alpha=0.01).fit(X_train, y_train)
    >>> X_test = np.random.randn(50, 3)
    >>> preds = ensemble_predict([m1, m2], X_test, method='mean')
    >>> preds.shape
    (50,)

    See Also
    --------
    walk_forward_train : Walk-forward evaluation for individual models.
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

    Use MDI as a fast, first-pass feature ranking after fitting a tree-based
    model.  MDI measures how much each feature contributes to reducing node
    impurity (Gini for classification, variance for regression) across all
    trees.

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
        Higher values indicate features that contributed more to splits.
        Values sum to 1.0 for scikit-learn tree ensembles.

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(300, 4)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    >>> imp = feature_importance_mdi(rf, ['momentum', 'vol', 'size', 'value'])
    >>> imp.index[0]  # most important feature
    'momentum'

    Notes
    -----
    MDI is biased toward high-cardinality and continuous features.  For an
    unbiased alternative, use ``feature_importance_mda`` (permutation
    importance).

    See Also
    --------
    feature_importance_mda : Permutation-based importance (unbiased).
    wraquant.ml.advanced.random_forest_importance : Combined RF fit + importance.
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

    Use MDA when you need an unbiased estimate of feature importance that
    accounts for feature interactions and is not affected by cardinality
    bias.  Unlike MDI, MDA evaluates on held-out data and directly measures
    how much predictive power is lost when a feature is shuffled.

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
        Number of permutation repeats per feature.  More repeats yield
        more stable estimates but increase runtime linearly.

    Returns
    -------
    pd.Series
        Mean importance values indexed by feature name, sorted
        descending.  Positive values indicate features whose permutation
        hurts the model score; negative values suggest noise features.

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(300, 4)
    >>> y = (X[:, 0] + 0.3 * X[:, 2] > 0).astype(int)
    >>> rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    >>> imp = feature_importance_mda(rf, X, y, ['mom', 'vol', 'size', 'val'])
    >>> imp.iloc[0] > 0  # top feature has positive importance
    True

    Notes
    -----
    MDA is model-agnostic and works with any estimator that exposes a
    ``score`` method.  Correlated features share importance: permuting one
    leaves its correlated partner to compensate, so both appear less
    important than they truly are.

    References
    ----------
    - Breiman (2001), "Random Forests"
    - Lopez de Prado (2018), "Advances in Financial Machine Learning", Ch. 8

    See Also
    --------
    feature_importance_mdi : Faster but biased impurity-based importance.
    wraquant.ml.pipeline.feature_importance_shap : SHAP-based importance.
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

    Use sequential feature selection when you want to find a compact
    subset of features that maximises predictive performance.  Forward
    selection greedily adds the best feature at each step; backward
    selection starts with all features and removes the least useful.

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
        Selection direction.  Forward is faster when ``n_features`` is
        small relative to total features; backward is faster when you
        want to drop only a few.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    list[str | int]
        Selected feature names (if *X* is a DataFrame) or column
        indices.

    Example
    -------
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np, pandas as pd
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.randn(200, 6),
    ...                   columns=['f1','f2','f3','f4','f5','f6'])
    >>> y = X['f1'] * 2 + X['f3'] + np.random.randn(200) * 0.1
    >>> selected = sequential_feature_selection(Ridge(), X, y, n_features=2)
    >>> len(selected)
    2

    See Also
    --------
    feature_importance_mdi : Impurity-based ranking (faster, less rigorous).
    feature_importance_mda : Permutation-based ranking.
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
