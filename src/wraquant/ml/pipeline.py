"""Financial ML pipeline utilities.

Provides chronology-aware pipeline wrappers, walk-forward backtesting with
PnL tracking, and SHAP-based feature importance -- all designed to prevent
data leakage that is rampant in naive ML-for-finance workflows.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "FinancialPipeline",
    "walk_forward_backtest",
    "feature_importance_shap",
]


# ---------------------------------------------------------------------------
# FinancialPipeline
# ---------------------------------------------------------------------------


class FinancialPipeline:
    """Sklearn Pipeline wrapper that enforces chronological splitting.

    Standard sklearn ``Pipeline`` + ``cross_val_score`` uses random K-Fold
    which leaks future information into the training set.
    ``FinancialPipeline`` wraps an sklearn ``Pipeline`` and replaces all
    cross-validation with purged K-fold that respects time ordering and
    applies an embargo window to prevent information leakage through
    overlapping labels.

    Parameters
    ----------
    steps : list[tuple[str, estimator]]
        List of ``(name, transform)`` tuples defining the pipeline,
        identical to the ``steps`` parameter of ``sklearn.pipeline.Pipeline``.
    n_splits : int
        Number of folds for purged K-fold cross-validation.
    embargo_pct : float
        Fraction of total samples to embargo after each test fold,
        preventing label leakage from overlapping targets.

    Example
    -------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> X = np.random.randn(500, 5)
    >>> y = X @ np.array([1, 0.5, 0, 0, 0]) + np.random.randn(500) * 0.1
    >>> pipe = FinancialPipeline(
    ...     steps=[('scaler', StandardScaler()), ('ridge', Ridge())],
    ...     n_splits=5,
    ... )
    >>> result = pipe.fit_evaluate(X, y)
    >>> len(result['fold_scores']) == 5
    True

    References
    ----------
    - Lopez de Prado (2018), "Advances in Financial Machine Learning", Ch. 7
    """

    def __init__(
        self,
        steps: list[tuple[str, Any]],
        n_splits: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        self.steps = steps
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self._pipeline: Any = None

    def _build_pipeline(self) -> Any:
        """Build the underlying sklearn Pipeline."""
        from sklearn.pipeline import Pipeline

        return Pipeline(self.steps)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> FinancialPipeline:
        """Fit the pipeline on the full dataset.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.Series or np.ndarray
            Target vector.

        Returns
        -------
        FinancialPipeline
            Self, for method chaining.
        """
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate predictions using the fitted pipeline.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline has not been fitted. Call fit() first.")
        return np.asarray(self._pipeline.predict(np.asarray(X)))

    def fit_evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Fit with purged K-fold cross-validation and return results.

        Uses purged K-fold splitting to evaluate the pipeline without data
        leakage. After cross-validation, fits the pipeline on the full
        dataset.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.Series or np.ndarray
            Target vector.

        Returns
        -------
        dict
            ``fold_scores``: list of per-fold R-squared scores,
            ``mean_score``: float mean of fold scores,
            ``std_score``: float std of fold scores,
            ``pipeline``: the fitted sklearn Pipeline.
        """
        from sklearn.base import clone
        from sklearn.metrics import r2_score

        from wraquant.ml.preprocessing import purged_kfold

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        fold_scores: list[float] = []
        for train_idx, test_idx in purged_kfold(
            X_arr, y_arr, n_splits=self.n_splits, embargo_pct=self.embargo_pct
        ):
            pipe = clone(self._build_pipeline())
            pipe.fit(X_arr[train_idx], y_arr[train_idx])
            preds = pipe.predict(X_arr[test_idx])
            score = float(r2_score(y_arr[test_idx], preds))
            fold_scores.append(score)

        # Fit on full data
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X_arr, y_arr)

        return {
            "fold_scores": fold_scores,
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "pipeline": self._pipeline,
        }


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------


def walk_forward_backtest(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    train_size: int = 252,
    test_size: int = 21,
    step_size: int = 21,
    expanding: bool = True,
) -> dict[str, Any]:
    """Full walk-forward ML backtest with PnL tracking.

    Walk-forward validation is the gold standard for evaluating ML models in
    finance because it mirrors real trading: train on historical data,
    predict the next period, observe actual outcome, then advance.

    Why walk-forward instead of standard cross-validation?
        Standard K-Fold CV randomly shuffles observations, allowing the model
        to "peek" at future data during training. In finance, this creates
        massive upward bias in performance estimates. Walk-forward enforces
        strict temporal ordering: the model only ever trains on data that
        would have been available at the time of prediction.

    The function supports both expanding windows (training set grows over
    time, using all available history) and rolling windows (fixed-size
    training window that slides forward). Expanding windows are preferred
    when you believe the data-generating process is stable; rolling windows
    are better when you expect structural breaks or regime changes.

    Parameters
    ----------
    model : estimator
        A scikit-learn-compatible estimator with ``fit`` and ``predict``.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector (typically forward returns for PnL calculation).
    train_size : int
        Number of training observations in the initial window.
    test_size : int
        Number of test observations per fold.
    step_size : int
        Number of observations to advance between folds.
    expanding : bool
        If True, the training window expands over time. If False, a
        rolling window of fixed ``train_size`` is used.

    Returns
    -------
    dict
        ``predictions``: np.ndarray of concatenated out-of-sample predictions,
        ``actuals``: np.ndarray of corresponding true values,
        ``pnl``: np.ndarray of per-period PnL (prediction * actual, assuming
        long when prediction > 0),
        ``sharpe``: float annualised Sharpe ratio of the PnL series
        (assuming 252 trading days),
        ``hit_rate``: float fraction of periods where prediction sign
        matches actual sign,
        ``equity_curve``: np.ndarray cumulative PnL.

    Example
    -------
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(600, 5)
    >>> y = X @ np.array([0.5, 0.3, 0, 0, 0]) + np.random.randn(600) * 0.5
    >>> result = walk_forward_backtest(Ridge(), X, y, train_size=200, test_size=20)
    >>> len(result['predictions']) > 0
    True
    >>> 'sharpe' in result
    True

    References
    ----------
    - Lopez de Prado (2018), "Advances in Financial Machine Learning", Ch. 12
    - Bailey et al. (2014), "The Deflated Sharpe Ratio"
    """
    from sklearn.base import clone

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n = len(X_arr)

    all_preds: list[np.ndarray] = []
    all_actuals: list[np.ndarray] = []

    start = 0
    while start + train_size + test_size <= n:
        train_start = 0 if expanding else start
        train_end = start + train_size
        test_end = min(train_end + test_size, n)

        X_train = X_arr[train_start:train_end]
        y_train = y_arr[train_start:train_end]
        X_test = X_arr[train_end:test_end]
        y_test = y_arr[train_end:test_end]

        m = clone(model)
        m.fit(X_train, y_train)
        preds = np.asarray(m.predict(X_test))

        all_preds.append(preds)
        all_actuals.append(np.asarray(y_test))

        start += step_size

    if not all_preds:
        return {
            "predictions": np.array([]),
            "actuals": np.array([]),
            "pnl": np.array([]),
            "sharpe": 0.0,
            "hit_rate": 0.0,
            "equity_curve": np.array([]),
        }

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)

    # PnL: long when prediction > 0, short when prediction < 0
    pnl = np.sign(predictions) * actuals

    # Sharpe ratio — delegate to canonical implementation
    from wraquant.risk.metrics import sharpe_ratio as _sharpe_ratio

    sharpe = _sharpe_ratio(pd.Series(pnl))

    # Hit rate: fraction of correct sign predictions
    hit_rate = float(np.mean(np.sign(predictions) == np.sign(actuals)))

    # Equity curve
    equity_curve = np.cumsum(pnl)

    return {
        "predictions": predictions,
        "actuals": actuals,
        "pnl": pnl,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "equity_curve": equity_curve,
    }


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------


@requires_extra("ml")
def feature_importance_shap(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None = None,
    max_samples: int = 500,
) -> dict[str, Any]:
    """Compute SHAP-based feature importance for any sklearn model.

    SHAP (SHapley Additive exPlanations) values provide a theoretically
    grounded decomposition of each prediction into per-feature contributions.
    Unlike impurity-based importance (MDI), SHAP values are consistent and
    account for feature interactions.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn-compatible estimator.
    X : pd.DataFrame or np.ndarray
        Feature matrix to explain (typically the test set).
    feature_names : Sequence[str] or None
        Feature names. If None and X is a DataFrame, column names are used.
    max_samples : int
        Maximum number of samples to use for computing SHAP values.
        Subsampled if X has more rows than this.

    Returns
    -------
    dict
        ``shap_values``: np.ndarray of shape ``(n_samples, n_features)``
        containing per-sample SHAP values,
        ``feature_importance``: np.ndarray of shape ``(n_features,)``
        giving mean absolute SHAP value per feature (sorted descending),
        ``feature_names``: list of feature names ordered by importance.

    Raises
    ------
    MissingDependencyError
        If shap is not installed.

    Example
    -------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(200, 5)
    >>> y = X[:, 0] * 2 + X[:, 1] + np.random.randn(200) * 0.1
    >>> model = RandomForestRegressor(n_estimators=50, random_state=42)
    >>> model.fit(X, y)
    RandomForestRegressor(n_estimators=50, random_state=42)
    >>> result = feature_importance_shap(model, X)
    >>> result["shap_values"].shape[1] == 5
    True

    References
    ----------
    - Lundberg & Lee (2017), "A Unified Approach to Interpreting Model
      Predictions"
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "shap is required for SHAP feature importance but is not installed. "
            "Install it with: pip install shap"
        )

    X_arr = np.asarray(X)

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    # Subsample if needed
    if len(X_arr) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_arr), size=max_samples, replace=False)
        X_sample = X_arr[idx]
    else:
        X_sample = X_arr

    # Use KernelExplainer for model-agnostic SHAP values
    # Use a small background dataset for efficiency
    bg_size = min(100, len(X_sample))
    background = shap.kmeans(X_sample, bg_size)
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_sample)

    shap_values = np.asarray(shap_values)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Sort by importance
    sort_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_names = [feature_names[i] for i in sort_idx]
    sorted_importance = mean_abs_shap[sort_idx]

    return {
        "shap_values": shap_values,
        "feature_importance": sorted_importance,
        "feature_names": sorted_names,
    }
