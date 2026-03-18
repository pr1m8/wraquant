"""Advanced scikit-learn models for quantitative finance.

Provides production-ready wrappers around SVM, Random Forest, Gradient
Boosting, Gaussian Process, Isolation Forest, and PCA -- all with
finance-specific defaults, comprehensive docstrings, and clean return
interfaces.

All functions guard sklearn imports behind ``@requires_extra('ml')`` so the
rest of wraquant works without scikit-learn installed.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "svm_classifier",
    "random_forest_importance",
    "gradient_boost_forecast",
    "gaussian_process_regression",
    "isolation_forest_anomaly",
    "pca_factor_model",
]


# ---------------------------------------------------------------------------
# SVM classifier
# ---------------------------------------------------------------------------


@requires_extra("ml")
def svm_classifier(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    kernel: Literal["rbf", "linear", "poly"] = "rbf",
    C_range: Sequence[float] = (0.1, 1.0, 10.0),
    gamma_range: Sequence[float | str] = ("scale", 0.01, 0.1),
    cv: int = 5,
) -> dict[str, Any]:
    """Train an SVM classifier for market regime classification.

    Support Vector Machines find the maximum-margin hyperplane separating
    classes. With the RBF kernel, SVMs can capture non-linear decision
    boundaries in feature space, making them effective for classifying
    market regimes (bull/bear/neutral) from derived features like
    volatility, momentum, and volume profiles.

    When to use:
        Use SVM when you have a moderate number of features (5-100),
        moderate dataset size (500-50k), and need robust classification
        with good generalisation. SVMs handle high-dimensional spaces well
        and are resistant to overfitting when C is properly tuned.

    Mathematical background:
        SVM solves:
            min_{w,b} (1/2) ||w||^2 + C * sum_i max(0, 1 - y_i(w.x_i + b))

        The RBF kernel maps inputs to infinite-dimensional space:
            K(x, x') = exp(-gamma * ||x - x'||^2)

        Grid search over C (regularisation) and gamma (kernel width)
        selects the best hyperparameters via cross-validation.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training labels (e.g., 1 = bull, 0 = neutral, -1 = bear).
    X_test : pd.DataFrame or np.ndarray
        Test feature matrix.
    y_test : pd.Series or np.ndarray
        Test labels.
    kernel : {'rbf', 'linear', 'poly'}
        SVM kernel function.
    C_range : Sequence[float]
        Regularisation parameter values to search.
    gamma_range : Sequence[float | str]
        Kernel coefficient values to search (ignored for linear kernel).
    cv : int
        Cross-validation folds for grid search.

    Returns
    -------
    dict
        ``model``: fitted SVC,
        ``predictions``: np.ndarray of test predictions,
        ``accuracy``: float,
        ``confusion_matrix``: np.ndarray,
        ``best_params``: dict of best C and gamma,
        ``cv_score``: float (mean CV accuracy).

    Example
    -------
    >>> import numpy as np
    >>> X = np.random.randn(200, 5)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> result = svm_classifier(X[:150], y[:150], X[150:], y[150:])
    >>> result["accuracy"] > 0.5
    True

    Caveats
    -------
    - Scale features before training (StandardScaler recommended).
    - SVMs are O(n^2) in memory and O(n^3) in time -- avoid for n > 100k.
    - For imbalanced classes, set ``class_weight='balanced'`` on the SVC.

    References
    ----------
    - Cortes & Vapnik (1995), "Support-Vector Networks"
    """
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    X_tr = np.asarray(X_train)
    y_tr = np.asarray(y_train)
    X_te = np.asarray(X_test)
    y_te = np.asarray(y_test)

    param_grid: dict[str, list[Any]] = {"C": list(C_range)}
    if kernel != "linear":
        param_grid["gamma"] = list(gamma_range)

    svc = SVC(kernel=kernel, class_weight="balanced")
    grid = GridSearchCV(svc, param_grid, cv=cv, scoring="accuracy", n_jobs=1)
    grid.fit(X_tr, y_tr)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_te)
    acc = float(accuracy_score(y_te, preds))
    cm = confusion_matrix(y_te, preds)

    return {
        "model": best_model,
        "predictions": preds,
        "accuracy": acc,
        "confusion_matrix": cm,
        "best_params": grid.best_params_,
        "cv_score": float(grid.best_score_),
    }


# ---------------------------------------------------------------------------
# Random Forest feature importance
# ---------------------------------------------------------------------------


@requires_extra("ml")
def random_forest_importance(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    feature_names: Sequence[str] | None = None,
    n_estimators: int = 100,
    max_depth: int | None = 5,
    random_state: int = 42,
    task: Literal["classification", "regression"] = "classification",
) -> dict[str, Any]:
    """Rank features by importance using a Random Forest.

    Random Forests aggregate many decorrelated decision trees and measure
    each feature's contribution to reducing impurity (Gini for
    classification, variance for regression). This produces a natural
    feature ranking useful for selecting the most predictive signals from
    a large universe of technical indicators, fundamental factors, or
    alternative data features.

    When to use:
        Use as a first-pass feature selector when you have many candidate
        features (>20) and want to identify which ones carry signal. Fast,
        non-parametric, and handles mixed feature types.

    Mathematical background:
        Mean Decrease Impurity (MDI) for feature j:
            Imp(j) = sum_{t in T_j} p(t) * Delta_i(t)

        where T_j is the set of tree nodes splitting on feature j, p(t) is
        the fraction of samples reaching node t, and Delta_i(t) is the
        impurity decrease. MDI is averaged over all trees in the forest.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector.
    feature_names : Sequence[str] or None
        Feature names. If None and X is a DataFrame, column names are used.
    n_estimators : int
        Number of trees.
    max_depth : int or None
        Maximum tree depth (None for unlimited).
    random_state : int
        Random seed for reproducibility.
    task : {'classification', 'regression'}
        Type of prediction task.

    Returns
    -------
    dict
        ``importance``: pd.Series of feature importances sorted descending,
        ``model``: fitted RandomForest estimator,
        ``oob_score``: float (out-of-bag score if available, else None).

    Example
    -------
    >>> import numpy as np
    >>> X = np.random.randn(300, 10)
    >>> y = (X[:, 0] + 0.5 * X[:, 3] > 0).astype(int)
    >>> result = random_forest_importance(X, y)
    >>> result["importance"].index[0]  # top feature is likely 0
    0

    Caveats
    -------
    - MDI importance is biased toward high-cardinality features; consider
      permutation importance (``feature_importance_mda``) as a complement.
    - Correlated features share importance, causing both to appear weaker.

    References
    ----------
    - Breiman (2001), "Random Forests"
    - Lopez de Prado (2018), "Advances in Financial Machine Learning", Ch.8
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = list(range(X_arr.shape[1]))

    if task == "classification":
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            oob_score=True,
            n_jobs=1,
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            oob_score=True,
            n_jobs=1,
        )

    rf.fit(X_arr, y_arr)

    importance = pd.Series(
        rf.feature_importances_,
        index=feature_names,
        name="importance",
    ).sort_values(ascending=False)

    oob = float(rf.oob_score_) if hasattr(rf, "oob_score_") else None

    return {
        "importance": importance,
        "model": rf,
        "oob_score": oob,
    }


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------


@requires_extra("ml")
def gradient_boost_forecast(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray | None = None,
    task: Literal["classification", "regression"] = "regression",
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    cv: int = 5,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Gradient boosting for forecasting or classification.

    Gradient Boosting sequentially fits weak learners (shallow trees) to
    the residuals of the ensemble, greedily minimising a loss function. It
    is the workhorse of tabular ML in quant finance -- used for return
    prediction, alpha factor construction, default prediction, and more.

    When to use:
        Use gradient boosting as your default tabular model. It handles
        non-linearities, feature interactions, and missing values naturally.
        Preferred over linear models when you have >500 samples and >5
        features.

    Mathematical background:
        At each stage m, the model adds a tree h_m that minimises:
            F_m(x) = F_{m-1}(x) + nu * h_m(x)

        where h_m fits the negative gradient of the loss:
            h_m = argmin_h sum_i L(y_i, F_{m-1}(x_i) + h(x_i))

        For regression with squared loss, h_m fits the residuals.
        For classification with log-loss, h_m fits the log-odds residuals.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training target.
    X_test : pd.DataFrame or np.ndarray
        Test feature matrix.
    y_test : pd.Series or np.ndarray or None
        Test target (if provided, test metrics are computed).
    task : {'regression', 'classification'}
        Prediction task.
    n_estimators : int
        Number of boosting stages.
    max_depth : int
        Maximum depth of individual trees.
    learning_rate : float
        Shrinkage applied to each tree's contribution.
    subsample : float
        Fraction of training samples used per tree (stochastic boosting).
    cv : int
        Cross-validation folds for reporting training CV score.
    feature_names : Sequence[str] or None
        Feature names for importance ranking.

    Returns
    -------
    dict
        ``model``: fitted GradientBoosting estimator,
        ``predictions``: np.ndarray of test predictions,
        ``feature_importance``: pd.Series (sorted descending),
        ``cv_scores``: np.ndarray of cross-validation scores,
        ``test_score``: float or None (R^2 for regression, accuracy for
        classification).

    Example
    -------
    >>> import numpy as np
    >>> X = np.random.randn(300, 5)
    >>> y = X[:, 0] * 2 + X[:, 1] + np.random.randn(300) * 0.5
    >>> result = gradient_boost_forecast(X[:250], y[:250], X[250:], y[250:])
    >>> result["test_score"] > 0
    True

    Caveats
    -------
    - Overfits if n_estimators is too large; use early stopping or CV.
    - Sensitive to learning_rate / n_estimators trade-off.
    - For >100k samples, consider XGBoost/LightGBM for speed.

    References
    ----------
    - Friedman (2001), "Greedy Function Approximation: A Gradient Boosting
      Machine"
    """
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.model_selection import cross_val_score

    X_tr = np.asarray(X_train)
    y_tr = np.asarray(y_train)
    X_te = np.asarray(X_test)

    if feature_names is None:
        if isinstance(X_train, pd.DataFrame):
            feature_names = list(X_train.columns)
        else:
            feature_names = list(range(X_tr.shape[1]))

    if task == "regression":
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
        )
        scoring = "r2"
    else:
        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
        )
        scoring = "accuracy"

    gb.fit(X_tr, y_tr)
    preds = gb.predict(X_te)

    cv_scores = cross_val_score(gb, X_tr, y_tr, cv=cv, scoring=scoring)

    importance = pd.Series(
        gb.feature_importances_,
        index=feature_names,
        name="importance",
    ).sort_values(ascending=False)

    test_score: float | None = None
    if y_test is not None:
        y_te = np.asarray(y_test)
        if task == "regression":
            ss_res = np.sum((y_te - preds) ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            test_score = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            test_score = float(np.mean(preds == y_te))

    return {
        "model": gb,
        "predictions": preds,
        "feature_importance": importance,
        "cv_scores": cv_scores,
        "test_score": test_score,
    }


# ---------------------------------------------------------------------------
# Gaussian Process regression
# ---------------------------------------------------------------------------


@requires_extra("ml")
def gaussian_process_regression(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    kernel: str = "rbf",
    alpha: float = 1e-2,
    n_restarts: int = 5,
) -> dict[str, Any]:
    """Gaussian Process regression with uncertainty quantification.

    Gaussian Processes (GPs) define a distribution over functions and
    provide both point predictions and calibrated confidence intervals.
    In finance, GPs are used for smooth yield-curve fitting,
    volatility-surface interpolation, and any setting where uncertainty
    matters as much as the prediction.

    When to use:
        Use GP when you need uncertainty estimates (e.g., confidence bands
        on a yield curve) and have a small-to-moderate dataset (<5000
        observations). The cubic complexity makes GPs impractical for
        large datasets without approximations.

    Mathematical background:
        A GP assumes f(x) ~ GP(m(x), k(x, x')), where:
            m(x) is the mean function (usually 0)
            k(x, x') is the kernel (covariance function)

        Posterior predictive at test point x*:
            mu* = k(x*, X) [K + sigma^2 I]^{-1} y
            sigma*^2 = k(x*, x*) - k(x*, X) [K + sigma^2 I]^{-1} k(X, x*)

        where K_{ij} = k(x_i, x_j) and sigma^2 is the noise variance.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    kernel : str
        Kernel type: ``'rbf'``, ``'matern'``, or ``'rational_quadratic'``.
    alpha : float
        Noise level (regularisation diagonal added to the kernel matrix).
    n_restarts : int
        Number of optimiser restarts for kernel hyperparameters.

    Returns
    -------
    dict
        ``predictions``: np.ndarray of mean predictions,
        ``std``: np.ndarray of predictive standard deviations,
        ``confidence_lower``: np.ndarray (mean - 1.96 * std),
        ``confidence_upper``: np.ndarray (mean + 1.96 * std),
        ``model``: fitted GaussianProcessRegressor.

    Example
    -------
    >>> import numpy as np
    >>> X_train = np.linspace(0, 10, 50).reshape(-1, 1)
    >>> y_train = np.sin(X_train).ravel() + np.random.randn(50) * 0.1
    >>> X_test = np.linspace(0, 10, 20).reshape(-1, 1)
    >>> result = gaussian_process_regression(X_train, y_train, X_test)
    >>> result["predictions"].shape
    (20,)
    >>> result["std"].shape
    (20,)

    Caveats
    -------
    - Complexity is O(n^3) for training and O(n^2) per prediction.
    - For large datasets, use sparse GP approximations (not included here).
    - Kernel choice strongly affects results; try multiple kernels.

    References
    ----------
    - Rasmussen & Williams (2006), "Gaussian Processes for Machine Learning"
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        Matern,
        RationalQuadratic,
        RBF,
    )

    X_tr = np.asarray(X_train)
    y_tr = np.asarray(y_train).ravel()
    X_te = np.asarray(X_test)

    kernel_map = {
        "rbf": ConstantKernel(1.0) * RBF(length_scale=1.0),
        "matern": ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
        "rational_quadratic": ConstantKernel(1.0) * RationalQuadratic(
            length_scale=1.0, alpha=1.0
        ),
    }

    kern = kernel_map.get(kernel)
    if kern is None:
        raise ValueError(
            f"Unknown kernel '{kernel}'; choose from 'rbf', 'matern', "
            f"'rational_quadratic'."
        )

    gp = GaussianProcessRegressor(
        kernel=kern,
        alpha=alpha,
        n_restarts_optimizer=n_restarts,
        random_state=42,
    )
    gp.fit(X_tr, y_tr)

    mean, std = gp.predict(X_te, return_std=True)

    return {
        "predictions": mean,
        "std": std,
        "confidence_lower": mean - 1.96 * std,
        "confidence_upper": mean + 1.96 * std,
        "model": gp,
    }


# ---------------------------------------------------------------------------
# Isolation Forest anomaly detection
# ---------------------------------------------------------------------------


@requires_extra("ml")
def isolation_forest_anomaly(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    """Detect anomalous days in return data using Isolation Forest.

    Isolation Forest detects anomalies by randomly partitioning data and
    measuring how quickly each observation is isolated. Anomalous points
    (outlier returns, flash crashes, liquidity events) are isolated in
    fewer splits because they sit far from the bulk of the distribution.

    When to use:
        Use for unsupervised anomaly detection in returns, volumes, or
        spreads. Works well when you do not have labelled anomalies and
        want to flag unusual market days for review. Robust to
        high-dimensional feature spaces.

    Mathematical background:
        For a sample x, the anomaly score is based on the average path
        length E[h(x)] across the isolation trees:
            s(x, n) = 2^{-E[h(x)] / c(n)}

        where c(n) is the average path length in a binary search tree of
        n samples. Score close to 1 means anomaly; close to 0.5 means
        normal.

    Parameters
    ----------
    returns : pd.Series, pd.DataFrame, or np.ndarray
        Return data. If 1-D, treated as a single feature; if 2-D, each
        column is a feature (e.g., return, volume, spread).
    contamination : float
        Expected fraction of anomalies in the dataset (0 < c < 0.5).
    n_estimators : int
        Number of isolation trees.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        ``anomaly_labels``: np.ndarray of -1 (anomaly) / 1 (normal),
        ``anomaly_scores``: np.ndarray of continuous anomaly scores
        (lower = more anomalous),
        ``anomaly_mask``: np.ndarray of bool (True for anomalies),
        ``n_anomalies``: int,
        ``model``: fitted IsolationForest.

    Example
    -------
    >>> import numpy as np
    >>> rets = np.random.randn(500) * 0.01
    >>> rets[100] = 0.15  # inject anomaly
    >>> result = isolation_forest_anomaly(rets, contamination=0.02)
    >>> result["anomaly_mask"][100]
    True

    Caveats
    -------
    - The contamination parameter is a prior; misspecification leads to
      over- or under-detection.
    - Isolation Forest assumes anomalies are both rare and different;
      clustered anomalies may be missed.
    - For time-series anomaly detection, consider adding lagged features.

    References
    ----------
    - Liu, Ting & Zhou (2008), "Isolation Forest"
    """
    from sklearn.ensemble import IsolationForest

    X = np.asarray(returns)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    )
    labels = iso.fit_predict(X)
    scores = iso.decision_function(X)

    anomaly_mask = labels == -1

    return {
        "anomaly_labels": labels,
        "anomaly_scores": scores,
        "anomaly_mask": anomaly_mask,
        "n_anomalies": int(anomaly_mask.sum()),
        "model": iso,
    }


# ---------------------------------------------------------------------------
# PCA factor model
# ---------------------------------------------------------------------------


@requires_extra("ml")
def pca_factor_model(
    returns: pd.DataFrame,
    n_components: int | None = None,
    explained_variance_threshold: float = 0.90,
) -> dict[str, Any]:
    """Build a PCA-based latent factor model from asset returns.

    Principal Component Analysis extracts orthogonal linear combinations
    of asset returns that explain the most variance. The first PC
    typically captures the market factor, the second often captures a
    value/growth or sector rotation, and so on.

    When to use:
        Use PCA factor models for dimensionality reduction in portfolio
        construction, risk decomposition, statistical arbitrage (pairs
        trading on residuals), and understanding co-movement structure.

    Mathematical background:
        Given return matrix R (T x N), PCA decomposes the covariance:
            Sigma = V Lambda V^T

        where Lambda = diag(lambda_1, ..., lambda_N) are eigenvalues and
        V are eigenvectors (loadings). Factor returns are:
            F = R @ V[:, :k]    (T x k)

        The fraction of variance explained by the first k components:
            sum(lambda_1..k) / sum(lambda_1..N)

    Parameters
    ----------
    returns : pd.DataFrame
        T x N return matrix (rows = observations, columns = assets).
    n_components : int or None
        Number of principal components. If None, selects enough to explain
        ``explained_variance_threshold`` of total variance.
    explained_variance_threshold : float
        Minimum cumulative explained variance ratio when ``n_components``
        is None.

    Returns
    -------
    dict
        ``loadings``: pd.DataFrame of shape ``(N, n_components)`` --
        asset loadings on each factor,
        ``factor_returns``: pd.DataFrame of shape ``(T, n_components)`` --
        time series of factor returns,
        ``explained_variance_ratio``: np.ndarray of per-component variance
        ratios,
        ``cumulative_variance``: np.ndarray of cumulative variance ratios,
        ``n_components``: int,
        ``model``: fitted PCA object.

    Example
    -------
    >>> import numpy as np, pandas as pd
    >>> returns = pd.DataFrame(np.random.randn(252, 20) * 0.01)
    >>> result = pca_factor_model(returns, n_components=3)
    >>> result["factor_returns"].shape
    (252, 3)

    Caveats
    -------
    - PCA is linear; for non-linear dimensionality reduction, use the VAE
      in ``wraquant.ml.deep.autoencoder_features``.
    - Eigenvalues from small samples are noisy; use Random Matrix Theory
      denoising (``wraquant.ml.preprocessing.denoised_correlation``) first.
    - Components are not guaranteed to have economic meaning.

    References
    ----------
    - Jolliffe (2002), "Principal Component Analysis"
    - Avellaneda & Lee (2010), "Statistical arbitrage in the US equities
      market"
    """
    from sklearn.decomposition import PCA

    R = np.asarray(returns, dtype=np.float64)

    if n_components is None:
        # Fit full PCA to find the right number of components
        pca_full = PCA()
        pca_full.fit(R)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, explained_variance_threshold) + 1)
        n_components = min(n_components, R.shape[1])

    pca = PCA(n_components=n_components)
    factor_returns = pca.fit_transform(R)

    # Loadings: eigenvectors scaled by sqrt(eigenvalue)
    loadings = pca.components_.T  # (N, n_components)

    asset_names = returns.columns if isinstance(returns, pd.DataFrame) else None
    factor_names = [f"PC{i + 1}" for i in range(n_components)]

    loadings_df = pd.DataFrame(
        loadings,
        index=asset_names if asset_names is not None else range(R.shape[1]),
        columns=factor_names,
    )

    index = returns.index if isinstance(returns, pd.DataFrame) else None
    factor_df = pd.DataFrame(
        factor_returns,
        index=index,
        columns=factor_names,
    )

    return {
        "loadings": loadings_df,
        "factor_returns": factor_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "n_components": n_components,
        "model": pca,
    }
