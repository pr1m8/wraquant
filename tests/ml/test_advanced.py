"""Tests for wraquant.ml.advanced — Advanced sklearn-based models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ml.advanced import (
    gaussian_process_regression,
    gradient_boost_forecast,
    isolation_forest_anomaly,
    pca_factor_model,
    random_forest_importance,
    svm_classifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def classification_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Synthetic binary classification data with clear separation."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X[:150], y[:150], X[150:], y[150:]


@pytest.fixture()
def regression_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Synthetic regression data."""
    np.random.seed(21)
    n = 300
    X = np.random.randn(n, 5)
    y = 2.0 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + np.random.randn(n) * 0.3
    return X[:250], y[:250], X[250:], y[250:]


@pytest.fixture()
def return_matrix() -> pd.DataFrame:
    """Synthetic return matrix for PCA."""
    np.random.seed(77)
    n = 252
    factor = np.random.randn(n) * 0.01
    data = {}
    for i in range(15):
        loading = np.random.uniform(0.5, 1.5)
        data[f"asset_{i}"] = factor * loading + np.random.randn(n) * 0.002
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# SVM classifier
# ---------------------------------------------------------------------------


class TestSVMClassifier:
    def test_output_keys(self, classification_data: tuple) -> None:
        X_tr, y_tr, X_te, y_te = classification_data
        result = svm_classifier(X_tr, y_tr, X_te, y_te, cv=3)
        expected = {
            "model",
            "predictions",
            "accuracy",
            "confusion_matrix",
            "best_params",
            "cv_score",
        }
        assert set(result.keys()) == expected

    def test_accuracy_above_chance(self, classification_data: tuple) -> None:
        X_tr, y_tr, X_te, y_te = classification_data
        result = svm_classifier(
            X_tr, y_tr, X_te, y_te, kernel="rbf", C_range=(1.0,), cv=3
        )
        assert result["accuracy"] > 0.5

    def test_linear_kernel(self, classification_data: tuple) -> None:
        X_tr, y_tr, X_te, y_te = classification_data
        result = svm_classifier(
            X_tr, y_tr, X_te, y_te, kernel="linear", C_range=(1.0,), cv=3
        )
        assert result["predictions"].shape == y_te.shape

    def test_confusion_matrix_shape(self, classification_data: tuple) -> None:
        X_tr, y_tr, X_te, y_te = classification_data
        result = svm_classifier(X_tr, y_tr, X_te, y_te, cv=3)
        n_classes = len(np.unique(y_te))
        assert result["confusion_matrix"].shape == (n_classes, n_classes)


# ---------------------------------------------------------------------------
# Random Forest importance
# ---------------------------------------------------------------------------


class TestRandomForestImportance:
    def test_importance_sorted(self, classification_data: tuple) -> None:
        X_tr, y_tr, _, _ = classification_data
        result = random_forest_importance(X_tr, y_tr)
        imp = result["importance"]
        assert isinstance(imp, pd.Series)
        # Should be sorted descending
        assert list(imp.values) == sorted(imp.values, reverse=True)

    def test_oob_score_exists(self, classification_data: tuple) -> None:
        X_tr, y_tr, _, _ = classification_data
        result = random_forest_importance(X_tr, y_tr)
        assert result["oob_score"] is not None
        assert 0.0 <= result["oob_score"] <= 1.0

    def test_regression_task(self, regression_data: tuple) -> None:
        X_tr, y_tr, _, _ = regression_data
        result = random_forest_importance(X_tr, y_tr, task="regression")
        assert len(result["importance"]) == X_tr.shape[1]

    def test_custom_feature_names(self, classification_data: tuple) -> None:
        X_tr, y_tr, _, _ = classification_data
        names = ["feat_a", "feat_b", "feat_c", "feat_d"]
        result = random_forest_importance(X_tr, y_tr, feature_names=names)
        assert set(result["importance"].index) == set(names)


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------


class TestGradientBoostForecast:
    def test_regression_output(self, regression_data: tuple) -> None:
        X_tr, y_tr, X_te, y_te = regression_data
        result = gradient_boost_forecast(
            X_tr, y_tr, X_te, y_te, task="regression", n_estimators=50, cv=3
        )
        assert "predictions" in result
        assert "feature_importance" in result
        assert result["test_score"] is not None

    def test_classification_output(self, classification_data: tuple) -> None:
        X_tr, y_tr, X_te, y_te = classification_data
        result = gradient_boost_forecast(
            X_tr, y_tr, X_te, y_te, task="classification", n_estimators=50, cv=3
        )
        assert result["test_score"] > 0.5

    def test_cv_scores_shape(self, regression_data: tuple) -> None:
        X_tr, y_tr, X_te, _ = regression_data
        result = gradient_boost_forecast(
            X_tr, y_tr, X_te, task="regression", n_estimators=30, cv=4
        )
        assert len(result["cv_scores"]) == 4

    def test_feature_importance_sorted(self, regression_data: tuple) -> None:
        X_tr, y_tr, X_te, _ = regression_data
        result = gradient_boost_forecast(
            X_tr, y_tr, X_te, task="regression", n_estimators=30, cv=3
        )
        imp = result["feature_importance"]
        assert list(imp.values) == sorted(imp.values, reverse=True)


# ---------------------------------------------------------------------------
# Gaussian Process
# ---------------------------------------------------------------------------


class TestGaussianProcessRegression:
    def test_output_shape(self) -> None:
        np.random.seed(10)
        X_tr = np.linspace(0, 5, 30).reshape(-1, 1)
        y_tr = np.sin(X_tr).ravel() + np.random.randn(30) * 0.1
        X_te = np.linspace(0, 5, 10).reshape(-1, 1)
        result = gaussian_process_regression(X_tr, y_tr, X_te)
        assert result["predictions"].shape == (10,)
        assert result["std"].shape == (10,)
        assert result["confidence_lower"].shape == (10,)
        assert result["confidence_upper"].shape == (10,)

    def test_confidence_interval_order(self) -> None:
        np.random.seed(10)
        X_tr = np.linspace(0, 5, 30).reshape(-1, 1)
        y_tr = np.sin(X_tr).ravel()
        X_te = np.linspace(0, 5, 10).reshape(-1, 1)
        result = gaussian_process_regression(X_tr, y_tr, X_te)
        assert np.all(result["confidence_lower"] <= result["predictions"])
        assert np.all(result["predictions"] <= result["confidence_upper"])

    def test_std_non_negative(self) -> None:
        np.random.seed(10)
        X_tr = np.linspace(0, 3, 20).reshape(-1, 1)
        y_tr = X_tr.ravel() ** 2
        X_te = np.linspace(0, 3, 8).reshape(-1, 1)
        result = gaussian_process_regression(X_tr, y_tr, X_te)
        assert np.all(result["std"] >= 0)

    def test_matern_kernel(self) -> None:
        np.random.seed(10)
        X_tr = np.linspace(0, 3, 25).reshape(-1, 1)
        y_tr = np.sin(X_tr).ravel()
        X_te = np.linspace(0, 3, 5).reshape(-1, 1)
        result = gaussian_process_regression(X_tr, y_tr, X_te, kernel="matern")
        assert result["predictions"].shape == (5,)

    def test_invalid_kernel_raises(self) -> None:
        X_tr = np.array([[1], [2], [3]])
        y_tr = np.array([1, 2, 3])
        X_te = np.array([[1.5]])
        with pytest.raises(ValueError, match="Unknown kernel"):
            gaussian_process_regression(X_tr, y_tr, X_te, kernel="invalid")


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------


class TestIsolationForestAnomaly:
    def test_output_keys(self) -> None:
        np.random.seed(42)
        rets = np.random.randn(300) * 0.01
        result = isolation_forest_anomaly(rets)
        expected = {
            "anomaly_labels",
            "anomaly_scores",
            "anomaly_mask",
            "n_anomalies",
            "model",
        }
        assert set(result.keys()) == expected

    def test_detects_injected_anomaly(self) -> None:
        np.random.seed(42)
        rets = np.random.randn(500) * 0.01
        # Inject extreme outliers
        rets[100] = 0.20
        rets[200] = -0.25
        result = isolation_forest_anomaly(rets, contamination=0.02)
        # The injected anomalies should be flagged
        assert result["anomaly_mask"][100]
        assert result["anomaly_mask"][200]

    def test_n_anomalies_matches_mask(self) -> None:
        np.random.seed(42)
        rets = np.random.randn(200) * 0.01
        result = isolation_forest_anomaly(rets, contamination=0.05)
        assert result["n_anomalies"] == int(result["anomaly_mask"].sum())

    def test_accepts_dataframe(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame(
            {"ret": np.random.randn(200) * 0.01, "vol": np.abs(np.random.randn(200))}
        )
        result = isolation_forest_anomaly(df, contamination=0.05)
        assert len(result["anomaly_labels"]) == 200


# ---------------------------------------------------------------------------
# PCA factor model
# ---------------------------------------------------------------------------


class TestPCAFactorModel:
    def test_output_shapes(self, return_matrix: pd.DataFrame) -> None:
        result = pca_factor_model(return_matrix, n_components=3)
        assert result["loadings"].shape == (15, 3)
        assert result["factor_returns"].shape == (252, 3)
        assert result["n_components"] == 3

    def test_explained_variance_sums_to_less_than_1(
        self, return_matrix: pd.DataFrame
    ) -> None:
        result = pca_factor_model(return_matrix, n_components=3)
        assert result["cumulative_variance"][-1] <= 1.0 + 1e-10

    def test_auto_components(self, return_matrix: pd.DataFrame) -> None:
        result = pca_factor_model(
            return_matrix, explained_variance_threshold=0.80
        )
        assert result["n_components"] >= 1
        assert result["cumulative_variance"][-1] >= 0.80 - 1e-10

    def test_loadings_have_asset_names(self, return_matrix: pd.DataFrame) -> None:
        result = pca_factor_model(return_matrix, n_components=2)
        assert list(result["loadings"].index) == list(return_matrix.columns)

    def test_factor_columns_named(self, return_matrix: pd.DataFrame) -> None:
        result = pca_factor_model(return_matrix, n_components=3)
        assert list(result["factor_returns"].columns) == ["PC1", "PC2", "PC3"]
