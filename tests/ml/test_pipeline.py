"""Tests for wraquant.ml.pipeline — Financial ML pipeline utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ml.pipeline import (
    FinancialPipeline,
    walk_forward_backtest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic regression data with known coefficients."""
    np.random.seed(42)
    X = np.random.randn(600, 5)
    y = X @ np.array([1.0, 0.5, 0.0, 0.0, 0.0]) + np.random.randn(600) * 0.3
    return X, y


# ---------------------------------------------------------------------------
# FinancialPipeline
# ---------------------------------------------------------------------------


class TestFinancialPipeline:
    def test_fit_evaluate_returns_correct_keys(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X, y = regression_data
        pipe = FinancialPipeline(
            steps=[("scaler", StandardScaler()), ("ridge", Ridge())],
            n_splits=3,
        )
        result = pipe.fit_evaluate(X, y)
        assert "fold_scores" in result
        assert "mean_score" in result
        assert "std_score" in result
        assert "pipeline" in result

    def test_correct_number_of_folds(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X, y = regression_data
        n_splits = 4
        pipe = FinancialPipeline(
            steps=[("scaler", StandardScaler()), ("ridge", Ridge())],
            n_splits=n_splits,
        )
        result = pipe.fit_evaluate(X, y)
        assert len(result["fold_scores"]) == n_splits

    def test_predict_after_fit(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X, y = regression_data
        pipe = FinancialPipeline(
            steps=[("scaler", StandardScaler()), ("ridge", Ridge())],
        )
        pipe.fit(X, y)
        preds = pipe.predict(X[:10])
        assert preds.shape == (10,)

    def test_predict_without_fit_raises(self) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge

        pipe = FinancialPipeline(steps=[("ridge", Ridge())])
        with pytest.raises(RuntimeError, match="not been fitted"):
            pipe.predict(np.zeros((5, 3)))

    def test_scores_are_finite(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge

        X, y = regression_data
        pipe = FinancialPipeline(steps=[("ridge", Ridge())], n_splits=3)
        result = pipe.fit_evaluate(X, y)
        assert all(np.isfinite(s) for s in result["fold_scores"])
        assert np.isfinite(result["mean_score"])


# ---------------------------------------------------------------------------
# walk_forward_backtest
# ---------------------------------------------------------------------------


class TestWalkForwardBacktest:
    def test_output_keys(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge

        X, y = regression_data
        result = walk_forward_backtest(
            Ridge(), X, y, train_size=200, test_size=20, step_size=20
        )
        assert "predictions" in result
        assert "actuals" in result
        assert "pnl" in result
        assert "sharpe" in result
        assert "hit_rate" in result
        assert "equity_curve" in result

    def test_predictions_length(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge

        X, y = regression_data
        result = walk_forward_backtest(
            Ridge(), X, y, train_size=200, test_size=20, step_size=20
        )
        assert len(result["predictions"]) == len(result["actuals"])
        assert len(result["predictions"]) > 0

    def test_pnl_shape_matches_predictions(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge

        X, y = regression_data
        result = walk_forward_backtest(
            Ridge(), X, y, train_size=200, test_size=20
        )
        assert len(result["pnl"]) == len(result["predictions"])

    def test_equity_curve_is_cumsum_of_pnl(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge

        X, y = regression_data
        result = walk_forward_backtest(
            Ridge(), X, y, train_size=200, test_size=20
        )
        np.testing.assert_allclose(
            result["equity_curve"], np.cumsum(result["pnl"])
        )

    def test_sharpe_is_finite(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge

        X, y = regression_data
        result = walk_forward_backtest(
            Ridge(), X, y, train_size=200, test_size=20
        )
        assert np.isfinite(result["sharpe"])

    def test_hit_rate_in_range(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge

        X, y = regression_data
        result = walk_forward_backtest(
            Ridge(), X, y, train_size=200, test_size=20
        )
        assert 0 <= result["hit_rate"] <= 1

    def test_rolling_window(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge

        X, y = regression_data
        result = walk_forward_backtest(
            Ridge(),
            X,
            y,
            train_size=200,
            test_size=20,
            expanding=False,
        )
        assert len(result["predictions"]) > 0

    def test_empty_when_insufficient_data(self) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import Ridge

        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        result = walk_forward_backtest(
            Ridge(), X, y, train_size=200, test_size=20
        )
        assert len(result["predictions"]) == 0


# ---------------------------------------------------------------------------
# feature_importance_shap (skip if shap not installed)
# ---------------------------------------------------------------------------


class TestFeatureImportanceSHAP:
    def test_shap_output_keys(self) -> None:
        shap = pytest.importorskip("shap")
        sklearn = pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestRegressor

        from wraquant.ml.pipeline import feature_importance_shap

        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        result = feature_importance_shap(model, X, max_samples=50)
        assert "shap_values" in result
        assert "feature_importance" in result
        assert "feature_names" in result

    def test_shap_values_shape(self) -> None:
        shap = pytest.importorskip("shap")
        sklearn = pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestRegressor

        from wraquant.ml.pipeline import feature_importance_shap

        np.random.seed(42)
        n_features = 5
        X = np.random.randn(80, n_features)
        y = X[:, 0] + np.random.randn(80) * 0.1
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        result = feature_importance_shap(model, X, max_samples=50)
        assert result["shap_values"].shape[1] == n_features
        assert len(result["feature_importance"]) == n_features
        assert len(result["feature_names"]) == n_features
