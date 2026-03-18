"""Tests for wraquant.ml.evaluation."""

from __future__ import annotations

import numpy as np

from wraquant.ml.evaluation import (
    backtest_predictions,
    classification_metrics,
    financial_metrics,
    learning_curve,
)

# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0])
        result = classification_metrics(y_true, y_pred)
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_with_probabilities(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2])
        result = classification_metrics(y_true, y_pred, y_prob=y_prob)
        assert "log_loss" in result
        assert "auc" in result
        assert result["auc"] > 0.9

    def test_all_wrong(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        result = classification_metrics(y_true, y_pred)
        assert result["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# financial_metrics
# ---------------------------------------------------------------------------


class TestFinancialMetrics:
    def test_expected_keys(self) -> None:
        y_true = np.array([1, -1, 1, 1, -1])
        y_pred = np.array([1, -1, 1, -1, 1])
        returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
        result = financial_metrics(y_true, y_pred, returns)
        expected_keys = {"strategy_return", "sharpe", "hit_rate", "profit_factor"}
        assert set(result.keys()) == expected_keys

    def test_perfect_signals(self) -> None:
        y_true = np.array([1, -1, 1])
        y_pred = np.array([1, -1, 1])
        returns = np.array([0.01, -0.02, 0.015])
        result = financial_metrics(y_true, y_pred, returns)
        # Strategy return should be all positive
        assert result["strategy_return"] > 0
        assert result["hit_rate"] == 1.0


# ---------------------------------------------------------------------------
# learning_curve (requires sklearn)
# ---------------------------------------------------------------------------


class TestLearningCurve:
    def test_output_structure(self) -> None:
        from sklearn.linear_model import LogisticRegression

        np.random.seed(1)
        X = np.random.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(max_iter=200)
        result = learning_curve(model, X, y, train_sizes=[0.3, 0.6, 0.9], cv=3)
        assert "train_sizes" in result
        assert "train_scores" in result
        assert "test_scores" in result
        assert len(result["train_sizes"]) == 3


# ---------------------------------------------------------------------------
# backtest_predictions
# ---------------------------------------------------------------------------


class TestBacktestPredictions:
    def test_output_structure(self) -> None:
        preds = np.array([1, 1, -1, 0, 1, -1, 1, 0, -1, 1])
        rets = np.random.randn(10) * 0.01
        result = backtest_predictions(preds, rets, cost_bps=10)
        expected_keys = {
            "gross_returns",
            "net_returns",
            "cumulative_return",
            "sharpe",
            "max_drawdown",
            "turnover",
        }
        assert set(result.keys()) == expected_keys

    def test_zero_cost(self) -> None:
        preds = np.array([1.0, 1.0, 1.0])
        rets = np.array([0.01, 0.02, 0.03])
        result = backtest_predictions(preds, rets, cost_bps=0)
        np.testing.assert_allclose(
            result["gross_returns"], result["net_returns"], atol=1e-12
        )

    def test_max_drawdown_non_positive(self) -> None:
        preds = np.array([1, 1, 1, 1, 1])
        rets = np.array([0.01, -0.05, -0.03, 0.02, 0.01])
        result = backtest_predictions(preds, rets, cost_bps=0)
        assert result["max_drawdown"] <= 0
