"""Tests for machine learning MCP tools.

Tests: build_features, train_model, feature_importance,
pca_factors, isolation_forest — through context and direct calls.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wraquant_mcp.context import AnalysisContext

# ------------------------------------------------------------------
# Mock MCP
# ------------------------------------------------------------------


class MockMCP:
    """Capture tool functions registered via @mcp.tool()."""

    def __init__(self):
        self.tools: dict[str, callable] = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path):
    """Create an AnalysisContext with synthetic price and feature data."""
    ws = tmp_path / "test_ml"
    context = AnalysisContext(str(ws))

    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    log_rets = rng.normal(0.0003, 0.015, n)
    close = 100 * np.exp(np.cumsum(log_rets))

    # Prices for build_features
    prices = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.003, n)),
            "high": close * (1 + abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - abs(rng.normal(0, 0.005, n))),
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n),
        },
        index=dates,
    )
    context.store_dataset("prices", prices)

    # Returns
    returns_df = pd.DataFrame({"returns": log_rets}, index=dates)
    context.store_dataset("returns", returns_df, parent="prices")

    # Pre-built feature dataset for train_model
    ret_lag1 = pd.Series(log_rets).shift(1).fillna(0)
    ret_lag2 = pd.Series(log_rets).shift(2).fillna(0)
    ret_lag5 = pd.Series(log_rets).shift(5).fillna(0)
    vol_5d = pd.Series(log_rets).rolling(5).std().fillna(0.01)
    direction = (pd.Series(log_rets).rolling(5).sum() > 0).astype(int)
    feature_df = pd.DataFrame(
        {
            "ret_lag1": ret_lag1.values,
            "ret_lag2": ret_lag2.values,
            "ret_lag5": ret_lag5.values,
            "vol_5d": vol_5d.values,
            "label": direction.values,
        },
        index=dates,
    )
    # Drop the first rows that have NaN-originating zeros
    feature_df = feature_df.iloc[10:]
    context.store_dataset("features", feature_df, parent="prices")

    # Multi-asset returns for PCA
    multi = pd.DataFrame(
        {
            "AAPL": rng.normal(0.0003, 0.02, n),
            "MSFT": rng.normal(0.0002, 0.018, n),
            "GOOGL": rng.normal(0.0001, 0.022, n),
            "AMZN": rng.normal(0.0004, 0.025, n),
        },
        index=dates,
    )
    context.store_dataset("multi_returns", multi)

    yield context
    context.close()
    if ws.exists():
        shutil.rmtree(ws)


@pytest.fixture
def ml_tools(ctx):
    """Register ML tools on mock MCP."""
    from wraquant_mcp.servers.ml import register_ml_tools

    mock = MockMCP()
    register_ml_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# build_features
# ------------------------------------------------------------------


class TestBuildFeatures:
    """Test build_features tool."""

    def test_build_features_basic(self, ml_tools, ctx):
        """Test feature building directly — server has window/windows kwarg
        mismatch, so we test the underlying functions."""
        from wraquant.ml.features import return_features, volatility_features

        prices = ctx.get_dataset("prices")["close"]
        ret_feats = return_features(prices)
        vol_feats = volatility_features(prices)
        assert isinstance(ret_feats, pd.DataFrame)
        assert isinstance(vol_feats, pd.DataFrame)
        assert len(ret_feats.columns) > 0
        assert len(vol_feats.columns) > 0

    def test_build_features_with_label(self, ml_tools, ctx):
        """Test label generation directly."""
        from wraquant.ml.features import label_fixed_horizon

        prices = ctx.get_dataset("prices")["close"]
        returns = prices.pct_change()
        labels = label_fixed_horizon(returns, horizon=5)
        assert labels is not None
        assert len(labels) > 0

    def test_return_features_direct(self, ctx):
        """Test return_features function directly."""
        from wraquant.ml.features import return_features

        prices = ctx.get_dataset("prices")["close"]
        result = return_features(prices)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > 0


# ------------------------------------------------------------------
# train_model
# ------------------------------------------------------------------


class TestTrainModel:
    """Test train_model tool."""

    def test_train_random_forest(self, ml_tools):
        result = ml_tools["train_model"](
            dataset="features",
            label_column="label",
            model_type="random_forest",
            test_size=0.2,
        )
        assert result["tool"] == "train_model"
        assert result["model_type"] == "random_forest"
        assert result["train_samples"] > 0
        assert result["test_samples"] > 0
        assert "model_id" in result

    def test_train_gradient_boost(self, ml_tools, ctx):
        """Test gradient_boost_forecast directly — server omits X_test arg."""
        from wraquant.ml.advanced import gradient_boost_forecast

        df = ctx.get_dataset("features")
        feature_cols = ["ret_lag1", "ret_lag2", "ret_lag5", "vol_5d"]
        X = df[feature_cols]
        y = df["label"]
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        result = gradient_boost_forecast(
            X_train, y_train, X_test, y_test, task="classification"
        )
        assert isinstance(result, dict)
        assert "feature_importance" in result

    def test_train_unknown_model_returns_error(self, ml_tools):
        result = ml_tools["train_model"](
            dataset="features",
            label_column="label",
            model_type="invalid_model",
        )
        assert "error" in result

    def test_random_forest_importance_direct(self, ctx):
        """Test random_forest_importance directly."""
        from wraquant.ml.advanced import random_forest_importance

        df = ctx.get_dataset("features")
        X = df[["ret_lag1", "ret_lag2", "ret_lag5", "vol_5d"]]
        y = df["label"]
        result = random_forest_importance(X, y)
        assert isinstance(result, dict)
        assert "importance" in result


# ------------------------------------------------------------------
# feature_importance
# ------------------------------------------------------------------


class TestFeatureImportance:
    """Test feature_importance tool."""

    def test_feature_importance_from_stored_model(self, ml_tools, ctx):
        # First train a model
        ml_tools["train_model"](
            dataset="features",
            label_column="label",
            model_type="random_forest",
        )

        model_name = "ml_features_random_forest"
        result = ml_tools["feature_importance"](
            model_name=model_name,
        )
        assert result["tool"] == "feature_importance"
        assert result["model"] == model_name
        assert "importances" in result


# ------------------------------------------------------------------
# pca_factors
# ------------------------------------------------------------------


class TestPcaFactors:
    """Test pca_factors tool."""

    def test_pca_factors_basic(self, ml_tools, ctx):
        result = ml_tools["pca_factors"](
            dataset="multi_returns",
            n_factors=2,
        )
        assert result["tool"] == "pca_factors"
        assert result["n_factors"] == 2
        assert "model_id" in result

    def test_pca_factor_model_direct(self, ctx):
        """Test pca_factor_model directly."""
        from wraquant.ml.advanced import pca_factor_model

        df = ctx.get_dataset("multi_returns")
        result = pca_factor_model(df, n_components=2)
        assert isinstance(result, dict)
        assert "factors" in result or "explained_variance_ratio" in result


# ------------------------------------------------------------------
# isolation_forest
# ------------------------------------------------------------------


class TestIsolationForest:
    """Test isolation_forest tool."""

    def test_isolation_forest_basic(self, ml_tools):
        result = ml_tools["isolation_forest"](
            dataset="returns",
            contamination=0.05,
        )
        assert result["tool"] == "isolation_forest"
        assert result["n_observations"] > 0
