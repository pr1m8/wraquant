"""Tests for statistical analysis MCP tools.

Tests: correlation_analysis, distribution_fit, regression,
stationarity_tests, cointegration_test, robust_stats.
"""

from __future__ import annotations

import sys
import shutil
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
    """Create an AnalysisContext with multi-asset and returns data."""
    ws = tmp_path / "test_stats"
    context = AnalysisContext(str(ws))

    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    # Single-asset returns
    returns = rng.normal(0.0003, 0.015, n)
    context.store_dataset(
        "returns",
        pd.DataFrame({"returns": returns}, index=dates),
    )

    # Multi-asset returns (correlated)
    factor = rng.normal(0, 0.01, n)
    multi = pd.DataFrame({
        "AAPL": factor * 1.2 + rng.normal(0, 0.008, n),
        "MSFT": factor * 0.9 + rng.normal(0, 0.007, n),
        "GOOGL": factor * 1.1 + rng.normal(0, 0.009, n),
        "AMZN": factor * 1.4 + rng.normal(0, 0.010, n),
    }, index=dates)
    context.store_dataset("multi_returns", multi)

    # Regression data (y = 2*x1 + 0.5*x2 + noise)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2 * x1 + 0.5 * x2 + rng.normal(0, 0.3, n)
    context.store_dataset(
        "regression_data",
        pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=dates),
    )

    # Cointegrated pair: y2 = 0.8 * y1 + stationary noise
    y1 = np.cumsum(rng.normal(0, 1, n))
    y2 = 0.8 * y1 + rng.normal(0, 0.5, n)
    context.store_dataset(
        "cointegrated_pair",
        pd.DataFrame({"series_a": y1, "series_b": y2}, index=dates),
    )

    yield context
    context.close()
    if ws.exists():
        shutil.rmtree(ws)


@pytest.fixture
def stats_tools(ctx):
    """Register stats tools on mock MCP."""
    from wraquant_mcp.servers.stats import register_stats_tools

    mock = MockMCP()
    register_stats_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# Correlation analysis
# ------------------------------------------------------------------


class TestCorrelationAnalysis:
    """Test correlation_analysis tool."""

    def test_pearson_correlation(self, stats_tools, ctx):
        result = stats_tools["correlation_analysis"](
            dataset="multi_returns", method="pearson",
        )
        assert result["tool"] == "correlation_analysis"
        assert result["method"] == "pearson"
        assert "dataset_id" in result
        assert result["shape"] == [4, 4]

        # Mean correlation should be positive (common factor)
        assert result["mean_correlation"] is not None
        assert isinstance(result["mean_correlation"], float)

        # Verify stored
        corr_df = ctx.get_dataset(result["dataset_id"])
        assert corr_df.shape == (4, 4)

    def test_spearman_correlation(self, stats_tools):
        result = stats_tools["correlation_analysis"](
            dataset="multi_returns", method="spearman",
        )
        assert result["method"] == "spearman"
        assert result["shape"] == [4, 4]

    def test_kendall_correlation(self, stats_tools):
        result = stats_tools["correlation_analysis"](
            dataset="multi_returns", method="kendall",
        )
        assert result["method"] == "kendall"

    def test_shrunk_covariance(self, stats_tools):
        result = stats_tools["correlation_analysis"](
            dataset="multi_returns", shrink=True,
        )
        assert result["method"] == "ledoit_wolf"
        assert "dataset_id" in result

    def test_correlation_stores_with_parent(self, stats_tools, ctx):
        result = stats_tools["correlation_analysis"](
            dataset="multi_returns",
        )
        ds_name = result["dataset_id"]
        lineage = ctx.registry.lineage(ds_name)
        assert "multi_returns" in lineage


# ------------------------------------------------------------------
# Distribution fit
# ------------------------------------------------------------------


class TestDistributionFit:
    """Test distribution_fit tool."""

    def test_best_fit_auto(self, stats_tools):
        result = stats_tools["distribution_fit"](
            dataset="returns", column="returns",
        )
        assert result["tool"] == "distribution_fit"
        assert "jarque_bera" in result
        assert "best_fit" in result

    def test_jarque_bera_included(self, stats_tools):
        result = stats_tools["distribution_fit"](
            dataset="returns", column="returns",
        )
        jb = result["jarque_bera"]
        assert "statistic" in jb
        assert "p_value" in jb

    def test_fit_distribution_direct(self, ctx):
        """Test fit_distribution function directly."""
        from wraquant.stats.distributions import fit_distribution

        returns = ctx.get_dataset("returns")["returns"].dropna()
        result = fit_distribution(returns, dist="norm")
        assert "params" in result
        assert "ks_statistic" in result or "ks_stat" in result

    def test_jarque_bera_direct(self, ctx):
        """Test jarque_bera function directly."""
        from wraquant.stats.distributions import jarque_bera

        returns = ctx.get_dataset("returns")["returns"].dropna()
        result = jarque_bera(returns)
        assert "statistic" in result
        assert "p_value" in result


# ------------------------------------------------------------------
# Regression
# ------------------------------------------------------------------


class TestRegression:
    """Test regression tool."""

    def test_ols_regression(self, stats_tools, ctx):
        result = stats_tools["regression"](
            dataset="regression_data",
            y_column="y",
            x_columns=["x1", "x2"],
            method="ols",
        )
        assert "model_id" in result
        assert result["method"] == "ols"

        # Model should be stored
        models = ctx.list_models()
        assert any("reg" in m for m in models)

    def test_ols_auto_detect_x_columns(self, stats_tools):
        result = stats_tools["regression"](
            dataset="regression_data",
            y_column="y",
            method="ols",
        )
        assert "model_id" in result

    def test_newey_west_regression(self, stats_tools):
        result = stats_tools["regression"](
            dataset="regression_data",
            y_column="y",
            x_columns=["x1", "x2"],
            method="newey_west",
        )
        assert result["method"] == "newey_west"
        assert "model_id" in result

    def test_rolling_regression(self, stats_tools):
        result = stats_tools["regression"](
            dataset="regression_data",
            y_column="y",
            x_columns=["x1", "x2"],
            method="rolling",
            window=60,
        )
        assert result["method"] == "rolling"

    def test_invalid_method_returns_error(self, stats_tools):
        result = stats_tools["regression"](
            dataset="regression_data",
            y_column="y",
            method="invalid_method",
        )
        assert "error" in result


# ------------------------------------------------------------------
# Stationarity tests
# ------------------------------------------------------------------


class TestStationarityTests:
    """Test stationarity_tests tool."""

    def test_stationarity_on_returns(self, stats_tools):
        result = stats_tools["stationarity_tests"](
            dataset="returns", column="returns",
        )
        assert result["tool"] == "stationarity_tests"
        assert "adf" in result
        assert "kpss" in result
        assert "phillips_perron" in result
        assert "variance_ratio" in result
        assert result["observations"] > 0

    def test_adf_result_structure(self, stats_tools):
        result = stats_tools["stationarity_tests"](
            dataset="returns", column="returns",
        )
        adf = result["adf"]
        assert "statistic" in adf
        assert "p_value" in adf

    def test_kpss_result_structure(self, stats_tools):
        result = stats_tools["stationarity_tests"](
            dataset="returns", column="returns",
        )
        kpss = result["kpss"]
        assert "statistic" in kpss
        assert "p_value" in kpss

    def test_returns_should_be_stationary(self, stats_tools):
        """Normal returns should be stationary — ADF should reject."""
        result = stats_tools["stationarity_tests"](
            dataset="returns", column="returns",
        )
        adf = result["adf"]
        # For iid normal returns, ADF p-value should be very small
        assert adf["p_value"] < 0.10


# ------------------------------------------------------------------
# Cointegration test
# ------------------------------------------------------------------


class TestCointegrationTest:
    """Test cointegration_test tool."""

    def test_engle_granger_on_cointegrated_pair(self, stats_tools):
        result = stats_tools["cointegration_test"](
            dataset="cointegrated_pair",
            column_a="series_a",
            column_b="series_b",
            method="engle_granger",
        )
        assert result["tool"] == "cointegration_test"
        assert result["method"] == "engle_granger"
        assert "result" in result
        assert "hedge_ratio" in result
        assert "half_life" in result

        # Hedge ratio should be close to 0.8 (by construction)
        assert isinstance(result["hedge_ratio"], float)

    def test_hedge_ratio_close_to_true_value(self, stats_tools):
        """Hedge ratio should be near 0.8 for our synthetic pair."""
        result = stats_tools["cointegration_test"](
            dataset="cointegrated_pair",
            column_a="series_a",
            column_b="series_b",
        )
        # Allow tolerance since it's estimated
        assert 0.5 < result["hedge_ratio"] < 1.2

    def test_half_life_is_finite(self, stats_tools):
        result = stats_tools["cointegration_test"](
            dataset="cointegrated_pair",
            column_a="series_a",
            column_b="series_b",
        )
        assert np.isfinite(result["half_life"])

    def test_cointegration_returns_columns(self, stats_tools):
        result = stats_tools["cointegration_test"](
            dataset="cointegrated_pair",
            column_a="series_a",
            column_b="series_b",
        )
        assert result["columns"] == ["series_a", "series_b"]


# ------------------------------------------------------------------
# Robust stats
# ------------------------------------------------------------------


class TestRobustStats:
    """Test robust_stats tool."""

    def test_robust_stats_basic(self, stats_tools):
        result = stats_tools["robust_stats"](
            dataset="returns", column="returns",
        )
        assert result["tool"] == "robust_stats"
        assert isinstance(result["mad"], float)
        assert isinstance(result["trimmed_mean"], float)
        assert isinstance(result["huber_mean"], float)
        assert isinstance(result["winsorized_mean"], float)
        assert isinstance(result["winsorized_std"], float)

    def test_robust_stats_outlier_detection(self, stats_tools):
        result = stats_tools["robust_stats"](
            dataset="returns", column="returns",
        )
        assert "outliers" in result
