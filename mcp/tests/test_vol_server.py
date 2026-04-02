"""Tests for volatility modeling MCP tools.

Tests: realized_volatility, ewma_volatility, forecast_volatility,
model_selection, news_impact_curve — through context and direct calls.
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
    """Create an AnalysisContext with OHLCV and returns data."""
    ws = tmp_path / "test_vol"
    context = AnalysisContext(str(ws))

    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    log_rets = rng.normal(0, 0.015, n)
    close = 100 * np.exp(np.cumsum(log_rets))

    prices = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * (1 + abs(rng.normal(0, 0.008, n))),
        "low": close * (1 - abs(rng.normal(0, 0.008, n))),
        "close": close,
        "volume": rng.integers(100_000, 1_000_000, n),
    }, index=dates)
    context.store_dataset("prices", prices)

    returns_df = pd.DataFrame({"returns": log_rets}, index=dates)
    context.store_dataset("returns", returns_df, parent="prices")

    yield context
    context.close()
    if ws.exists():
        shutil.rmtree(ws)


@pytest.fixture
def vol_tools(ctx):
    """Register vol tools on mock MCP."""
    from wraquant_mcp.servers.vol import register_vol_tools

    mock = MockMCP()
    register_vol_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# Realized volatility
# ------------------------------------------------------------------


class TestRealizedVolatility:
    """Test realized_volatility tool."""

    def test_yang_zhang(self, vol_tools, ctx):
        result = vol_tools["realized_volatility"](
            dataset="prices", method="yang_zhang", window=21,
        )
        assert result["tool"] == "realized_volatility"
        assert result["method"] == "yang_zhang"
        assert result["window"] == 21
        assert isinstance(result["current_vol"], float)
        assert result["current_vol"] > 0
        assert isinstance(result["mean_vol"], float)

        # Verify stored in context
        ds_name = result["dataset_id"]
        stored = ctx.get_dataset(ds_name)
        assert "realized_vol" in stored.columns

    def test_parkinson(self, vol_tools):
        result = vol_tools["realized_volatility"](
            dataset="prices", method="parkinson", window=21,
        )
        assert result["method"] == "parkinson"
        assert result["current_vol"] > 0

    def test_garman_klass(self, vol_tools):
        result = vol_tools["realized_volatility"](
            dataset="prices", method="garman_klass", window=21,
        )
        assert result["method"] == "garman_klass"
        assert result["current_vol"] > 0

    def test_rogers_satchell(self, vol_tools):
        result = vol_tools["realized_volatility"](
            dataset="prices", method="rogers_satchell", window=21,
        )
        assert result["method"] == "rogers_satchell"
        assert result["current_vol"] > 0

    def test_close_to_close(self, vol_tools):
        result = vol_tools["realized_volatility"](
            dataset="prices", method="close_to_close", window=21,
        )
        assert result["method"] == "close_to_close"
        assert result["current_vol"] > 0

    def test_unknown_method_returns_error(self, vol_tools):
        result = vol_tools["realized_volatility"](
            dataset="prices", method="invalid_method",
        )
        assert "error" in result

    def test_stores_with_lineage(self, vol_tools, ctx):
        result = vol_tools["realized_volatility"](
            dataset="prices", method="yang_zhang",
        )
        ds_name = result["dataset_id"]
        lineage = ctx.registry.lineage(ds_name)
        assert "prices" in lineage


# ------------------------------------------------------------------
# EWMA volatility
# ------------------------------------------------------------------


class TestEwmaVolatility:
    """Test ewma_volatility tool."""

    def test_ewma_basic(self, vol_tools, ctx):
        result = vol_tools["ewma_volatility"](
            dataset="returns", column="returns", span=30,
        )
        assert result["tool"] == "ewma_volatility"
        assert result["span"] == 30
        assert isinstance(result["current_vol"], float)
        assert result["current_vol"] > 0
        assert isinstance(result["mean_vol"], float)

        # Verify stored
        ds_name = result["dataset_id"]
        stored = ctx.get_dataset(ds_name)
        assert "ewma_vol" in stored.columns

    def test_ewma_different_spans(self, vol_tools):
        r_short = vol_tools["ewma_volatility"](
            dataset="returns", span=10,
        )
        r_long = vol_tools["ewma_volatility"](
            dataset="returns", span=60,
        )
        # Shorter span should be more responsive — mean vols should differ
        assert r_short["mean_vol"] != r_long["mean_vol"]

    def test_ewma_stores_with_parent(self, vol_tools, ctx):
        result = vol_tools["ewma_volatility"](dataset="returns")
        ds_name = result["dataset_id"]
        lineage = ctx.registry.lineage(ds_name)
        assert "returns" in lineage


# ------------------------------------------------------------------
# Forecast volatility (direct wraquant call)
# ------------------------------------------------------------------


class TestForecastVolatility:
    """Test GARCH forecasting through direct wraquant calls."""

    def test_garch_forecast(self, ctx):
        """Fit GARCH and forecast directly."""
        from wraquant.vol.models import garch_forecast

        returns = ctx.get_dataset("returns")["returns"].dropna().values
        result = garch_forecast(returns, horizon=10)

        assert "forecast_volatility" in result
        forecast = result["forecast_volatility"]
        assert len(forecast) == 10
        assert all(np.isfinite(v) for v in forecast)

        # Store forecast in context
        fc_df = pd.DataFrame({"forecast_vol": forecast})
        stored = ctx.store_dataset("garch_forecast", fc_df, source_op="forecast_volatility")
        assert stored["dataset_id"] == "garch_forecast"
        assert stored["rows"] == 10

    def test_garch_fit_and_store(self, ctx):
        """Fit GARCH and store as model."""
        from wraquant.vol.models import garch_fit

        returns = ctx.get_dataset("returns")["returns"].dropna().values
        result = garch_fit(returns)

        assert "persistence" in result
        assert "aic" in result

        stored = ctx.store_model(
            "garch_test", result,
            model_type="GARCH",
            source_dataset="returns",
            metrics={
                "persistence": float(result["persistence"]),
                "aic": float(result["aic"]),
            },
        )
        assert stored["model_id"] == "garch_test"
        assert stored["model_type"] == "GARCH"


# ------------------------------------------------------------------
# Model selection (direct wraquant call)
# ------------------------------------------------------------------


class TestModelSelection:
    """Test GARCH model selection."""

    def test_model_selection_ranks_models(self, vol_tools):
        result = vol_tools["model_selection"](
            dataset="returns", column="returns",
            models=["GARCH", "EGARCH", "GJR"],
        )
        assert result["tool"] == "model_selection"
        assert "ranking" in result

    def test_model_selection_default_models(self, vol_tools):
        result = vol_tools["model_selection"](
            dataset="returns", column="returns",
        )
        assert "ranking" in result


# ------------------------------------------------------------------
# News impact curve (direct wraquant call)
# ------------------------------------------------------------------


class TestNewsImpactCurve:
    """Test news impact curve through direct wraquant calls."""

    def test_news_impact_curve(self, ctx):
        """Compute NIC directly from returns."""
        from wraquant.vol.models import news_impact_curve

        returns = ctx.get_dataset("returns")["returns"].dropna().values
        result = news_impact_curve(returns, model_type="GARCH", n_points=50)

        assert "shocks" in result
        assert "conditional_variance" in result
        assert len(result["shocks"]) == 50
        assert len(result["conditional_variance"]) == 50

        # Store in context
        nic_df = pd.DataFrame({
            "shock": result["shocks"],
            "variance": result["conditional_variance"],
        })
        stored = ctx.store_dataset("nic_result", nic_df, source_op="news_impact_curve")
        assert stored["rows"] == 50

    def test_nic_egarch(self, ctx):
        """NIC for EGARCH (asymmetric)."""
        from wraquant.vol.models import news_impact_curve

        returns = ctx.get_dataset("returns")["returns"].dropna().values
        result = news_impact_curve(returns, model_type="EGARCH", n_points=50)

        assert len(result["shocks"]) == 50
        # EGARCH should show asymmetry
        if "asymmetry" in result:
            assert result["asymmetry"] is not None
