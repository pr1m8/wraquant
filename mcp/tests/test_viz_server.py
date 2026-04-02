"""Tests for visualization MCP tools.

Tests: plot_returns, plot_distribution, plot_equity_curve,
plot_drawdown, plot_correlation, portfolio_dashboard.

Viz tools return dicts with base64-encoded PNG images. Tests verify
the tools run without error and return the expected structure.
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
    """Create an AnalysisContext with synthetic data for plotting."""
    ws = tmp_path / "test_viz"
    context = AnalysisContext(str(ws))

    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    log_rets = rng.normal(0.0003, 0.015, n)
    close = 100 * np.exp(np.cumsum(log_rets))

    # OHLCV prices
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

    # Multi-asset returns for correlation plots
    multi = pd.DataFrame(
        {
            "AAPL": rng.normal(0.0003, 0.02, n),
            "MSFT": rng.normal(0.0002, 0.018, n),
            "GOOGL": rng.normal(0.0001, 0.022, n),
        },
        index=dates,
    )
    context.store_dataset("multi_returns", multi)

    yield context
    context.close()
    if ws.exists():
        shutil.rmtree(ws)


@pytest.fixture
def viz_tools(ctx):
    """Register viz tools on mock MCP."""
    from wraquant_mcp.servers.viz import register_viz_tools

    mock = MockMCP()
    register_viz_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _assert_viz_result(result: dict, tool_name: str) -> None:
    """Verify common structure of viz tool results."""
    assert result["tool"] == tool_name
    assert "image_base64" in result
    assert result["format"] == "png"
    # image_base64 can be empty string if rendering backend not available
    assert isinstance(result["image_base64"], str)


# ------------------------------------------------------------------
# plot_returns
# ------------------------------------------------------------------


class TestPlotReturns:
    """Test plot_returns tool."""

    def test_plot_cumulative_returns(self, viz_tools):
        result = viz_tools["plot_returns"](
            dataset="returns",
            column="returns",
            cumulative=True,
        )
        _assert_viz_result(result, "plot_returns")
        assert result["type"] == "cumulative"

    def test_plot_drawdown_returns(self, viz_tools):
        result = viz_tools["plot_returns"](
            dataset="returns",
            column="returns",
            cumulative=False,
        )
        _assert_viz_result(result, "plot_returns")
        assert result["type"] == "drawdown"


# ------------------------------------------------------------------
# plot_distribution
# ------------------------------------------------------------------


class TestPlotDistribution:
    """Test plot_distribution tool."""

    def test_plot_distribution_basic(self, viz_tools):
        result = viz_tools["plot_distribution"](
            dataset="returns",
            column="returns",
        )
        _assert_viz_result(result, "plot_distribution")
        assert "skewness" in result
        assert "kurtosis" in result
        assert isinstance(result["skewness"], float)
        assert isinstance(result["kurtosis"], float)


# ------------------------------------------------------------------
# plot_equity_curve
# ------------------------------------------------------------------


class TestPlotEquityCurve:
    """Test plot_equity_curve tool."""

    def test_plot_equity_curve_basic(self, viz_tools):
        result = viz_tools["plot_equity_curve"](
            dataset="returns",
            column="returns",
        )
        _assert_viz_result(result, "plot_equity_curve")


# ------------------------------------------------------------------
# plot_drawdown
# ------------------------------------------------------------------


class TestPlotDrawdown:
    """Test plot_drawdown tool."""

    def test_plot_drawdown_basic(self, viz_tools):
        result = viz_tools["plot_drawdown"](
            dataset="returns",
            column="returns",
        )
        _assert_viz_result(result, "plot_drawdown")


# ------------------------------------------------------------------
# plot_correlation
# ------------------------------------------------------------------


class TestPlotCorrelation:
    """Test plot_correlation tool."""

    def test_plot_correlation_pearson(self, viz_tools):
        result = viz_tools["plot_correlation"](
            dataset="multi_returns",
            method="pearson",
        )
        _assert_viz_result(result, "plot_correlation")
        assert result["method"] == "pearson"
        assert len(result["assets"]) == 3

    def test_plot_correlation_spearman(self, viz_tools):
        result = viz_tools["plot_correlation"](
            dataset="multi_returns",
            method="spearman",
        )
        _assert_viz_result(result, "plot_correlation")
        assert result["method"] == "spearman"


# ------------------------------------------------------------------
# plot_heatmap
# ------------------------------------------------------------------


class TestPlotHeatmap:
    """Test plot_heatmap tool."""

    def test_plot_heatmap_basic(self, viz_tools):
        result = viz_tools["plot_heatmap"](
            dataset="multi_returns",
        )
        _assert_viz_result(result, "plot_heatmap")
        assert result["n_assets"] == 3


# ------------------------------------------------------------------
# portfolio_dashboard
# ------------------------------------------------------------------


class TestPortfolioDashboard:
    """Test portfolio_dashboard tool."""

    def test_dashboard_basic(self, viz_tools):
        result = viz_tools["portfolio_dashboard"](
            dataset="returns",
            column="returns",
        )
        _assert_viz_result(result, "portfolio_dashboard")


# ------------------------------------------------------------------
# plot_rolling_metrics
# ------------------------------------------------------------------


class TestPlotRollingMetrics:
    """Test plot_rolling_metrics tool."""

    def test_rolling_metrics_basic(self, viz_tools):
        result = viz_tools["plot_rolling_metrics"](
            dataset="returns",
            column="returns",
            window=63,
        )
        _assert_viz_result(result, "plot_rolling_metrics")


# ------------------------------------------------------------------
# plot_candlestick
# ------------------------------------------------------------------


class TestPlotCandlestick:
    """Test plot_candlestick tool."""

    def test_candlestick_basic(self, viz_tools):
        result = viz_tools["plot_candlestick"](
            dataset="prices",
        )
        _assert_viz_result(result, "plot_candlestick")
        assert result["n_bars"] > 0
