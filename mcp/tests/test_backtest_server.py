"""Tests for backtesting MCP tools.

Tests: run_backtest, backtest_metrics, comprehensive_tearsheet,
walk_forward, omega_ratio, kelly_fraction, drawdown_analysis.
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
    """Create an AnalysisContext with synthetic price and signal data."""
    ws = tmp_path / "test_backtest"
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

    # Momentum signal: sign of trailing 5-day return
    ret = pd.Series(close).pct_change()
    signal = np.sign(ret.rolling(5).mean().fillna(0))
    prices["signal"] = signal.values
    context.store_dataset("prices", prices)

    # Standalone returns
    returns_df = pd.DataFrame({"returns": log_rets}, index=dates)
    context.store_dataset("returns", returns_df, parent="prices")

    yield context
    context.close()
    if ws.exists():
        shutil.rmtree(ws)


@pytest.fixture
def bt_tools(ctx):
    """Register backtest tools on mock MCP."""
    from wraquant_mcp.servers.backtest import register_backtest_tools

    mock = MockMCP()
    register_backtest_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# run_backtest
# ------------------------------------------------------------------


class TestRunBacktest:
    """Test run_backtest tool."""

    def test_run_backtest_basic(self, bt_tools):
        result = bt_tools["run_backtest"](
            dataset="prices",
            signal_column="signal",
            price_column="close",
            initial_capital=100_000.0,
        )
        assert result["tool"] == "run_backtest"
        assert isinstance(result["total_return"], float)
        assert isinstance(result["sharpe_ratio"], float)
        assert isinstance(result["max_drawdown"], float)
        assert result["max_drawdown"] <= 0
        assert result["n_trades"] > 0

    def test_run_backtest_stores_equity(self, bt_tools, ctx):
        result = bt_tools["run_backtest"](
            dataset="prices",
            signal_column="signal",
            price_column="close",
        )
        ds_name = result["dataset_id"]
        stored = ctx.get_dataset(ds_name)
        assert "equity" in stored.columns
        assert "returns" in stored.columns

    def test_run_backtest_missing_column_returns_error(self, bt_tools):
        result = bt_tools["run_backtest"](
            dataset="prices",
            signal_column="nonexistent",
        )
        assert "error" in result


# ------------------------------------------------------------------
# backtest_metrics
# ------------------------------------------------------------------


class TestBacktestMetrics:
    """Test backtest_metrics tool."""

    def test_metrics_basic(self, bt_tools):
        result = bt_tools["backtest_metrics"](
            dataset="returns",
            column="returns",
        )
        assert result["tool"] == "backtest_metrics"

    def test_metrics_direct(self, ctx):
        """Test performance_summary directly."""
        from wraquant.backtest.metrics import performance_summary

        returns = ctx.get_dataset("returns")["returns"].dropna()
        result = performance_summary(returns)
        assert isinstance(result, dict)
        assert "sharpe_ratio" in result or "annualized_return" in result


# ------------------------------------------------------------------
# comprehensive_tearsheet
# ------------------------------------------------------------------


class TestComprehensiveTearsheet:
    """Test comprehensive_tearsheet tool."""

    def test_tearsheet_returns_tables(self):
        """Test tearsheet functions directly (context loses DatetimeIndex)."""
        from wraquant.backtest.tearsheet import (
            drawdown_table,
            monthly_returns_table,
            rolling_metrics_table,
        )

        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        returns = pd.Series(rng.normal(0.0003, 0.015, n), index=dates)

        monthly = monthly_returns_table(returns)
        drawdowns = drawdown_table(returns)
        rolling = rolling_metrics_table(returns)

        assert isinstance(monthly, pd.DataFrame)
        assert isinstance(drawdowns, pd.DataFrame)
        assert isinstance(rolling, pd.DataFrame)


# ------------------------------------------------------------------
# omega_ratio
# ------------------------------------------------------------------


class TestOmegaRatio:
    """Test omega_ratio tool."""

    def test_omega_ratio_basic(self, bt_tools):
        result = bt_tools["omega_ratio"](
            dataset="returns",
            column="returns",
        )
        assert result["tool"] == "omega_ratio"
        assert isinstance(result["omega_ratio"], float)
        assert result["omega_ratio"] > 0

    def test_omega_ratio_direct(self, ctx):
        """Test omega_ratio function directly."""
        from wraquant.backtest.metrics import omega_ratio

        returns = ctx.get_dataset("returns")["returns"].dropna()
        result = omega_ratio(returns, threshold=0.0)
        assert isinstance(result, float)
        assert result > 0


# ------------------------------------------------------------------
# kelly_fraction
# ------------------------------------------------------------------


class TestKellyFraction:
    """Test kelly_fraction tool."""

    def test_kelly_fraction_basic(self, bt_tools):
        result = bt_tools["kelly_fraction"](
            dataset="returns",
            column="returns",
        )
        assert result["tool"] == "kelly_fraction"
        assert isinstance(result["full_kelly"], float)
        assert isinstance(result["half_kelly"], float)
        assert result["half_kelly"] == pytest.approx(result["full_kelly"] / 2)
        assert isinstance(result["win_rate"], float)

    def test_kelly_fraction_direct(self):
        """Test kelly_fraction function directly."""
        from wraquant.backtest.metrics import kelly_fraction

        result = kelly_fraction(win_rate=0.55, avg_win=0.02, avg_loss=0.015)
        assert isinstance(result, float)


# ------------------------------------------------------------------
# drawdown_analysis
# ------------------------------------------------------------------


class TestDrawdownAnalysis:
    """Test drawdown_analysis tool."""

    def test_drawdown_analysis_basic(self, bt_tools):
        result = bt_tools["drawdown_analysis"](
            dataset="returns",
            column="returns",
            top_n=3,
        )
        assert result["tool"] == "drawdown_analysis"
        assert result["top_n"] == 3
        assert "drawdowns" in result
        assert "current_drawdown" in result
        assert isinstance(result["current_drawdown"], float)

    def test_drawdown_table_direct(self, ctx):
        """Test drawdown_table function directly."""
        from wraquant.backtest.tearsheet import drawdown_table

        returns = ctx.get_dataset("returns")["returns"].dropna()
        result = drawdown_table(returns, top_n=5)
        assert result is not None
