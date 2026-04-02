"""Tests for technical analysis MCP tools.

Tests: list_indicators, multi_indicator, scan_signals.
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
    """Create an AnalysisContext with OHLCV data."""
    ws = tmp_path / "test_ta"
    context = AnalysisContext(str(ws))

    rng = np.random.default_rng(42)
    n = 252
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    log_rets = rng.normal(0.0003, 0.015, n)
    close = 100 * np.exp(np.cumsum(log_rets))

    prices = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * (1 + abs(rng.normal(0, 0.008, n))),
        "low": close * (1 - abs(rng.normal(0, 0.008, n))),
        "close": close,
        "volume": rng.integers(1000, 100_000, n),
    }, index=dates)
    context.store_dataset("prices", prices)

    yield context
    context.close()
    if ws.exists():
        shutil.rmtree(ws)


@pytest.fixture
def ta_tools(ctx):
    """Register TA tools on mock MCP."""
    from wraquant_mcp.servers.ta import register_ta_tools

    mock = MockMCP()
    register_ta_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# List indicators
# ------------------------------------------------------------------


class TestListIndicators:
    """Test list_indicators tool."""

    def test_list_all_categories(self, ta_tools):
        result = ta_tools["list_indicators"]()
        assert result["tool"] == "list_indicators"
        assert "categories" in result
        assert "total" in result
        assert result["total"] > 50

        # Check expected categories exist
        cats = result["categories"]
        for expected in ["overlap", "momentum", "volume", "trend", "volatility"]:
            assert expected in cats
            assert "indicators" in cats[expected]
            assert "count" in cats[expected]

    def test_list_specific_category(self, ta_tools):
        result = ta_tools["list_indicators"](category="momentum")
        assert result["category"] == "momentum"
        assert "indicators" in result
        assert "rsi" in result["indicators"]
        assert "macd" in result["indicators"]
        assert result["count"] > 0

    def test_list_overlap_category(self, ta_tools):
        result = ta_tools["list_indicators"](category="overlap")
        assert "sma" in result["indicators"]
        assert "ema" in result["indicators"]
        assert "bollinger_bands" in result["indicators"]

    def test_list_volume_category(self, ta_tools):
        result = ta_tools["list_indicators"](category="volume")
        assert "obv" in result["indicators"]
        assert "mfi" in result["indicators"]

    def test_list_unknown_category_returns_error(self, ta_tools):
        result = ta_tools["list_indicators"](category="nonexistent")
        assert "error" in result
        assert "categories" in result  # should suggest valid categories

    def test_list_all_contains_fibonacci(self, ta_tools):
        result = ta_tools["list_indicators"]()
        assert "fibonacci" in result["categories"]
        fib = result["categories"]["fibonacci"]
        assert "fibonacci_retracements" in fib["indicators"]

    def test_list_all_contains_exotic(self, ta_tools):
        result = ta_tools["list_indicators"]()
        assert "exotic" in result["categories"]

    def test_list_all_contains_smoothing(self, ta_tools):
        result = ta_tools["list_indicators"]()
        assert "smoothing" in result["categories"]

    def test_list_patterns(self, ta_tools):
        result = ta_tools["list_indicators"](category="patterns")
        assert "doji" in result["indicators"]
        assert "hammer" in result["indicators"]
        assert "engulfing" in result["indicators"]

    def test_list_statistics(self, ta_tools):
        result = ta_tools["list_indicators"](category="statistics")
        assert "zscore" in result["indicators"]
        assert "hurst_exponent" in result["indicators"]


# ------------------------------------------------------------------
# Multi indicator
# ------------------------------------------------------------------


class TestMultiIndicator:
    """Test multi_indicator tool."""

    def test_compute_single_indicator(self, ta_tools, ctx):
        result = ta_tools["multi_indicator"](
            dataset="prices",
            indicators=["rsi"],
            column="close",
            period=14,
        )
        assert result["tool"] == "multi_indicator"
        assert "rsi" in result["computed"]
        assert "dataset_id" in result

        # Verify stored with RSI column
        ds = ctx.get_dataset(result["dataset_id"])
        assert "rsi" in ds.columns

    def test_compute_multiple_indicators(self, ta_tools, ctx):
        result = ta_tools["multi_indicator"](
            dataset="prices",
            indicators=["rsi", "sma", "ema"],
            column="close",
            period=14,
        )
        assert len(result["computed"]) >= 2  # at least 2 should succeed
        ds = ctx.get_dataset(result["dataset_id"])
        # Original columns should still be present
        assert "close" in ds.columns

    def test_compute_stores_with_parent(self, ta_tools, ctx):
        result = ta_tools["multi_indicator"](
            dataset="prices",
            indicators=["rsi"],
        )
        ds_name = result["dataset_id"]
        lineage = ctx.registry.lineage(ds_name)
        assert "prices" in lineage

    def test_invalid_indicator_reports_error(self, ta_tools):
        result = ta_tools["multi_indicator"](
            dataset="prices",
            indicators=["nonexistent_indicator"],
        )
        assert result["errors"] is not None
        assert any("nonexistent_indicator" in e for e in result["errors"])

    def test_mixed_valid_invalid(self, ta_tools):
        result = ta_tools["multi_indicator"](
            dataset="prices",
            indicators=["rsi", "totally_fake"],
        )
        # RSI should succeed
        assert "rsi" in result["computed"]
        # Fake should be in errors
        assert any("totally_fake" in e for e in result["errors"])

    def test_invalid_column_returns_error(self, ta_tools):
        result = ta_tools["multi_indicator"](
            dataset="prices",
            indicators=["rsi"],
            column="nonexistent_column",
        )
        assert "error" in result

    def test_summaries_contain_latest_value(self, ta_tools):
        result = ta_tools["multi_indicator"](
            dataset="prices",
            indicators=["rsi"],
            period=14,
        )
        assert "summaries" in result
        if "rsi" in result["summaries"]:
            assert "latest" in result["summaries"]["rsi"]
            assert isinstance(result["summaries"]["rsi"]["latest"], float)


# ------------------------------------------------------------------
# Scan signals
# ------------------------------------------------------------------


class TestScanSignals:
    """Test scan_signals tool."""

    def test_scan_returns_indicators(self, ta_tools):
        result = ta_tools["scan_signals"](
            dataset="prices", column="close",
        )
        assert result["tool"] == "scan_signals"
        assert "indicators" in result
        assert "consensus" in result
        assert result["consensus"] in ("overbought", "oversold", "neutral")

    def test_scan_has_rsi(self, ta_tools):
        result = ta_tools["scan_signals"](dataset="prices")
        indicators = result["indicators"]
        if "rsi" in indicators:
            assert "value" in indicators["rsi"]
            assert "signal" in indicators["rsi"]
            assert indicators["rsi"]["signal"] in ("overbought", "oversold", "neutral")

    def test_scan_counts(self, ta_tools):
        result = ta_tools["scan_signals"](dataset="prices")
        assert "overbought_count" in result
        assert "oversold_count" in result
        assert isinstance(result["overbought_count"], int)
        assert isinstance(result["oversold_count"], int)
        assert result["overbought_count"] >= 0
        assert result["oversold_count"] >= 0

    def test_scan_consensus_logic(self, ta_tools):
        """Consensus should be OB if >= 2 indicators OB, OS if >= 2 OS."""
        result = ta_tools["scan_signals"](dataset="prices")
        if result["overbought_count"] >= 2:
            assert result["consensus"] == "overbought"
        elif result["oversold_count"] >= 2:
            assert result["consensus"] == "oversold"
        else:
            assert result["consensus"] == "neutral"
