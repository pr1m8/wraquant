"""Tests for market microstructure MCP tools.

Tests: liquidity_metrics, toxicity_analysis, market_quality,
price_impact.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
import shutil

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
    ws = tmp_path / "test_ws"
    context = AnalysisContext(str(ws))
    rng = np.random.default_rng(42)
    prices = pd.DataFrame({
        "close": 100 + rng.normal(0, 1, 252).cumsum(),
        "volume": rng.integers(1000, 10000, 252).astype(float),
    })
    context.store_dataset("prices", prices)
    returns = prices["close"].pct_change().dropna()
    context.store_dataset("returns", returns.to_frame("returns"), parent="prices")
    yield context
    context.close()


@pytest.fixture
def micro_tools(ctx):
    """Register microstructure tools and return them."""
    from wraquant_mcp.servers.microstructure import register_microstructure_tools

    mock = MockMCP()
    register_microstructure_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# Liquidity metrics
# ------------------------------------------------------------------


class TestLiquidityMetrics:
    """Test liquidity_metrics tool."""

    def test_liquidity_returns_amihud_kyle(self, micro_tools):
        """Liquidity metrics returns Amihud and Kyle lambda values."""
        result = micro_tools["liquidity_metrics"](
            dataset="prices",
            price_col="close",
            volume_col="volume",
            window=20,
        )

        assert result["tool"] == "liquidity_metrics"
        assert isinstance(result["amihud_illiquidity"], float)
        assert np.isfinite(result["amihud_illiquidity"])
        assert result["amihud_illiquidity"] >= 0

        assert isinstance(result["kyle_lambda_latest"], float)
        assert np.isfinite(result["kyle_lambda_latest"])

        # Roll spread can be None (NaN sanitized to None) if serial
        # covariance is non-negative, so just check the key exists
        assert "roll_spread" in result

        assert isinstance(result["observations"], int)
        assert result["observations"] > 0
        assert result["window"] == 20

        # Should store a dataset
        assert "dataset_id" in result


# ------------------------------------------------------------------
# Toxicity analysis
# ------------------------------------------------------------------


class TestToxicityAnalysis:
    """Test toxicity_analysis tool."""

    def test_toxicity_returns_vpin(self, micro_tools, ctx):
        """Toxicity analysis returns VPIN estimate."""
        # Need OHLC data for bulk volume classification
        rng = np.random.default_rng(55)
        n = 252
        close = 100 + rng.normal(0, 1, n).cumsum()
        high = close + np.abs(rng.normal(0, 0.5, n))
        low = close - np.abs(rng.normal(0, 0.5, n))
        volume = rng.integers(5000, 50000, n).astype(float)

        ctx.store_dataset("ohlcv", pd.DataFrame({
            "open": close + rng.normal(0, 0.2, n),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }))

        result = micro_tools["toxicity_analysis"](
            dataset="ohlcv",
            price_col="close",
            volume_col="volume",
            n_buckets=20,
        )

        assert result["tool"] == "toxicity_analysis"
        assert result["n_buckets"] == 20
        assert isinstance(result["observations"], int)
        assert result["observations"] > 0

        # VPIN should be present and between 0 and 1
        if result["vpin_latest"] is not None:
            assert isinstance(result["vpin_latest"], float)
            assert 0 <= result["vpin_latest"] <= 1
        if result["vpin_mean"] is not None:
            assert isinstance(result["vpin_mean"], float)
            assert 0 <= result["vpin_mean"] <= 1

        # OFI should be present
        if result["ofi_latest"] is not None:
            assert isinstance(result["ofi_latest"], float)
            assert np.isfinite(result["ofi_latest"])

        # Should store a dataset
        assert "dataset_id" in result


# ------------------------------------------------------------------
# Market quality
# ------------------------------------------------------------------


class TestMarketQuality:
    """Test market_quality tool."""

    def test_market_quality_returns_variance_ratio(self, micro_tools):
        """Market quality returns variance ratio and efficiency ratio."""
        result = micro_tools["market_quality"](
            dataset="prices",
            price_col="close",
        )

        assert result["tool"] == "market_quality"

        # Variance ratio is a dict with vr, z_stat, p_value
        vr = result["variance_ratio"]
        assert isinstance(vr, dict)
        assert "vr" in vr
        assert isinstance(vr["vr"], float)
        assert np.isfinite(vr["vr"])
        assert vr["vr"] > 0
        assert "z_stat" in vr
        assert "p_value" in vr

        # Market efficiency ratio
        mer = result["market_efficiency_ratio"]
        assert isinstance(mer, dict)

        assert isinstance(result["observations"], int)
        assert result["observations"] > 0


# ------------------------------------------------------------------
# Price impact
# ------------------------------------------------------------------


class TestPriceImpact:
    """Test price_impact tool."""

    def test_price_impact_returns_values(self, micro_tools):
        """Price impact returns mean and median impact."""
        result = micro_tools["price_impact"](
            dataset="prices",
            price_col="close",
            volume_col="volume",
        )

        assert result["tool"] == "price_impact"

        if result["mean_impact"] is not None:
            assert isinstance(result["mean_impact"], float)
            assert np.isfinite(result["mean_impact"])
        if result["median_impact"] is not None:
            assert isinstance(result["median_impact"], float)
            assert np.isfinite(result["median_impact"])

        assert isinstance(result["observations"], int)
        assert result["observations"] > 0

        # Should store a dataset
        assert "dataset_id" in result
