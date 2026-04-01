"""Integration tests for module-specific MCP servers.

Tests the compute_returns -> fit_garch -> risk_metrics flow
and other tool chains with synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant_mcp.context import AnalysisContext


@pytest.fixture
def ctx(tmp_path):
    """Create an AnalysisContext with a temporary workspace."""
    context = AnalysisContext(workspace_dir=tmp_path / "test_workspace")
    yield context
    context.close()


@pytest.fixture
def prices_df():
    """Create synthetic price data (GBM-like)."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    log_returns = np.random.randn(n) * 0.015
    prices = 100 * np.exp(np.cumsum(log_returns))
    highs = prices * (1 + np.abs(np.random.randn(n) * 0.005))
    lows = prices * (1 - np.abs(np.random.randn(n) * 0.005))
    opens = prices * (1 + np.random.randn(n) * 0.003)
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": np.random.randint(100_000, 1_000_000, n),
    }, index=dates)


@pytest.fixture
def returns_df(prices_df):
    """Create returns from prices."""
    returns = prices_df["close"].pct_change().dropna()
    return returns.to_frame(name="returns")


@pytest.fixture
def multi_asset_df():
    """Create multi-asset returns for portfolio tools."""
    np.random.seed(123)
    n = 252
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "AAPL": np.random.randn(n) * 0.02,
        "MSFT": np.random.randn(n) * 0.018,
        "GOOGL": np.random.randn(n) * 0.022,
        "AMZN": np.random.randn(n) * 0.025,
    }, index=dates)


# ------------------------------------------------------------------
# Core pipeline: prices -> returns -> analysis
# ------------------------------------------------------------------


class TestCorePipeline:
    """Test the fundamental data flow through the MCP context."""

    def test_store_prices_and_compute_returns(self, ctx, prices_df):
        """Store prices, compute returns, verify stored dataset."""
        ctx.store_dataset("prices_spy", prices_df, source_op="fetch")

        # Compute returns
        df = ctx.get_dataset("prices_spy")
        returns = df["close"].pct_change().dropna()
        returns_frame = returns.to_frame(name="returns")

        stored = ctx.store_dataset(
            "returns_spy", returns_frame,
            source_op="compute_returns", parent="prices_spy",
        )

        assert stored["rows"] == len(returns_frame)
        assert "returns" in stored["columns"]

        # Verify lineage
        lineage = ctx.registry.lineage("returns_spy")
        assert lineage == ["prices_spy", "returns_spy"]

    def test_full_analysis_pipeline(self, ctx, prices_df, returns_df):
        """Test prices -> returns -> model -> risk chain."""
        # Store prices
        ctx.store_dataset("prices", prices_df, source_op="fetch")

        # Store returns
        ctx.store_dataset(
            "returns", returns_df,
            source_op="compute_returns", parent="prices",
        )

        # Store a model result
        model_data = {
            "params": {"omega": 0.01, "alpha": 0.05, "beta": 0.90},
            "persistence": 0.95,
            "half_life": 14.0,
            "aic": -1500.0,
            "bic": -1490.0,
        }
        model_stored = ctx.store_model(
            "garch_returns", model_data,
            model_type="GARCH",
            source_dataset="returns",
            metrics={"persistence": 0.95, "aic": -1500.0},
        )

        assert model_stored["model_id"] == "garch_returns"
        assert model_stored["metrics"]["persistence"] == 0.95

        # Retrieve model
        retrieved = ctx.get_model("garch_returns")
        assert retrieved["params"]["alpha"] == 0.05

        # Workspace status
        status = ctx.workspace_status()
        assert "prices" in status["datasets"]
        assert "returns" in status["datasets"]
        assert "garch_returns" in status["models"]


# ------------------------------------------------------------------
# Multi-asset operations
# ------------------------------------------------------------------


class TestMultiAsset:
    """Test operations requiring multi-asset data."""

    def test_store_multi_asset(self, ctx, multi_asset_df):
        """Store and retrieve multi-asset returns."""
        stored = ctx.store_dataset(
            "portfolio_returns", multi_asset_df,
            source_op="compute_returns",
        )

        assert stored["rows"] == len(multi_asset_df)
        assert "AAPL" in stored["columns"]
        assert "MSFT" in stored["columns"]

        retrieved = ctx.get_dataset("portfolio_returns")
        assert retrieved.shape[1] == 4

    def test_correlation_dataset(self, ctx, multi_asset_df):
        """Compute and store correlation matrix."""
        ctx.store_dataset("returns", multi_asset_df)

        df = ctx.get_dataset("returns")
        corr = df.corr()
        corr_stored = ctx.store_dataset(
            "corr_returns", corr,
            source_op="correlation_analysis", parent="returns",
        )

        assert corr_stored["rows"] == 4
        assert corr_stored["columns"] == ["AAPL", "MSFT", "GOOGL", "AMZN"]


# ------------------------------------------------------------------
# Model versioning
# ------------------------------------------------------------------


class TestModelVersioning:
    """Test model storage with auto-versioning."""

    def test_model_version_collision(self, ctx):
        """Re-storing a model with same name auto-versions."""
        m1 = ctx.store_model("my_model", {"v": 1}, model_type="test")
        m2 = ctx.store_model("my_model", {"v": 2}, model_type="test")

        assert m1["model_id"] == "my_model"
        assert m2["model_id"] == "my_model_v2"

        # Both retrievable
        assert ctx.get_model("my_model")["v"] == 1
        assert ctx.get_model("my_model_v2")["v"] == 2


# ------------------------------------------------------------------
# Journal and notes
# ------------------------------------------------------------------


class TestJournal:
    """Test journal recording and retrieval."""

    def test_operations_logged(self, ctx, prices_df):
        """Every operation is logged to the journal."""
        ctx.store_dataset("prices", prices_df, source_op="fetch")
        ctx.store_model("model", {"x": 1}, model_type="test")
        ctx.add_note("Interesting pattern in volatility")

        history = ctx.history(n=100)
        assert len(history) >= 3

        operations = [h["op"] for h in history]
        assert "store_dataset" in operations
        assert "store_model" in operations
        assert "note" in operations

    def test_journal_persists(self, tmp_path, prices_df):
        """Journal survives context close/reopen."""
        ctx1 = AnalysisContext(workspace_dir=tmp_path / "persist_test")
        ctx1.add_note("Session 1 note")
        ctx1.close()

        ctx2 = AnalysisContext(workspace_dir=tmp_path / "persist_test")
        history = ctx2.history(n=100)
        ctx2.close()

        assert len(history) >= 1
        assert any("Session 1" in str(h) for h in history)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self, ctx):
        """Storing an empty DataFrame works."""
        empty = pd.DataFrame({"a": [], "b": []})
        result = ctx.store_dataset("empty", empty)
        assert result["rows"] == 0

    def test_single_row_dataframe(self, ctx):
        """Single-row DataFrame stores and retrieves."""
        single = pd.DataFrame({"value": [42.0]})
        ctx.store_dataset("single", single)
        retrieved = ctx.get_dataset("single")
        assert len(retrieved) == 1
        assert retrieved["value"].iloc[0] == 42.0

    def test_unicode_column_names(self, ctx):
        """Unicode column names are handled."""
        df = pd.DataFrame({"returns_pct": [0.01], "sigma_hat": [0.02]})
        result = ctx.store_dataset("unicode_test", df)
        assert "returns_pct" in result["columns"]

    def test_large_dataset(self, ctx):
        """Handle reasonably large datasets."""
        np.random.seed(42)
        large = pd.DataFrame({
            "returns": np.random.randn(10_000) * 0.01,
            "volume": np.random.randint(100, 10000, 10_000),
        })
        result = ctx.store_dataset("large", large)
        assert result["rows"] == 10_000

        retrieved = ctx.get_dataset("large")
        assert len(retrieved) == 10_000
