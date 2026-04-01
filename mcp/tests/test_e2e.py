"""End-to-end tests for wraquant-mcp.

Tests the full pipeline: context → adaptor → tools → results.
Uses synthetic data — no network calls or real market data.
"""

from __future__ import annotations

import json
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add mcp source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def workspace_dir(tmp_path):
    """Create a temporary workspace directory."""
    ws = tmp_path / "test_workspace"
    yield str(ws)
    if ws.exists():
        shutil.rmtree(ws)


@pytest.fixture
def ctx(workspace_dir):
    """Create an AnalysisContext for testing."""
    from wraquant_mcp.context import AnalysisContext

    context = AnalysisContext(workspace_dir)
    yield context
    context.close()


@pytest.fixture
def sample_prices():
    """Generate synthetic price data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    close = 100 + rng.normal(0, 1, 252).cumsum()
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.5, 252),
            "high": close + abs(rng.normal(0, 1, 252)),
            "low": close - abs(rng.normal(0, 1, 252)),
            "close": close,
            "volume": rng.integers(1000, 10000, 252),
        },
        index=dates,
    )


@pytest.fixture
def sample_returns(sample_prices):
    """Compute returns from sample prices."""
    return sample_prices["close"].pct_change().dropna().to_frame("returns")


# ======================================================================
# Context Tests
# ======================================================================


class TestContext:
    def test_store_and_get_dataset(self, ctx, sample_prices):
        result = ctx.store_dataset("prices", sample_prices)
        assert result["dataset_id"] == "prices"
        assert result["rows"] == 252

        df = ctx.get_dataset("prices")
        assert df.shape == (252, 5)
        assert "close" in df.columns

    def test_auto_versioning(self, ctx, sample_prices):
        ctx.store_dataset("prices", sample_prices)
        r2 = ctx.store_dataset("prices", sample_prices)
        assert r2["dataset_id"] == "prices_v2"

        datasets = ctx.list_datasets()
        assert "prices" in datasets
        assert "prices_v2" in datasets

    def test_store_and_get_model(self, ctx):
        model = {"type": "test", "params": {"alpha": 0.1}}
        result = ctx.store_model("test_model", model, model_type="test")
        assert result["model_id"] == "test_model"

        retrieved = ctx.get_model("test_model")
        assert retrieved["params"]["alpha"] == 0.1

    def test_lineage(self, ctx, sample_prices, sample_returns):
        ctx.store_dataset("prices", sample_prices)
        ctx.store_dataset("returns", sample_returns, parent="prices")
        lineage = ctx.registry.lineage("returns")
        assert lineage == ["prices", "returns"]

    def test_workspace_status(self, ctx, sample_prices):
        ctx.store_dataset("prices", sample_prices)
        status = ctx.workspace_status()
        assert "prices" in status["datasets"]
        assert status["total"] >= 1

    def test_journal(self, ctx, sample_prices):
        ctx.store_dataset("prices", sample_prices)
        ctx.add_note("Testing the journal")
        history = ctx.history()
        assert len(history) >= 2
        assert any(e["op"] == "note" for e in history)

    def test_dataset_info(self, ctx, sample_prices):
        ctx.store_dataset("prices", sample_prices)
        info = ctx.dataset_info("prices")
        assert info["rows"] == 252
        assert "close" in info["columns"]
        assert "head" in info
        assert "lineage" in info

    def test_missing_dataset_raises(self, ctx):
        with pytest.raises(KeyError):
            ctx.get_dataset("nonexistent")

    def test_missing_model_raises(self, ctx):
        with pytest.raises(KeyError):
            ctx.get_model("nonexistent")


# ======================================================================
# ID Registry Tests
# ======================================================================


class TestIDRegistry:
    def test_register_and_get(self):
        from wraquant_mcp.ids import IDRegistry

        reg = IDRegistry()
        name = reg.register("test", "dataset")
        assert name == "test"
        assert reg.exists("test")

    def test_auto_version(self):
        from wraquant_mcp.ids import IDRegistry

        reg = IDRegistry()
        reg.register("test", "dataset")
        name2 = reg.register("test", "dataset")
        assert name2 == "test_v2"

    def test_lineage(self):
        from wraquant_mcp.ids import IDRegistry

        reg = IDRegistry()
        reg.register("a", "dataset")
        reg.register("b", "dataset", parent="a")
        reg.register("c", "dataset", parent="b")
        assert reg.lineage("c") == ["a", "b", "c"]

    def test_latest(self):
        from wraquant_mcp.ids import IDRegistry

        reg = IDRegistry()
        reg.register("x", "dataset")
        reg.register("x", "dataset")
        latest = reg.latest("x")
        assert latest == "x_v2"

    def test_list_by_type(self):
        from wraquant_mcp.ids import IDRegistry

        reg = IDRegistry()
        reg.register("ds1", "dataset")
        reg.register("ds2", "dataset")
        reg.register("mod1", "model")
        assert len(reg.list_datasets()) == 2
        assert len(reg.list_models()) == 1


# ======================================================================
# Adaptor Tests
# ======================================================================


class TestAdaptor:
    def test_detect_data_params(self):
        from wraquant_mcp.adaptor import _detect_data_params

        def func(returns, period=14, benchmark=None):
            pass

        params = _detect_data_params(func)
        assert "returns" in params
        assert "benchmark" in params
        assert "period" not in params

    def test_handle_scalar_result(self):
        from wraquant_mcp.adaptor import _handle_result
        from wraquant_mcp.context import AnalysisContext

        ctx = AnalysisContext("/tmp/test_adaptor")
        result = _handle_result(ctx, 1.5, "test", "test", {})
        assert result["result"] == 1.5
        ctx.close()

    def test_handle_dict_result(self):
        from wraquant_mcp.adaptor import _handle_result
        from wraquant_mcp.context import AnalysisContext

        ctx = AnalysisContext("/tmp/test_adaptor2")
        result = _handle_result(
            ctx, {"sharpe": 1.5, "vol": 0.2}, "test", "test", {}
        )
        assert result["result"]["sharpe"] == 1.5
        ctx.close()

    def test_handle_series_result(self):
        from wraquant_mcp.adaptor import _handle_result
        from wraquant_mcp.context import AnalysisContext

        ctx = AnalysisContext("/tmp/test_adaptor3")
        series = pd.Series([1.0, 2.0, 3.0], name="rsi")
        result = _handle_result(ctx, series, "test", "test", {})
        assert "dataset_id" in result
        assert result["summary"]["mean"] == 2.0
        ctx.close()


# ======================================================================
# Tool Integration Tests
# ======================================================================


class TestToolIntegration:
    """Test the actual wraquant tool functions through the context."""

    def test_store_compute_returns_analyze_flow(self, ctx, sample_prices):
        """Full flow: store prices → compute returns → analyze."""
        # Store prices
        ctx.store_dataset("prices_test", sample_prices)

        # Compute returns manually (simulating compute_returns tool)
        prices = ctx.get_dataset("prices_test")
        returns = prices["close"].pct_change().dropna()
        ctx.store_dataset(
            "returns_test",
            returns.to_frame("returns"),
            source_op="compute_returns",
            parent="prices_test",
        )

        # Verify returns stored
        ret_df = ctx.get_dataset("returns_test")
        assert len(ret_df) == 251
        assert "returns" in ret_df.columns

        # Compute risk metrics
        from wraquant.risk.metrics import sharpe_ratio, max_drawdown

        sr = float(sharpe_ratio(ret_df["returns"]))
        assert np.isfinite(sr)

    def test_garch_fit_and_store(self, ctx, sample_returns):
        """Test GARCH fitting through context."""
        ctx.store_dataset("returns", sample_returns)

        # Fit GARCH
        from wraquant.vol.models import garch_fit

        df = ctx.get_dataset("returns")
        result = garch_fit(df["returns"].dropna().values)

        # Store model
        stored = ctx.store_model(
            "garch_test",
            result,
            model_type="GARCH",
            source_dataset="returns",
            metrics={
                "persistence": float(result["persistence"]),
                "aic": float(result["aic"]),
            },
        )
        assert stored["model_id"] == "garch_test"

        # Retrieve model
        model = ctx.get_model("garch_test")
        assert "persistence" in model

    def test_regime_detection_and_store(self, ctx, sample_returns):
        """Test regime detection through context."""
        ctx.store_dataset("returns", sample_returns)

        from wraquant.regimes.base import detect_regimes

        df = ctx.get_dataset("returns")
        result = detect_regimes(df["returns"].dropna().values, method="hmm", n_regimes=2)

        # Store regime model
        stored = ctx.store_model(
            "hmm_test",
            result,
            model_type="hmm_2state",
            source_dataset="returns",
        )
        assert stored["model_id"] == "hmm_test"

        # Store regime states as dataset
        states_df = pd.DataFrame({"regime": result.states})
        ctx.store_dataset("regime_states", states_df, source_op="detect_regimes")
        assert "regime_states" in ctx.list_datasets()

    def test_ta_indicator_and_store(self, ctx, sample_prices):
        """Test TA indicator computation through context."""
        ctx.store_dataset("prices", sample_prices)

        from wraquant.ta import rsi

        df = ctx.get_dataset("prices")
        indicator = rsi(df["close"], period=14)

        df["rsi"] = indicator.values
        ctx.store_dataset(
            "prices_rsi", df, source_op="compute_indicator", parent="prices"
        )

        result = ctx.get_dataset("prices_rsi")
        assert "rsi" in result.columns

    def test_multi_step_pipeline(self, ctx, sample_prices):
        """Test full pipeline: prices → returns → GARCH → risk metrics."""
        # 1. Store prices
        ctx.store_dataset("prices", sample_prices)

        # 2. Compute returns
        prices = ctx.get_dataset("prices")
        returns = prices["close"].pct_change().dropna()
        ctx.store_dataset(
            "returns", returns.to_frame("returns"),
            source_op="compute_returns", parent="prices",
        )

        # 3. Risk metrics
        from wraquant.risk.metrics import sharpe_ratio, sortino_ratio

        ret = ctx.get_dataset("returns")["returns"]
        metrics = {
            "sharpe": float(sharpe_ratio(ret)),
            "sortino": float(sortino_ratio(ret)),
        }
        assert np.isfinite(metrics["sharpe"])

        # 4. Verify lineage
        lineage = ctx.registry.lineage("returns")
        assert "prices" in lineage

        # 5. Verify workspace has everything
        status = ctx.workspace_status()
        assert len(status["datasets"]) >= 2

    def test_sanitize_for_json(self):
        """Test that _sanitize_for_json handles numpy types."""
        from wraquant_mcp.context import _sanitize_for_json

        data = {
            "float64": np.float64(1.5),
            "int64": np.int64(42),
            "array": np.array([1.0, 2.0]),
            "bool": np.bool_(True),
            "nested": {"value": np.float64(3.14)},
            "none": None,
            "string": "hello",
        }
        result = _sanitize_for_json(data)

        assert isinstance(result["float64"], float)
        assert isinstance(result["int64"], int)
        assert isinstance(result["array"], list)
        assert isinstance(result["bool"], bool)
        assert isinstance(result["nested"]["value"], float)
        assert result["none"] is None
        assert result["string"] == "hello"

        # Verify JSON serializable
        json.dumps(result)


# ======================================================================
# Workspace Tests
# ======================================================================


class TestWorkspace:
    def test_workspace_creates_directory(self, workspace_dir):
        from wraquant_mcp.context import AnalysisContext

        ctx = AnalysisContext(workspace_dir)
        assert Path(workspace_dir).exists()
        assert (Path(workspace_dir) / "data.duckdb").exists()
        assert (Path(workspace_dir) / "models").exists()
        ctx.close()

    def test_journal_persistence(self, ctx, sample_prices):
        ctx.store_dataset("prices", sample_prices)
        ctx.add_note("Test note")

        # Read journal from disk
        journal_path = Path(ctx.workspace_dir) / "journal.jsonl"
        assert journal_path.exists()
        with open(journal_path) as f:
            lines = f.readlines()
        assert len(lines) >= 2

    def test_manifest(self, ctx):
        manifest_path = Path(ctx.workspace_dir) / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "workspace" in manifest
        assert "created" in manifest
