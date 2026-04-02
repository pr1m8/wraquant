"""Tests for AnalysisContext — DuckDB state manager.

Covers: store/get dataset, store/get model, ID versioning,
lineage tracking, workspace_status.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


@pytest.fixture
def ctx(tmp_path):
    """Create an AnalysisContext with a temporary workspace."""
    context = AnalysisContext(workspace_dir=tmp_path / "test_workspace")
    yield context
    context.close()


@pytest.fixture
def sample_df():
    """Create a simple price DataFrame."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    return pd.DataFrame({
        "close": prices,
        "volume": np.random.randint(1000, 10000, n),
    }, index=dates)


@pytest.fixture
def returns_df(sample_df):
    """Create a returns DataFrame from prices."""
    returns = sample_df["close"].pct_change().dropna()
    return returns.to_frame(name="returns")


# ------------------------------------------------------------------
# Dataset operations
# ------------------------------------------------------------------


class TestDatasetOperations:
    """Test store/get/list dataset operations."""

    def test_store_and_get_dataset(self, ctx, sample_df):
        """Store a DataFrame and retrieve it."""
        result = ctx.store_dataset("prices_aapl", sample_df, source_op="test")

        assert result["dataset_id"] == "prices_aapl"
        assert result["rows"] == len(sample_df)
        assert "close" in result["columns"]

        retrieved = ctx.get_dataset("prices_aapl")
        assert len(retrieved) == len(sample_df)
        assert "close" in retrieved.columns

    def test_dataset_exists(self, ctx, sample_df):
        """Check dataset existence."""
        assert not ctx.dataset_exists("nonexistent")
        ctx.store_dataset("test_data", sample_df)
        assert ctx.dataset_exists("test_data")

    def test_list_datasets(self, ctx, sample_df):
        """List all datasets."""
        ctx.store_dataset("ds1", sample_df)
        ctx.store_dataset("ds2", sample_df)
        datasets = ctx.list_datasets()
        assert "ds1" in datasets
        assert "ds2" in datasets

    def test_get_missing_dataset_raises(self, ctx):
        """Getting a missing dataset raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            ctx.get_dataset("does_not_exist")

    def test_dataset_info(self, ctx, sample_df):
        """Dataset info returns metadata, head, and stats."""
        ctx.store_dataset("prices", sample_df, source_op="fetch")
        info = ctx.dataset_info("prices")

        assert info["dataset_id"] == "prices"
        assert info["rows"] == len(sample_df)
        assert "close" in info["columns"]
        assert info["source_op"] == "fetch"
        assert isinstance(info["head"], list)
        assert len(info["head"]) <= 3


# ------------------------------------------------------------------
# ID versioning
# ------------------------------------------------------------------


class TestIDVersioning:
    """Test automatic versioning on name collision."""

    def test_auto_version_on_collision(self, ctx, sample_df):
        """Storing with the same name auto-increments version."""
        r1 = ctx.store_dataset("prices", sample_df)
        r2 = ctx.store_dataset("prices", sample_df)

        assert r1["dataset_id"] == "prices"
        assert r2["dataset_id"] == "prices_v2"

    def test_triple_version(self, ctx, sample_df):
        """Third store with same name gets _v3."""
        ctx.store_dataset("data", sample_df)
        ctx.store_dataset("data", sample_df)
        r3 = ctx.store_dataset("data", sample_df)
        assert r3["dataset_id"] == "data_v3"


# ------------------------------------------------------------------
# Model operations
# ------------------------------------------------------------------


class TestModelOperations:
    """Test store/get model operations."""

    def test_store_and_get_model(self, ctx):
        """Store and retrieve a model."""
        model_data = {"params": {"alpha": 0.05, "beta": 0.9}, "aic": 100.5}
        result = ctx.store_model(
            "garch_test", model_data,
            model_type="GARCH",
            source_dataset="returns",
            metrics={"aic": 100.5, "persistence": 0.95},
        )

        assert result["model_id"] == "garch_test"
        assert result["model_type"] == "GARCH"
        assert "metrics" in result

        retrieved = ctx.get_model("garch_test")
        assert retrieved["params"]["alpha"] == 0.05

    def test_get_missing_model_raises(self, ctx):
        """Getting a missing model raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            ctx.get_model("nonexistent_model")

    def test_list_models(self, ctx):
        """List all stored models."""
        ctx.store_model("model_a", {"data": 1}, model_type="test")
        ctx.store_model("model_b", {"data": 2}, model_type="test")
        models = ctx.list_models()
        assert "model_a" in models
        assert "model_b" in models


# ------------------------------------------------------------------
# Lineage tracking
# ------------------------------------------------------------------


class TestLineage:
    """Test lineage/provenance tracking."""

    def test_lineage_chain(self, ctx, sample_df, returns_df):
        """Lineage traces derivation chain."""
        ctx.store_dataset("prices", sample_df, source_op="fetch")
        ctx.store_dataset(
            "returns", returns_df,
            source_op="compute_returns", parent="prices",
        )

        lineage = ctx.registry.lineage("returns")
        assert lineage == ["prices", "returns"]

    def test_lineage_single_resource(self, ctx, sample_df):
        """Single resource has lineage of just itself."""
        ctx.store_dataset("standalone", sample_df)
        lineage = ctx.registry.lineage("standalone")
        assert lineage == ["standalone"]


# ------------------------------------------------------------------
# Workspace status
# ------------------------------------------------------------------


class TestWorkspaceStatus:
    """Test workspace_status and history."""

    def test_workspace_status(self, ctx, sample_df):
        """Workspace status shows datasets and models."""
        ctx.store_dataset("prices", sample_df)
        ctx.store_model("model", {"x": 1}, model_type="test")

        status = ctx.workspace_status()
        assert "prices" in status["datasets"]
        assert "model" in status["models"]
        assert status["total"] == 2

    def test_history(self, ctx, sample_df):
        """Journal records operations."""
        ctx.store_dataset("prices", sample_df, source_op="fetch")
        ctx.add_note("Test note")

        history = ctx.history(n=10)
        assert len(history) >= 2
        ops = [h["op"] for h in history]
        assert "store_dataset" in ops
        assert "note" in ops

    def test_add_note(self, ctx):
        """Notes are recorded in journal."""
        result = ctx.add_note("Research finding: vol clustering")
        assert result["status"] == "noted"


# ------------------------------------------------------------------
# Sanitization
# ------------------------------------------------------------------


class TestSanitization:
    """Test _sanitize_for_json converts numpy/pandas types."""

    def test_numpy_int(self):
        assert _sanitize_for_json(np.int64(42)) == 42
        assert isinstance(_sanitize_for_json(np.int64(42)), int)

    def test_numpy_float(self):
        assert _sanitize_for_json(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(_sanitize_for_json(np.float64(3.14)), float)

    def test_numpy_array(self):
        result = _sanitize_for_json(np.array([1.0, 2.0, 3.0]))
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_numpy_bool(self):
        assert _sanitize_for_json(np.bool_(True)) is True

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2024-01-15")
        result = _sanitize_for_json(ts)
        assert isinstance(result, str)
        assert "2024" in result

    def test_nan_to_none(self):
        assert _sanitize_for_json(np.nan) is None
        assert _sanitize_for_json(float("nan")) is None

    def test_nested_dict(self):
        data = {
            "value": np.float64(1.5),
            "array": np.array([1, 2]),
            "nested": {"x": np.int32(10)},
        }
        result = _sanitize_for_json(data)
        assert result["value"] == 1.5
        assert result["array"] == [1, 2]
        assert result["nested"]["x"] == 10

    def test_list_of_numpy(self):
        data = [np.float64(1.0), np.int64(2), np.bool_(False)]
        result = _sanitize_for_json(data)
        assert result == [1.0, 2, False]
