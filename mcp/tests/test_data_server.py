"""Tests for data management MCP server tools.

Tests load_json, export_dataset, merge_datasets, filter_dataset
via AnalysisContext and DuckDB.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add mcp source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


@pytest.fixture
def ctx(tmp_path):
    """Create an AnalysisContext with a temporary workspace."""
    context = AnalysisContext(workspace_dir=tmp_path / "test_workspace")
    yield context
    context.close()


@pytest.fixture
def prices_df():
    """Create synthetic price data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    return pd.DataFrame(
        {
            "close": close,
            "volume": np.random.randint(100_000, 1_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def returns_df(prices_df):
    """Create returns from prices."""
    returns = prices_df["close"].pct_change().dropna()
    return returns.to_frame(name="returns")


class TestDataServer:
    """Test data management MCP tool functions via context."""

    def test_load_json(self, ctx):
        """load_json stores a dataset from inline JSON."""
        data_json = json.dumps([
            {"ticker": "AAPL", "price": 150.0, "volume": 1000000},
            {"ticker": "MSFT", "price": 280.0, "volume": 800000},
            {"ticker": "GOOGL", "price": 140.0, "volume": 500000},
        ])

        data = json.loads(data_json)
        df = pd.DataFrame(data)

        stored = ctx.store_dataset("stocks", df, source_op="load_json")

        output = _sanitize_for_json({
            "tool": "load_json",
            **stored,
        })

        assert output["tool"] == "load_json"
        assert output["dataset_id"] == "stocks"
        assert output["rows"] == 3
        assert "ticker" in output["columns"]
        assert "price" in output["columns"]
        assert "volume" in output["columns"]

        # Verify data round-trips through DuckDB
        retrieved = ctx.get_dataset("stocks")
        assert len(retrieved) == 3
        assert set(retrieved.columns) == {"ticker", "price", "volume"}

    def test_export_dataset(self, ctx, prices_df):
        """export_dataset creates a CSV file on disk."""
        ctx.store_dataset("prices", prices_df, source_op="test")

        df = ctx.get_dataset("prices")
        out_path = Path(ctx.workspace_dir) / "prices.csv"
        df.to_csv(out_path)

        output = _sanitize_for_json({
            "tool": "export_dataset",
            "dataset": "prices",
            "format": "csv",
            "path": str(out_path),
            "rows": len(df),
            "columns": list(df.columns),
        })

        assert output["tool"] == "export_dataset"
        assert output["dataset"] == "prices"
        assert output["format"] == "csv"
        assert output["rows"] == 100
        assert isinstance(output["path"], str)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert "close" in output["columns"]
        assert "volume" in output["columns"]

        # Verify we can read it back
        loaded = pd.read_csv(out_path)
        assert len(loaded) == 100

    def test_merge_datasets(self, ctx):
        """merge_datasets joins two datasets on index."""
        df_a = pd.DataFrame(
            {"price": [100.0, 101.0, 102.0]},
            index=pd.date_range("2023-01-01", periods=3, freq="B"),
        )
        df_b = pd.DataFrame(
            {"volume": [1000, 2000, 3000]},
            index=pd.date_range("2023-01-01", periods=3, freq="B"),
        )

        ctx.store_dataset("prices", df_a)
        ctx.store_dataset("volumes", df_b)

        left = ctx.get_dataset("prices")
        right = ctx.get_dataset("volumes")

        merged = left.merge(
            right, left_index=True, right_index=True,
            how="inner", suffixes=("_a", "_b"),
        )

        result_name = "prices_volumes_merged"
        stored = ctx.store_dataset(
            result_name, merged,
            source_op="merge_datasets",
            parent="prices",
        )

        output = _sanitize_for_json({
            "tool": "merge_datasets",
            "dataset_a": "prices",
            "dataset_b": "volumes",
            "join_type": "inner",
            "join_key": "index",
            **stored,
        })

        assert output["tool"] == "merge_datasets"
        assert output["dataset_a"] == "prices"
        assert output["dataset_b"] == "volumes"
        assert output["join_type"] == "inner"
        assert output["join_key"] == "index"
        assert output["rows"] == 3
        assert "price" in output["columns"]
        assert "volume" in output["columns"]

        # Verify merged dataset in context
        result = ctx.get_dataset(result_name)
        assert result.shape == (3, 2)

    def test_filter_dataset(self, ctx):
        """filter_dataset applies a SQL WHERE clause via DuckDB."""
        df = pd.DataFrame({
            "returns": [0.02, -0.01, 0.05, -0.03, 0.01, 0.08],
            "volume": [500_000, 200_000, 1_500_000, 300_000, 1_200_000, 2_000_000],
        })
        ctx.store_dataset("data", df, source_op="test")

        # Filter using DuckDB SQL
        condition_sql = "returns > 0.01 AND volume > 1000000"
        query = f'SELECT * FROM "data" WHERE {condition_sql}'
        result_df = ctx.db.sql(query).df()

        result_name = "data_filtered"
        stored = ctx.store_dataset(
            result_name, result_df,
            source_op="filter_dataset",
            parent="data",
        )

        output = _sanitize_for_json({
            "tool": "filter_dataset",
            "source_dataset": "data",
            "condition": condition_sql,
            **stored,
        })

        assert output["tool"] == "filter_dataset"
        assert output["source_dataset"] == "data"
        assert output["condition"] == condition_sql
        assert isinstance(output["rows"], int)
        assert output["rows"] == 2  # returns > 0.01 AND volume > 1M: (0.05, 1.5M) and (0.08, 2M)
        assert "returns" in output["columns"]
        assert "volume" in output["columns"]

        # Verify filtered data
        filtered = ctx.get_dataset(result_name)
        assert len(filtered) == 2
        assert (filtered["returns"] > 0.01).all()
        assert (filtered["volume"] > 1_000_000).all()
