"""Tests for math MCP server tools.

Tests correlation_network, levy_simulate, optimal_stopping,
spectral_analysis via underlying wraquant.math functions.
"""

from __future__ import annotations

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
def multi_asset_returns():
    """Create multi-asset returns for network analysis."""
    np.random.seed(123)
    n = 252
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "AAPL": np.random.randn(n) * 0.02,
            "MSFT": np.random.randn(n) * 0.018,
            "GOOGL": np.random.randn(n) * 0.022,
            "AMZN": np.random.randn(n) * 0.025,
        },
        index=dates,
    )


@pytest.fixture
def prices_df():
    """Create synthetic price data for spectral/stopping analysis."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    return pd.DataFrame({"close": close}, index=dates)


class TestMathServer:
    """Test math MCP tool functions via underlying wraquant.math."""

    def test_correlation_network(self, ctx, multi_asset_returns):
        """correlation_network builds a graph from multi-asset returns."""
        from wraquant.math.network import (
            centrality_measures,
            correlation_network as _corr_net,
        )

        ctx.store_dataset("returns", multi_asset_returns)
        df = ctx.get_dataset("returns")
        returns = df.select_dtypes(include=[np.number]).dropna()

        threshold = 0.5
        net = _corr_net(returns, threshold=threshold)
        centrality = centrality_measures(net["adjacency"])

        output = _sanitize_for_json({
            "tool": "correlation_network",
            "dataset": "returns",
            "threshold": threshold,
            "n_assets": len(net["asset_names"]),
            "labels": net["asset_names"],
            "n_edges": int((net["adjacency"] != 0).sum() // 2),
            "centrality": {
                label: float(centrality["degree"][i])
                for i, label in enumerate(net["asset_names"])
            },
        })

        assert output["tool"] == "correlation_network"
        assert output["n_assets"] == 4
        assert isinstance(output["labels"], list)
        assert "AAPL" in output["labels"]
        assert "MSFT" in output["labels"]
        assert isinstance(output["n_edges"], int)
        assert output["n_edges"] >= 0
        assert isinstance(output["centrality"], dict)
        for asset in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
            assert asset in output["centrality"]
            assert isinstance(output["centrality"][asset], float)

    def test_levy_simulate(self, ctx):
        """levy_simulate produces a Variance Gamma path."""
        from wraquant.math.levy import variance_gamma_simulate

        path = variance_gamma_simulate(
            sigma=0.2, nu=0.5, theta=-0.1, n_steps=1000, seed=42,
        )

        path_df = pd.DataFrame({"path": path})
        stored = ctx.store_dataset("levy_vg", path_df, source_op="levy_simulate")

        increments = np.diff(path)

        output = _sanitize_for_json({
            "tool": "levy_simulate",
            "model": "variance_gamma",
            "n_steps": 1000,
            "final_value": float(path[-1]),
            "max_value": float(path.max()),
            "min_value": float(path.min()),
            "increment_stats": {
                "mean": float(increments.mean()),
                "std": float(increments.std()),
            },
            **stored,
        })

        assert output["tool"] == "levy_simulate"
        assert output["model"] == "variance_gamma"
        assert output["n_steps"] == 1000
        assert isinstance(output["final_value"], float)
        assert isinstance(output["max_value"], float)
        assert isinstance(output["min_value"], float)
        assert output["max_value"] >= output["min_value"]
        assert "increment_stats" in output
        assert isinstance(output["increment_stats"]["mean"], float)
        assert isinstance(output["increment_stats"]["std"], float)
        assert output["increment_stats"]["std"] > 0
        assert output["rows"] == len(path)  # stored dataset rows
        assert "dataset_id" in output

    def test_optimal_stopping_cusum(self, ctx):
        """optimal_stopping detects a change via CUSUM."""
        from wraquant.math.optimal_stopping import cusum_stopping

        # Create data with a mean shift at index 50
        np.random.seed(42)
        obs = np.concatenate([
            np.random.randn(50) * 0.01,
            np.random.randn(50) * 0.01 + 0.05,
        ])
        prices = pd.DataFrame({"close": obs})
        ctx.store_dataset("shift_data", prices)

        series = ctx.get_dataset("shift_data")["close"].dropna()
        target_mean = float(series.mean())
        threshold = 0.2

        result = cusum_stopping(
            observations=series.values,
            target_mean=target_mean,
            threshold=threshold,
        )

        output = _sanitize_for_json({
            "tool": "optimal_stopping",
            "dataset": "shift_data",
            "column": "close",
            "method": "cusum",
            "threshold": threshold,
            "target_mean": target_mean,
            **result,
        })

        assert output["tool"] == "optimal_stopping"
        assert output["method"] == "cusum"
        assert isinstance(output["threshold"], float)
        assert isinstance(output["target_mean"], float)
        assert "detected" in output
        assert isinstance(output["detected"], bool)
        assert "stopping_time" in output
        assert isinstance(output["stopping_time"], int)

    def test_spectral_analysis(self, ctx):
        """spectral_analysis returns frequencies and entropy."""
        from wraquant.math.spectral import (
            dominant_frequencies,
            fft_decompose,
            spectral_entropy,
        )

        # Create a signal with a clear cycle
        t = np.arange(256)
        signal = np.sin(2 * np.pi * t / 21) + 0.5 * np.sin(2 * np.pi * t / 63)
        signal_df = pd.DataFrame({"close": signal})
        ctx.store_dataset("cyclic_data", signal_df)

        series = ctx.get_dataset("cyclic_data")["close"].dropna()

        fft_result = fft_decompose(series.values)
        dominant = dominant_frequencies(series.values)
        entropy = spectral_entropy(series.values)

        output = _sanitize_for_json({
            "tool": "spectral_analysis",
            "dataset": "cyclic_data",
            "column": "close",
            "spectral_entropy": float(entropy),
            "dominant_frequencies": dominant["frequency"].tolist(),
            "dominant_periods": (1.0 / dominant["frequency"][dominant["frequency"] > 0]).tolist(),
            "dominant_amplitudes": dominant["amplitude"].tolist(),
            "observations": len(series),
        })

        assert output["tool"] == "spectral_analysis"
        assert isinstance(output["spectral_entropy"], float)
        assert 0.0 <= output["spectral_entropy"] <= 1.0
        assert isinstance(output["dominant_frequencies"], list)
        assert len(output["dominant_frequencies"]) > 0
        assert isinstance(output["dominant_periods"], list)
        assert len(output["dominant_periods"]) > 0
        assert isinstance(output["dominant_amplitudes"], list)
        assert output["observations"] == 256
        # Entropy should be relatively low for a periodic signal
        assert output["spectral_entropy"] < 0.5
