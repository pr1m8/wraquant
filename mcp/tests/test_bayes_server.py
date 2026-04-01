"""Tests for Bayesian inference MCP tools.

Tests: bayesian_sharpe, bayesian_regression, bayesian_changepoint,
bayesian_portfolio, model_comparison_bayesian.
"""

from __future__ import annotations

import json
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
def bayes_tools(ctx):
    """Register Bayesian tools and return them."""
    from wraquant_mcp.servers.bayes import register_bayes_tools

    mock = MockMCP()
    register_bayes_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# Bayesian Sharpe
# ------------------------------------------------------------------


class TestBayesianSharpe:
    """Test bayesian_sharpe tool."""

    def test_bayesian_sharpe_returns_mean_and_ci(self, bayes_tools):
        """Bayesian Sharpe returns posterior mean and credible interval."""
        result = bayes_tools["bayesian_sharpe"](
            dataset="returns",
            column="returns",
            n_samples=2000,
        )

        assert result["tool"] == "bayesian_sharpe"
        assert isinstance(result["posterior_mean"], float)
        assert np.isfinite(result["posterior_mean"])
        assert isinstance(result["posterior_std"], float)
        assert result["posterior_std"] > 0
        assert isinstance(result["credible_interval_95"], list)
        assert len(result["credible_interval_95"]) == 2
        ci_lower, ci_upper = result["credible_interval_95"]
        assert ci_lower < ci_upper
        assert isinstance(result["prob_positive"], float)
        assert 0 <= result["prob_positive"] <= 1
        assert result["n_samples"] == 2000


# ------------------------------------------------------------------
# Bayesian regression
# ------------------------------------------------------------------


class TestBayesianRegression:
    """Test bayesian_regression tool."""

    def test_bayesian_regression_returns_coefficients(self, bayes_tools, ctx):
        """Bayesian regression returns posterior mean coefficients."""
        rng = np.random.default_rng(99)
        n = 200

        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 2.0 * x1 + 0.5 * x2 + rng.normal(0, 0.3, n)

        ctx.store_dataset("reg_data", pd.DataFrame({
            "y": y,
            "x1": x1,
            "x2": x2,
        }))

        result = bayes_tools["bayesian_regression"](
            dataset="reg_data",
            y_column="y",
            x_columns_json=json.dumps(["x1", "x2"]),
        )

        assert result["tool"] == "bayesian_regression"
        assert "posterior_mean" in result
        posterior_mean = result["posterior_mean"]
        assert isinstance(posterior_mean, list)
        assert len(posterior_mean) >= 2
        # All coefficients should be finite
        for coef in posterior_mean:
            assert np.isfinite(coef)

        assert "posterior_std" in result
        for std in result["posterior_std"]:
            assert std > 0

        assert isinstance(result["log_marginal_likelihood"], float)
        assert np.isfinite(result["log_marginal_likelihood"])

        # Should store the model
        assert "model_id" in result


# ------------------------------------------------------------------
# Bayesian changepoint
# ------------------------------------------------------------------


class TestBayesianChangepoint:
    """Test bayesian_changepoint tool."""

    def test_bayesian_changepoint_detects_known_break(self, bayes_tools, ctx):
        """Changepoint detection finds a known mean shift."""
        rng = np.random.default_rng(77)
        n = 300

        # Mean shift at index 150
        data = np.concatenate([
            rng.normal(0, 0.5, 150),
            rng.normal(3, 0.5, 150),
        ])

        ctx.store_dataset("cp_data", pd.DataFrame({"returns": data}))

        result = bayes_tools["bayesian_changepoint"](
            dataset="cp_data",
            column="returns",
            hazard=100.0,
            threshold=0.3,
        )

        assert result["tool"] == "bayesian_changepoint"
        assert isinstance(result["n_changepoints"], int)
        assert result["n_changepoints"] >= 1
        assert isinstance(result["changepoint_indices"], list)
        # At least one detected changepoint should be near index 150
        assert any(
            abs(cp - 150) < 30
            for cp in result["changepoint_indices"]
        ), f"No changepoint near 150: {result['changepoint_indices']}"

        assert result["hazard"] == 100.0
        assert result["threshold"] == 0.3
        # Should store a dataset
        assert "dataset_id" in result


# ------------------------------------------------------------------
# Bayesian portfolio
# ------------------------------------------------------------------


class TestBayesianPortfolio:
    """Test bayesian_portfolio tool."""

    def test_bayesian_portfolio_returns_weights(self, bayes_tools, ctx):
        """Bayesian portfolio returns weight distribution."""
        rng = np.random.default_rng(55)
        n = 200

        # 3-asset returns
        multi = pd.DataFrame({
            "A": rng.normal(0.001, 0.02, n),
            "B": rng.normal(0.0005, 0.015, n),
            "C": rng.normal(0.0008, 0.018, n),
        })
        ctx.store_dataset("multi_returns", multi)

        result = bayes_tools["bayesian_portfolio"](
            dataset="multi_returns",
            n_samples=1000,
        )

        assert result["tool"] == "bayesian_portfolio"
        assert isinstance(result["weights_mean"], list)
        assert len(result["weights_mean"]) == 3
        # Weights should sum to approximately 1
        assert abs(sum(result["weights_mean"]) - 1.0) < 0.2
        # All weights should be finite
        for w in result["weights_mean"]:
            assert np.isfinite(w)

        assert isinstance(result["weights_std"], list)
        assert len(result["weights_std"]) == 3
        for s in result["weights_std"]:
            assert s >= 0

        assert isinstance(result["expected_return"], float)
        assert np.isfinite(result["expected_return"])
        assert isinstance(result["expected_risk"], float)
        assert result["expected_risk"] > 0
        assert result["assets"] == ["A", "B", "C"]


# ------------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------------


class TestModelComparison:
    """Test model_comparison_bayesian tool."""

    def test_model_comparison_ranks_models(self, bayes_tools, ctx):
        """Model comparison ranks candidate models by marginal likelihood."""
        rng = np.random.default_rng(22)
        n = 200

        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        x3 = rng.normal(0, 1, n)  # irrelevant noise
        y = 3.0 * x1 + 1.0 * x2 + rng.normal(0, 0.5, n)

        ctx.store_dataset("compare_data", pd.DataFrame({
            "y": y,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }))

        models = [
            {"name": "full", "x_columns": ["x1", "x2"]},
            {"name": "partial", "x_columns": ["x1"]},
            {"name": "noise", "x_columns": ["x3"]},
        ]

        result = bayes_tools["model_comparison_bayesian"](
            dataset="compare_data",
            column="y",
            models_json=json.dumps(models),
        )

        assert result["tool"] == "model_comparison_bayesian"
        assert "ranking" in result
        ranking = result["ranking"]
        assert isinstance(ranking, list)
        assert len(ranking) == 3

        # Each model entry should have a name or identifier
        names = [r.get("model", r.get("name", "")) for r in ranking]
        # The ranking should contain all models
        assert len(names) == 3
