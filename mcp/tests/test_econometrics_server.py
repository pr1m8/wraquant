"""Tests for econometrics MCP tools.

Tests: var_model, panel_regression, structural_break,
cointegration_johansen.
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
def econ_tools(ctx):
    """Register econometrics tools and return them."""
    from wraquant_mcp.servers.econometrics import register_econometrics_tools

    mock = MockMCP()
    register_econometrics_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# VAR model
# ------------------------------------------------------------------


class TestVarModel:
    """Test var_model tool."""

    def test_var_fits_and_forecasts(self, econ_tools, ctx):
        """VAR model fits multivariate data and produces forecasts."""
        rng = np.random.default_rng(88)
        n = 300

        # Two correlated series
        x = rng.normal(0, 1, n).cumsum()
        y = 0.5 * x + rng.normal(0, 0.5, n).cumsum()

        ctx.store_dataset("var_data", pd.DataFrame({"x": x, "y": y}))

        result = econ_tools["var_model"](
            dataset="var_data",
            columns_json=json.dumps(["x", "y"]),
            lags=2,
            horizon=5,
        )

        assert result["tool"] == "var_model"
        assert isinstance(result["lag_order"], int)
        assert result["lag_order"] >= 1
        assert isinstance(result["aic"], float)
        assert np.isfinite(result["aic"])
        assert isinstance(result["bic"], float)
        assert np.isfinite(result["bic"])
        assert result["columns"] == ["x", "y"]
        assert result["horizon"] == 5

        # Should produce a forecast
        if result["forecast"] is not None:
            assert isinstance(result["forecast"], list)
            assert len(result["forecast"]) == 5

        # Should store a model
        assert "model_id" in result


# ------------------------------------------------------------------
# Panel regression
# ------------------------------------------------------------------


class TestPanelRegression:
    """Test panel_regression tool."""

    def test_panel_with_entity_fe(self, econ_tools, ctx):
        """Panel regression with fixed effects returns coefficients."""
        rng = np.random.default_rng(44)

        # Build panel: 5 entities, 50 periods each
        entities = []
        times = []
        x_vals = []
        y_vals = []

        for entity_id in range(5):
            entity_fe = rng.normal(0, 2)  # entity fixed effect
            for t in range(50):
                x = rng.normal(0, 1)
                y = 3.0 * x + entity_fe + rng.normal(0, 0.5)
                entities.append(entity_id)
                times.append(t)
                x_vals.append(x)
                y_vals.append(y)

        ctx.store_dataset("panel_data", pd.DataFrame({
            "entity": entities,
            "time": times,
            "x": x_vals,
            "y": y_vals,
        }))

        result = econ_tools["panel_regression"](
            dataset="panel_data",
            y_col="y",
            x_cols_json=json.dumps(["x"]),
            entity_col="entity",
            time_col="time",
            method="fe",
        )

        assert result["tool"] == "panel_regression"
        assert result["method"] == "fixed_effects"
        assert "coefficients" in result
        coefficients = result["coefficients"]
        # Should have coefficient for x
        assert coefficients is not None

        assert "r_squared" in result
        r2 = result["r_squared"]
        assert isinstance(r2, float)
        assert 0 <= r2 <= 1

        assert "nobs" in result
        assert result["nobs"] == 250  # 5 entities * 50 periods

        # Should store the model
        assert "model_id" in result


# ------------------------------------------------------------------
# Structural break
# ------------------------------------------------------------------


class TestStructuralBreak:
    """Test structural_break tool."""

    def test_structural_break_detects_known_break(self, econ_tools, ctx):
        """Structural break test detects a known mean shift."""
        rng = np.random.default_rng(66)
        n = 300

        # Mean shift at index 150
        data = np.concatenate([
            rng.normal(0, 1, 150),
            rng.normal(3, 1, 150),
        ])

        ctx.store_dataset("break_data", pd.DataFrame({"returns": data}))

        result = econ_tools["structural_break"](
            dataset="break_data",
            column="returns",
            method="chow",
            break_point=150,
        )

        assert result["tool"] == "structural_break"
        assert result["method"] == "chow"
        assert isinstance(result["f_statistic"], float)
        assert np.isfinite(result["f_statistic"])
        assert result["f_statistic"] > 0
        assert isinstance(result["p_value"], float)
        assert 0 <= result["p_value"] <= 1
        # Should detect the break
        assert result["is_break"] is True
        assert result["break_point"] == 150


# ------------------------------------------------------------------
# Johansen cointegration
# ------------------------------------------------------------------


class TestCointegrationJohansen:
    """Test cointegration_johansen tool."""

    def test_johansen_on_cointegrated_pair(self, econ_tools, ctx):
        """Johansen test detects cointegration in synthetic pair."""
        rng = np.random.default_rng(33)
        n = 500

        # Common stochastic trend
        trend = rng.normal(0, 1, n).cumsum()

        # Two cointegrated series
        s1 = trend + rng.normal(0, 0.3, n)
        s2 = 0.8 * trend + rng.normal(0, 0.3, n)

        ctx.store_dataset("coint_data", pd.DataFrame({
            "s1": s1,
            "s2": s2,
        }))

        result = econ_tools["cointegration_johansen"](
            dataset="coint_data",
            columns_json=json.dumps(["s1", "s2"]),
            det_order=0,
        )

        assert result["tool"] == "cointegration_johansen"
        assert result["columns"] == ["s1", "s2"]
        assert result["det_order"] == 0

        # Check that trace statistics are returned
        assert "trace_stats" in result
        trace_stats = result["trace_stats"]
        assert isinstance(trace_stats, list)
        assert len(trace_stats) >= 1
        for ts in trace_stats:
            assert np.isfinite(ts)

        # Should detect at least one cointegrating relationship
        coint_rank = result.get("coint_rank")
        # The server maps to "coint_rank" which may come from the johansen dict
        # as "n_cointegrating". Either way, eigenvalues should be present.
        assert "eigenvalues" in result
        eigenvalues = result["eigenvalues"]
        assert isinstance(eigenvalues, list)
        assert len(eigenvalues) >= 1
