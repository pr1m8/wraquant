"""Tests for causal inference MCP tools.

Tests: granger_causality, event_study, diff_in_diff,
instrumental_variable, regression_discontinuity, mediation_analysis,
synthetic_control.
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
def causal_tools(ctx):
    """Register causal tools and return them."""
    from wraquant_mcp.servers.causal import register_causal_tools

    mock = MockMCP()
    register_causal_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# Granger causality
# ------------------------------------------------------------------


class TestGrangerCausality:
    """Test granger_causality tool."""

    def test_granger_with_correlated_series(self, causal_tools, ctx):
        """Granger causality on two correlated series returns valid result."""
        rng = np.random.default_rng(123)
        n = 300

        # x Granger-causes y: y_t depends on x_{t-1}
        x = rng.normal(0, 1, n).cumsum()
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * x[t - 1] + rng.normal(0, 0.5)

        ctx.store_dataset("series_a", pd.DataFrame({"close": x}))
        ctx.store_dataset("series_b", pd.DataFrame({"close": y}))

        result = causal_tools["granger_causality"](
            dataset_a="series_a",
            dataset_b="series_b",
            col_a="close",
            col_b="close",
            max_lag=5,
        )

        assert result["tool"] == "granger_causality"
        assert isinstance(result["f_statistic"], float)
        assert np.isfinite(result["f_statistic"])
        assert isinstance(result["p_value"], float)
        assert 0 <= result["p_value"] <= 1
        assert isinstance(result["optimal_lag"], int)
        assert result["optimal_lag"] >= 1
        assert isinstance(result["reject"], bool)
        assert "direction" in result
        # With this strong causal structure, we expect rejection
        assert result["reject"] is True


# ------------------------------------------------------------------
# Event study
# ------------------------------------------------------------------


class TestEventStudy:
    """Test event_study tool."""

    def test_event_study_with_synthetic_event(self, causal_tools, ctx):
        """Event study around a synthetic shock returns valid CAR."""
        rng = np.random.default_rng(99)
        n = 300
        returns = rng.normal(0, 0.01, n)
        # Inject a positive shock at index 150
        returns[150] = 0.10

        ctx.store_dataset(
            "event_returns",
            pd.DataFrame({"returns": returns}),
        )

        result = causal_tools["event_study"](
            dataset="event_returns",
            column="returns",
            event_dates_json=json.dumps(["150"]),
            estimation_window=60,
            event_window=5,
        )

        assert result["tool"] == "event_study"
        assert isinstance(result["car"], float)
        assert np.isfinite(result["car"])
        assert isinstance(result["car_t_stat"], float)
        assert np.isfinite(result["car_t_stat"])
        assert isinstance(result["car_p_value"], float)
        assert 0 <= result["car_p_value"] <= 1
        assert result["n_events"] >= 1
        assert isinstance(result["estimation_alpha"], float)
        assert isinstance(result["estimation_beta"], float)


# ------------------------------------------------------------------
# Diff-in-diff
# ------------------------------------------------------------------


class TestDiffInDiff:
    """Test diff_in_diff tool."""

    def test_did_with_treatment_control(self, causal_tools, ctx):
        """DID detects a treatment effect in synthetic data."""
        rng = np.random.default_rng(7)
        n = 400

        treatment = np.array([0] * 200 + [1] * 200)
        post = np.array(([0] * 100 + [1] * 100) * 2)

        # Outcome: treatment effect of +5 in the post-treatment group
        outcome = (
            10.0
            + 2.0 * post
            + 3.0 * treatment
            + 5.0 * (treatment * post)
            + rng.normal(0, 1, n)
        )

        ctx.store_dataset("did_data", pd.DataFrame({
            "outcome": outcome,
            "treatment": treatment.astype(float),
            "post": post.astype(float),
        }))

        result = causal_tools["diff_in_diff"](
            dataset="did_data",
            treatment_col="treatment",
            post_col="post",
            outcome_col="outcome",
        )

        assert result["tool"] == "diff_in_diff"
        assert isinstance(result["ate"], float)
        assert np.isfinite(result["ate"])
        # True ATE is 5.0; should be close
        assert abs(result["ate"] - 5.0) < 2.0
        assert isinstance(result["se"], float)
        assert result["se"] > 0
        assert isinstance(result["t_stat"], float)
        assert isinstance(result["p_value"], float)
        assert 0 <= result["p_value"] <= 1
        # Should be statistically significant
        assert result["p_value"] < 0.05
        assert isinstance(result["pre_treatment_mean"], float)
        assert isinstance(result["post_treatment_mean"], float)
        assert isinstance(result["pre_control_mean"], float)
        assert isinstance(result["post_control_mean"], float)


# ------------------------------------------------------------------
# Instrumental variable
# ------------------------------------------------------------------


class TestInstrumentalVariable:
    """Test instrumental_variable tool."""

    def test_iv_basic(self, causal_tools, ctx):
        """IV estimation with a valid instrument returns expected keys."""
        rng = np.random.default_rng(55)
        n = 500

        # z -> x -> y with confounding
        z = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)  # confounder
        x = 0.8 * z + 0.5 * u + rng.normal(0, 0.3, n)
        y = 2.0 * x + u + rng.normal(0, 0.5, n)

        ctx.store_dataset("iv_data", pd.DataFrame({
            "y": y,
            "x": x,
            "z": z,
        }))

        result = causal_tools["instrumental_variable"](
            dataset="iv_data",
            y_col="y",
            x_col="x",
            instrument_col="z",
        )

        assert result["tool"] == "instrumental_variable"
        assert isinstance(result["coefficient"], float)
        assert np.isfinite(result["coefficient"])
        # True coefficient is 2.0; IV should recover approximately
        assert abs(result["coefficient"] - 2.0) < 1.0
        assert isinstance(result["se"], float)
        assert result["se"] > 0
        assert isinstance(result["ci_lower"], float)
        assert isinstance(result["ci_upper"], float)
        assert result["ci_lower"] < result["ci_upper"]
        assert isinstance(result["first_stage_f"], float)
        assert result["first_stage_f"] > 10  # strong instrument
        assert isinstance(result["n_obs"], int)
        assert result["n_obs"] == n


# ------------------------------------------------------------------
# Regression discontinuity
# ------------------------------------------------------------------


class TestRegressionDiscontinuity:
    """Test regression_discontinuity tool."""

    def test_rd_around_cutoff(self, causal_tools, ctx):
        """RD detects a jump at the cutoff in synthetic data."""
        rng = np.random.default_rng(77)
        n = 500

        running = rng.uniform(-2, 2, n)
        treatment_effect = 3.0
        outcome = (
            1.0 * running
            + treatment_effect * (running >= 0).astype(float)
            + rng.normal(0, 0.5, n)
        )

        ctx.store_dataset("rd_data", pd.DataFrame({
            "running": running,
            "outcome": outcome,
        }))

        result = causal_tools["regression_discontinuity"](
            dataset="rd_data",
            running_col="running",
            outcome_col="outcome",
            cutoff=0.0,
        )

        assert result["tool"] == "regression_discontinuity"
        assert isinstance(result["ate"], float)
        assert np.isfinite(result["ate"])
        # Should detect the ~3.0 treatment effect
        assert abs(result["ate"] - 3.0) < 2.0
        assert isinstance(result["se"], float)
        assert result["se"] > 0
        assert isinstance(result["t_stat"], float)
        assert isinstance(result["p_value"], float)
        assert 0 <= result["p_value"] <= 1
        assert isinstance(result["n_treated"], int)
        assert isinstance(result["n_control"], int)
        assert result["n_treated"] > 0
        assert result["n_control"] > 0
        assert result["cutoff"] == 0.0


# ------------------------------------------------------------------
# Mediation analysis
# ------------------------------------------------------------------


class TestMediationAnalysis:
    """Test mediation_analysis tool."""

    def test_mediation_basic(self, causal_tools, ctx):
        """Mediation analysis decomposes total into direct + indirect."""
        rng = np.random.default_rng(33)
        n = 500

        treatment = rng.normal(0, 1, n)
        # Path a: treatment -> mediator
        mediator = 0.6 * treatment + rng.normal(0, 0.5, n)
        # Path b: mediator -> outcome; path c': treatment -> outcome
        outcome = 0.4 * treatment + 0.5 * mediator + rng.normal(0, 0.5, n)

        ctx.store_dataset("mediation_data", pd.DataFrame({
            "treatment": treatment,
            "mediator": mediator,
            "outcome": outcome,
        }))

        result = causal_tools["mediation_analysis"](
            dataset="mediation_data",
            treatment_col="treatment",
            mediator_col="mediator",
            outcome_col="outcome",
        )

        assert result["tool"] == "mediation_analysis"
        assert isinstance(result["total_effect"], float)
        assert isinstance(result["direct_effect"], float)
        assert isinstance(result["indirect_effect"], float)
        assert np.isfinite(result["total_effect"])
        assert np.isfinite(result["direct_effect"])
        assert np.isfinite(result["indirect_effect"])

        # Total should approximately equal direct + indirect
        assert abs(
            result["total_effect"]
            - (result["direct_effect"] + result["indirect_effect"])
        ) < 0.2

        assert isinstance(result["sobel_stat"], float)
        assert isinstance(result["sobel_p"], float)
        assert 0 <= result["sobel_p"] <= 1
        assert isinstance(result["proportion_mediated"], float)
        assert isinstance(result["path_a"], float)
        assert isinstance(result["path_b"], float)


# ------------------------------------------------------------------
# Synthetic control
# ------------------------------------------------------------------


class TestSyntheticControl:
    """Test synthetic_control tool."""

    def test_synthetic_control_basic(self, causal_tools, ctx):
        """Synthetic control with treated + donor units returns valid output."""
        rng = np.random.default_rng(11)
        pre = 60
        post = 30
        total = pre + post

        # Common factor
        factor = rng.normal(0, 1, total).cumsum()

        # Treated unit: follows factor, with treatment effect post-intervention
        treated = factor + rng.normal(0, 0.5, total)
        treated[pre:] += 5.0  # treatment effect

        # Donor units: follow factor without treatment
        donor1 = 0.5 * factor + rng.normal(0, 0.3, total)
        donor2 = 0.8 * factor + rng.normal(0, 0.4, total)
        donor3 = 0.3 * factor + rng.normal(0, 0.2, total)

        ctx.store_dataset("treated", pd.DataFrame({"value": treated}))
        ctx.store_dataset("donor1", pd.DataFrame({"value": donor1}))
        ctx.store_dataset("donor2", pd.DataFrame({"value": donor2}))
        ctx.store_dataset("donor3", pd.DataFrame({"value": donor3}))

        result = causal_tools["synthetic_control"](
            treated_dataset="treated",
            donor_datasets_json=json.dumps(["donor1", "donor2", "donor3"]),
            pre_period=pre,
            post_period=post,
        )

        assert result["tool"] == "synthetic_control"
        assert isinstance(result["ate"], float)
        assert np.isfinite(result["ate"])
        # Should detect positive treatment effect
        assert result["ate"] > 0
        assert isinstance(result["weights"], list)
        assert len(result["weights"]) == 3
        # Weights should sum to approximately 1
        assert abs(sum(result["weights"]) - 1.0) < 0.1
        assert isinstance(result["pre_rmse"], float)
        assert result["pre_rmse"] >= 0
