"""Tests for the ToolAdaptor — wraquant function wrapping for MCP.

Tests: wrap() resolves dataset IDs, handles missing datasets,
routes different result types, auto-detection of data params.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wraquant_mcp.adaptor import ToolAdaptor, _detect_data_params, _handle_result
from wraquant_mcp.context import AnalysisContext


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path):
    """Create an AnalysisContext with synthetic data."""
    ws = tmp_path / "test_adaptor"
    context = AnalysisContext(str(ws))

    rng = np.random.default_rng(42)
    n = 252
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + rng.normal(0, 1, n).cumsum()

    prices = pd.DataFrame({
        "open": close + rng.normal(0, 0.5, n),
        "high": close + abs(rng.normal(0, 1, n)),
        "low": close - abs(rng.normal(0, 1, n)),
        "close": close,
        "volume": rng.integers(1000, 10000, n),
    }, index=dates)
    context.store_dataset("prices", prices)

    returns = prices["close"].pct_change().dropna()
    context.store_dataset(
        "returns", returns.to_frame("returns"), parent="prices",
    )

    yield context
    context.close()
    if ws.exists():
        shutil.rmtree(ws)


@pytest.fixture
def adaptor(ctx):
    """Create a ToolAdaptor."""
    return ToolAdaptor(ctx)


# ------------------------------------------------------------------
# _detect_data_params
# ------------------------------------------------------------------


class TestDetectDataParams:
    """Test auto-detection of data parameter names."""

    def test_detects_returns_param(self):
        def func(returns, period=14):
            pass
        params = _detect_data_params(func)
        assert "returns" in params
        assert "period" not in params

    def test_detects_benchmark_param(self):
        def func(returns, benchmark, window=60):
            pass
        params = _detect_data_params(func)
        assert "returns" in params
        assert "benchmark" in params
        assert "window" not in params

    def test_detects_prices_param(self):
        def func(prices, method="close"):
            pass
        params = _detect_data_params(func)
        assert "prices" in params

    def test_detects_data_param(self):
        def func(data, n_components=3):
            pass
        params = _detect_data_params(func)
        assert "data" in params

    def test_detects_ohlcv_params(self):
        def func(open, high, low, close, volume, window=21):
            pass
        params = _detect_data_params(func)
        for p in ["open", "high", "low", "close", "volume"]:
            assert p in params
        assert "window" not in params

    def test_detects_y_X_params(self):
        def func(y, X, add_constant=True):
            pass
        params = _detect_data_params(func)
        assert "y" in params
        assert "X" in params
        assert "add_constant" not in params

    def test_detects_factors_param(self):
        def func(returns, factors, n_factors=3):
            pass
        params = _detect_data_params(func)
        assert "returns" in params
        assert "factors" in params

    def test_no_data_params(self):
        def func(name, value, option=True):
            pass
        params = _detect_data_params(func)
        assert params == []

    def test_detects_observations_param(self):
        def func(observations, alpha=0.05):
            pass
        params = _detect_data_params(func)
        assert "observations" in params


# ------------------------------------------------------------------
# wrap() — dataset resolution
# ------------------------------------------------------------------


class TestWrapDatasetResolution:
    """Test that wrap() correctly resolves dataset IDs from context."""

    def test_resolves_dataset_id_to_dataframe(self, adaptor, ctx):
        """When a data param receives a dataset ID, it should resolve to DataFrame."""
        def compute_mean(returns):
            return float(returns["returns"].mean())

        wrapped = adaptor.wrap(compute_mean, "test")
        result = wrapped(returns="returns")

        assert "result" in result
        assert isinstance(result["result"], float)

    def test_resolves_multiple_datasets(self, adaptor, ctx):
        """Multiple data params should all resolve."""
        def compute_corr(returns, data):
            return float(returns["returns"].corr(data["returns"]))

        # Store a second dataset
        rng = np.random.default_rng(99)
        other = pd.DataFrame({"returns": rng.normal(0, 0.01, 251)})
        ctx.store_dataset("other_returns", other)

        wrapped = adaptor.wrap(compute_corr, "test")
        result = wrapped(returns="returns", data="other_returns")
        assert "result" in result
        assert isinstance(result["result"], float)

    def test_passes_through_non_data_params(self, adaptor, ctx):
        """Non-data params (period, window, etc.) should pass through unchanged."""
        def compute_rolling(returns, window=20):
            return returns["returns"].rolling(window).mean()

        wrapped = adaptor.wrap(compute_rolling, "test")
        result = wrapped(returns="returns", window=10)
        assert "dataset_id" in result  # Series result gets stored

    def test_handles_missing_dataset_gracefully(self, adaptor, ctx):
        """If dataset doesn't exist, the string passes through to the function."""
        def identity(returns):
            return returns  # Just return what was passed

        wrapped = adaptor.wrap(identity, "test")
        result = wrapped(returns="nonexistent_dataset")

        # The string "nonexistent_dataset" should be passed through
        # since it doesn't exist in context
        assert "result" in result
        assert result["result"] == "nonexistent_dataset"

    def test_explicit_data_params(self, adaptor, ctx):
        """Test explicit data_params override."""
        def custom_func(my_data, period=14):
            return float(my_data["returns"].mean())

        wrapped = adaptor.wrap(custom_func, "test", data_params=["my_data"])
        # Store data under a name the function expects
        rng = np.random.default_rng(42)
        ctx.store_dataset("my_data", pd.DataFrame({"returns": rng.normal(0, 0.01, 100)}))

        result = wrapped(my_data="my_data")
        assert "result" in result
        assert isinstance(result["result"], float)


# ------------------------------------------------------------------
# wrap() — model resolution
# ------------------------------------------------------------------


class TestWrapModelResolution:
    """Test that wrap() resolves model IDs from context."""

    def test_resolves_model_id(self, adaptor, ctx):
        """Model params should resolve to stored model objects."""
        model_data = {"params": {"alpha": 0.05}, "aic": -100}
        ctx.store_model("test_model", model_data, model_type="test")

        def use_model(model, horizon=10):
            return {"aic": model["aic"], "horizon": horizon}

        wrapped = adaptor.wrap(use_model, "test", model_params=["model"])
        result = wrapped(model="test_model", horizon=5)
        assert result["result"]["aic"] == -100
        assert result["result"]["horizon"] == 5

    def test_missing_model_passes_string(self, adaptor, ctx):
        """Missing model ID should pass the string through."""
        def use_model(model):
            return str(model)

        wrapped = adaptor.wrap(use_model, "test", model_params=["model"])
        result = wrapped(model="nonexistent_model")
        assert "result" in result


# ------------------------------------------------------------------
# wrap() — error handling
# ------------------------------------------------------------------


class TestWrapErrorHandling:
    """Test that wrap() handles errors gracefully."""

    def test_function_exception_returns_error(self, adaptor, ctx):
        def failing_func(returns):
            raise ValueError("Something went wrong")

        wrapped = adaptor.wrap(failing_func, "test")
        result = wrapped(returns="returns")
        assert "error" in result
        assert result["error"] == "ValueError"
        assert "Something went wrong" in result["message"]

    def test_error_includes_tool_name(self, adaptor, ctx):
        def bad_func(returns):
            raise RuntimeError("crash")

        wrapped = adaptor.wrap(bad_func, "mymod")
        result = wrapped(returns="returns")
        assert result["tool"] == "mymod.bad_func"


# ------------------------------------------------------------------
# _handle_result — result type routing
# ------------------------------------------------------------------


class TestHandleResult:
    """Test _handle_result routes different types correctly."""

    def test_handle_scalar_int(self, ctx):
        result = _handle_result(ctx, 42, "func", "mod", {})
        assert result["result"] == 42.0
        assert isinstance(result["result"], float)

    def test_handle_scalar_float(self, ctx):
        result = _handle_result(ctx, 3.14, "func", "mod", {})
        assert result["result"] == pytest.approx(3.14)

    def test_handle_numpy_scalar(self, ctx):
        result = _handle_result(ctx, np.float64(2.718), "func", "mod", {})
        assert result["result"] == pytest.approx(2.718)

    def test_handle_series(self, ctx):
        series = pd.Series([1.0, 2.0, 3.0, 4.0], name="test_values")
        result = _handle_result(ctx, series, "func", "mod", {})

        assert "dataset_id" in result
        assert "summary" in result
        assert result["summary"]["mean"] == pytest.approx(2.5)
        assert result["summary"]["min"] == pytest.approx(1.0)
        assert result["summary"]["max"] == pytest.approx(4.0)

        # Verify stored in context
        stored = ctx.get_dataset(result["dataset_id"])
        assert len(stored) == 4

    def test_handle_dataframe(self, ctx):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _handle_result(ctx, df, "func", "mod", {})

        assert "dataset_id" in result
        stored = ctx.get_dataset(result["dataset_id"])
        assert stored.shape == (2, 2)

    def test_handle_scalar_dict(self, ctx):
        d = {"sharpe": 1.5, "sortino": 2.0, "max_dd": -0.15}
        result = _handle_result(ctx, d, "func", "mod", {})

        assert "result" in result
        assert result["result"]["sharpe"] == 1.5
        assert result["result"]["sortino"] == 2.0

    def test_handle_dict_with_series_values(self, ctx):
        d = {
            "upper": pd.Series([1.0, 2.0, 3.0]),
            "lower": pd.Series([0.5, 1.0, 1.5]),
            "period": 14,
        }
        result = _handle_result(ctx, d, "func", "mod", {})

        # Series values should be stored as dataset
        assert "dataset_id" in result
        # Scalar values should be included
        assert result.get("period") == 14

    def test_handle_string_fallback(self, ctx):
        result = _handle_result(ctx, "some string result", "func", "mod", {})
        assert result["result"] == "some string result"

    def test_handle_dataclass_with_persistence(self, ctx):
        """Test that model-like results get stored."""
        class MockGARCH:
            def __init__(self):
                self.persistence = 0.95
                self.aic = -1500.0
                self.bic = -1490.0
                self.conditional_volatility = np.array([0.01, 0.02, 0.015])

        model = MockGARCH()
        result = _handle_result(ctx, model, "garch_fit", "vol", {})

        assert "model_id" in result
        assert "metrics" in result
        assert result["metrics"]["persistence"] == 0.95

        # Conditional volatility should also be stored as dataset
        datasets = ctx.list_datasets()
        assert any("conditional_volatility" in d for d in datasets)

    def test_handle_dataclass_with_states(self, ctx):
        """Test regime result with states array."""
        class MockRegime:
            def __init__(self):
                self.n_regimes = 2
                self.states = np.array([0, 1, 0, 1, 0])
                self.probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])

        result = _handle_result(ctx, MockRegime(), "detect", "regimes", {})
        assert "model_id" in result

        # States should be stored as dataset
        datasets = ctx.list_datasets()
        assert any("states" in d for d in datasets)


# ------------------------------------------------------------------
# Full wrap() pipeline
# ------------------------------------------------------------------


class TestFullPipeline:
    """Test complete wrap → call → store pipeline."""

    def test_wrap_function_returning_series(self, adaptor, ctx):
        def rolling_mean(returns, window=20):
            return returns["returns"].rolling(window).mean()

        wrapped = adaptor.wrap(rolling_mean, "stats")
        result = wrapped(returns="returns", window=10)

        assert "dataset_id" in result
        assert "summary" in result
        stored = ctx.get_dataset(result["dataset_id"])
        assert len(stored) > 0

    def test_wrap_function_returning_dict(self, adaptor, ctx):
        def compute_stats(returns):
            r = returns["returns"]
            return {
                "mean": float(r.mean()),
                "std": float(r.std()),
                "skew": float(r.skew()),
            }

        wrapped = adaptor.wrap(compute_stats, "stats")
        result = wrapped(returns="returns")

        assert "result" in result
        assert "mean" in result["result"]
        assert "std" in result["result"]
        assert "skew" in result["result"]

    def test_wrapped_function_preserves_name(self, adaptor, ctx):
        def my_analysis(returns):
            return 42.0

        wrapped = adaptor.wrap(my_analysis, "risk")
        assert wrapped.__name__ == "risk_my_analysis"

    def test_wrapped_function_preserves_docstring(self, adaptor, ctx):
        def my_func(returns):
            """This is the docstring."""
            return 1.0

        wrapped = adaptor.wrap(my_func, "test")
        assert wrapped.__doc__ == "This is the docstring."
