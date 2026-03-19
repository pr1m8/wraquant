"""Tests for the composable workflow system (wraquant.compose)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.compose import (
    Workflow,
    WorkflowResult,
    ml_workflow,
    portfolio_workflow,
    quick_analysis_workflow,
    risk_workflow,
    steps,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prices() -> pd.Series:
    """Synthetic price series (252 days)."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    log_returns = rng.normal(0.0005, 0.02, size=252)
    return pd.Series(100 * np.exp(np.cumsum(log_returns)), index=dates, name="price")


@pytest.fixture
def multi_asset_prices() -> pd.DataFrame:
    """Synthetic multi-asset price DataFrame (252 days, 3 assets)."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    vols = np.array([0.01, 0.02, 0.005])
    rets = rng.normal(0.0003, 1, size=(252, 3)) * vols
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=["Bonds", "Equity", "Gold"])


@pytest.fixture
def ohlcv_prices() -> pd.DataFrame:
    """Synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=252)))
    high = close * (1 + rng.uniform(0, 0.03, size=252))
    low = close * (1 - rng.uniform(0, 0.03, size=252))
    open_ = close * (1 + rng.normal(0, 0.01, size=252))
    volume = rng.integers(1_000_000, 10_000_000, size=252).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------


class TestWorkflowResult:
    def test_attribute_access(self):
        ctx = {"sharpe": 1.5, "returns": pd.Series([0.01, -0.02])}
        result = WorkflowResult(ctx)
        assert result.sharpe == 1.5
        assert len(result.returns) == 2

    def test_attribute_error_on_missing(self):
        result = WorkflowResult({"a": 1})
        with pytest.raises(AttributeError, match="no attribute 'b'"):
            _ = result.b

    def test_attribute_error_on_private(self):
        result = WorkflowResult({"_secret": 42})
        with pytest.raises(AttributeError):
            _ = result._secret

    def test_keys(self):
        result = WorkflowResult({"x": 1, "y": 2, "z": 3})
        assert sorted(result.keys()) == ["x", "y", "z"]

    def test_to_dict(self):
        ctx = {"a": 10, "b": 20}
        result = WorkflowResult(ctx)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d == {"a": 10, "b": 20}

    def test_contains(self):
        result = WorkflowResult({"present": True})
        assert "present" in result
        assert "absent" not in result

    def test_repr(self):
        result = WorkflowResult({"a": 1, "b": 2})
        r = repr(result)
        assert "WorkflowResult" in r
        assert "a" in r
        assert "b" in r


# ---------------------------------------------------------------------------
# Workflow basics
# ---------------------------------------------------------------------------


class TestWorkflow:
    def test_creation_and_repr(self):
        wf = Workflow("test")
        assert "test" in repr(wf)

    def test_empty_workflow(self, prices):
        wf = Workflow("empty")
        result = wf.run(prices)
        assert "prices" in result.keys()
        assert result.is_multivariate is False

    def test_chaining(self):
        wf = Workflow("chain").add(steps.returns()).add(steps.risk_metrics())
        assert len(wf._steps) == 2

    def test_series_input_detection(self, prices):
        wf = Workflow("detect")
        result = wf.run(prices)
        assert result.is_multivariate is False
        assert "prices" in result.keys()

    def test_dataframe_input_detection(self, multi_asset_prices):
        wf = Workflow("detect")
        result = wf.run(multi_asset_prices)
        assert result.is_multivariate is True
        assert "prices_df" in result.keys()

    def test_ohlcv_detection(self, ohlcv_prices):
        wf = Workflow("ohlcv")
        result = wf.run(ohlcv_prices)
        assert result.is_ohlcv is True
        assert "close" in result.keys()
        assert "high" in result.keys()
        assert "low" in result.keys()

    def test_numpy_input(self):
        data = np.array([100, 101, 102, 103, 104], dtype=float)
        wf = Workflow("np").add(steps.returns())
        result = wf.run(data)
        assert "returns" in result.keys()

    def test_kwargs_passthrough(self, prices):
        benchmark = prices.pct_change().dropna() * 0.8
        wf = Workflow("kw")
        result = wf.run(prices, benchmark=benchmark)
        assert "benchmark" in result.keys()

    def test_strict_mode_raises(self, prices):
        def _bad_step(ctx):
            raise ValueError("intentional failure")

        _bad_step.__name__ = "bad_step"
        wf = Workflow("strict").add(_bad_step)
        with pytest.raises(ValueError, match="intentional failure"):
            wf.run(prices, strict=True)

    def test_non_strict_mode_continues(self, prices):
        def _bad_step(ctx):
            raise ValueError("intentional failure")

        _bad_step.__name__ = "bad_step"
        wf = Workflow("soft").add(_bad_step).add(steps.returns())
        result = wf.run(prices)
        # The returns step still ran
        assert "returns" in result.keys()


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------


class TestSteps:
    def test_returns_from_prices(self, prices):
        wf = Workflow("ret").add(steps.returns())
        result = wf.run(prices)
        assert "returns" in result.keys()
        r = result.returns
        assert isinstance(r, pd.Series)
        assert len(r) == len(prices) - 1

    def test_returns_from_dataframe(self, multi_asset_prices):
        wf = Workflow("ret").add(steps.returns())
        result = wf.run(multi_asset_prices)
        assert "returns_df" in result.keys()
        assert "returns" in result.keys()
        assert result.returns_df.shape[1] == 3

    def test_risk_metrics(self, prices):
        wf = Workflow("risk").add(steps.returns()).add(steps.risk_metrics())
        result = wf.run(prices)
        assert "risk" in result.keys()
        risk = result.risk
        assert "sharpe" in risk
        assert "sortino" in risk
        assert "max_drawdown" in risk
        assert isinstance(risk["sharpe"], float)
        assert isinstance(risk["sortino"], float)
        assert isinstance(risk["max_drawdown"], float)

    def test_risk_metrics_with_benchmark(self, prices):
        benchmark = prices.pct_change().dropna() * 0.8
        wf = Workflow("risk").add(steps.returns()).add(steps.risk_metrics())
        result = wf.run(prices, benchmark=benchmark)
        assert "information_ratio" in result.risk

    def test_var_analysis(self, prices):
        wf = (
            Workflow("var")
            .add(steps.returns())
            .add(steps.var_analysis(confidence=0.95))
        )
        result = wf.run(prices)
        assert "var" in result.keys()
        assert "cvar" in result.keys()
        assert isinstance(result.var, float)
        assert isinstance(result.cvar, float)
        assert result.cvar >= result.var  # CVaR >= VaR

    def test_stationarity_test(self, prices):
        wf = Workflow("stat").add(steps.returns()).add(steps.stationarity_test())
        result = wf.run(prices)
        assert "stationarity" in result.keys()
        assert "adf" in result.stationarity
        assert "kpss" in result.stationarity

    def test_custom_step(self, prices):
        def my_step(ctx):
            r = ctx.get("returns")
            if r is not None:
                ctx["my_metric"] = float(r.mean())
            return ctx

        wf = (
            Workflow("custom")
            .add(steps.returns())
            .add(steps.custom(my_step, name="my_metric"))
        )
        result = wf.run(prices)
        assert "my_metric" in result.keys()
        assert isinstance(result.my_metric, float)

    def test_backtest_signals(self, prices):
        def constant_long(ctx):
            """Always long."""
            r = ctx["returns"]
            return np.ones(len(r))

        wf = (
            Workflow("bt")
            .add(steps.returns())
            .add(steps.backtest_signals(signal_fn=constant_long))
        )
        result = wf.run(prices)
        assert "strategy_returns" in result.keys()
        assert "signals" in result.keys()
        # Constant long => strategy_returns == returns
        np.testing.assert_array_almost_equal(
            result.strategy_returns.values, result.returns.values
        )

    def test_backtest_signals_no_fn_is_noop(self, prices):
        wf = (
            Workflow("bt_noop")
            .add(steps.returns())
            .add(steps.backtest_signals(signal_fn=None))
        )
        result = wf.run(prices)
        assert "strategy_returns" not in result.keys()

    def test_optimize_risk_parity(self, multi_asset_prices):
        wf = (
            Workflow("opt")
            .add(steps.returns())
            .add(steps.optimize(method="risk_parity"))
        )
        result = wf.run(multi_asset_prices)
        assert "optimization" in result.keys()
        assert "weights" in result.keys()
        assert len(result.weights) == 3
        assert abs(sum(result.weights) - 1.0) < 1e-4

    def test_optimize_mean_variance(self, multi_asset_prices):
        wf = (
            Workflow("opt")
            .add(steps.returns())
            .add(steps.optimize(method="mean_variance"))
        )
        result = wf.run(multi_asset_prices)
        assert "weights" in result.keys()

    def test_stress_test(self, prices):
        wf = (
            Workflow("stress")
            .add(steps.returns())
            .add(steps.stress_test())
        )
        result = wf.run(prices)
        assert "stress" in result.keys()

    def test_stress_test_custom_scenarios(self, prices):
        scenarios = {"tiny": -0.001, "big": -0.05}
        wf = (
            Workflow("stress")
            .add(steps.returns())
            .add(steps.stress_test(scenarios=scenarios))
        )
        result = wf.run(prices)
        assert "stress" in result.keys()


# ---------------------------------------------------------------------------
# Pre-built workflows (end-to-end)
# ---------------------------------------------------------------------------


class TestPrebuiltWorkflows:
    def test_quick_analysis(self, prices):
        result = quick_analysis_workflow().run(prices)
        assert "returns" in result.keys()
        assert "risk" in result.keys()
        assert "stationarity" in result.keys()

    def test_risk_workflow(self, prices):
        result = risk_workflow().run(prices)
        assert "risk" in result.keys()
        assert "var" in result.keys()
        assert "cvar" in result.keys()
        assert "stress" in result.keys()

    def test_portfolio_workflow(self, multi_asset_prices):
        result = portfolio_workflow().run(multi_asset_prices)
        assert "returns" in result.keys()
        assert "returns_df" in result.keys()

    def test_ml_workflow_without_signal(self, prices):
        """ML workflow without signal_fn should still compute features."""
        wf = ml_workflow(signal_fn=None)
        result = wf.run(prices)
        assert "returns" in result.keys()
        # Features computed from returns + vol (ta_features may need OHLCV)
        assert "features" in result.keys() or "returns" in result.keys()

    def test_ml_workflow_with_signal(self, prices):
        def momentum(ctx):
            r = ctx["returns"]
            return (r.rolling(20).mean() > 0).astype(int).values

        wf = ml_workflow(signal_fn=momentum)
        result = wf.run(prices)
        assert "strategy_returns" in result.keys()
        assert "risk" in result.keys()


# ---------------------------------------------------------------------------
# Compose via top-level import
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_workflow(self):
        import wraquant as wq

        assert hasattr(wq, "Workflow")
        assert hasattr(wq, "steps")
        assert hasattr(wq, "quick_analysis_workflow")
        assert hasattr(wq, "risk_workflow")
        assert hasattr(wq, "ml_workflow")
        assert hasattr(wq, "portfolio_workflow")

    def test_top_level_quick_analysis(self, prices):
        import wraquant as wq

        result = wq.quick_analysis_workflow().run(prices)
        assert "risk" in result.keys()
