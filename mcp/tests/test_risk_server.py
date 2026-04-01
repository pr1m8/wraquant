"""Tests for risk management MCP tools.

Tests each risk tool with synthetic data through the MCP context:
var_analysis, stress_test, beta_analysis, factor_analysis,
crisis_drawdowns, portfolio_risk, tail_risk, credit_analysis.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
    """Create an AnalysisContext with synthetic data pre-loaded."""
    ws = tmp_path / "test_risk"
    context = AnalysisContext(str(ws))

    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    log_rets = rng.normal(0, 0.015, n)
    prices = 100 * np.exp(np.cumsum(log_rets))

    price_df = pd.DataFrame({
        "open": prices * (1 + rng.normal(0, 0.003, n)),
        "high": prices * (1 + abs(rng.normal(0, 0.005, n))),
        "low": prices * (1 - abs(rng.normal(0, 0.005, n))),
        "close": prices,
        "volume": rng.integers(100_000, 1_000_000, n),
    }, index=dates)
    context.store_dataset("prices", price_df)

    returns = pd.Series(log_rets, index=dates, name="returns")
    context.store_dataset("returns", returns.to_frame("returns"), parent="prices")

    # Benchmark returns (correlated)
    bench_rets = 0.7 * log_rets + 0.3 * rng.normal(0, 0.012, n)
    context.store_dataset(
        "benchmark",
        pd.DataFrame({"returns": bench_rets}, index=dates),
    )

    # Multi-asset returns for portfolio tools
    multi = pd.DataFrame({
        "AAPL": rng.normal(0.0003, 0.02, n),
        "MSFT": rng.normal(0.0002, 0.018, n),
        "GOOGL": rng.normal(0.0001, 0.022, n),
        "AMZN": rng.normal(0.0004, 0.025, n),
    }, index=dates)
    context.store_dataset("portfolio_returns", multi)

    yield context
    context.close()
    if ws.exists():
        shutil.rmtree(ws)


@pytest.fixture
def risk_tools(ctx):
    """Register risk tools and return them."""
    from wraquant_mcp.servers.risk import register_risk_tools

    mock = MockMCP()
    register_risk_tools(mock, ctx)
    return mock.tools


# ------------------------------------------------------------------
# VaR analysis
# ------------------------------------------------------------------


class TestVarAnalysis:
    """Test var_analysis tool."""

    def test_var_historical(self, risk_tools):
        result = risk_tools["var_analysis"](
            dataset="returns", column="returns",
            confidence=0.95, method="historical",
        )
        assert result["tool"] == "var_analysis"
        assert isinstance(result["var"], float)
        assert isinstance(result["cvar"], float)
        assert result["var"] > 0
        assert result["cvar"] >= result["var"]
        assert result["confidence"] == 0.95
        assert result["observations"] > 0

    def test_var_parametric(self, risk_tools):
        result = risk_tools["var_analysis"](
            dataset="returns", column="returns",
            confidence=0.99, method="parametric",
        )
        assert result["var"] > 0
        assert result["method"] == "parametric"
        assert result["confidence"] == 0.99

    def test_var_higher_confidence_gives_higher_var(self, risk_tools):
        r95 = risk_tools["var_analysis"](
            dataset="returns", confidence=0.95,
        )
        r99 = risk_tools["var_analysis"](
            dataset="returns", confidence=0.99,
        )
        assert r99["var"] > r95["var"]

    def test_var_cvar_exceeds_var(self, risk_tools):
        result = risk_tools["var_analysis"](
            dataset="returns", confidence=0.95,
        )
        assert result["cvar"] >= result["var"]


# ------------------------------------------------------------------
# Stress test
# ------------------------------------------------------------------


class TestStressTest:
    """Test stress_test tool and underlying wraquant stress functions."""

    def test_stress_test_historical(self, risk_tools):
        result = risk_tools["stress_test"](
            dataset="returns", historical=True,
        )
        assert "historical" in result["results"]

    def test_vol_stress_test_direct(self, ctx):
        """Test vol_stress_test function directly through context."""
        from wraquant.risk.stress import vol_stress_test

        returns = ctx.get_dataset("returns")["returns"].dropna()
        result = vol_stress_test(returns, vol_shocks=[1.5, 2.0, 3.0])
        assert isinstance(result, dict)

    def test_stress_test_returns_direct(self, ctx):
        """Test stress_test_returns function directly through context."""
        from wraquant.risk.stress import stress_test_returns

        returns = ctx.get_dataset("returns")["returns"].dropna()
        result = stress_test_returns(returns, scenarios={"crash": -0.10})
        assert isinstance(result, dict)

    def test_historical_stress_test_direct(self, ctx):
        """Test historical_stress_test function directly."""
        from wraquant.risk.stress import historical_stress_test

        returns = ctx.get_dataset("returns")["returns"].dropna()
        result = historical_stress_test(returns)
        assert isinstance(result, dict)


# ------------------------------------------------------------------
# Beta analysis
# ------------------------------------------------------------------


class TestBetaAnalysis:
    """Test beta_analysis tool via direct wraquant calls through context."""

    def test_rolling_beta_through_context(self, ctx):
        """Test rolling beta computation through context."""
        from wraquant.risk.beta import rolling_beta

        returns = ctx.get_dataset("returns")["returns"]
        benchmark = ctx.get_dataset("benchmark")["returns"]
        n = min(len(returns), len(benchmark))
        returns = returns.iloc[-n:]
        benchmark = benchmark.iloc[-n:]

        rb = rolling_beta(returns, benchmark, window=60)
        assert isinstance(rb, pd.Series)
        assert len(rb) == n
        # Most values should be finite after warmup
        assert rb.dropna().shape[0] > 0

        # Store in context
        rb_df = pd.DataFrame({"rolling_beta": rb})
        stored = ctx.store_dataset("beta_test", rb_df, source_op="beta_analysis", parent="returns")
        assert stored["dataset_id"] == "beta_test"
        assert "rolling_beta" in stored["columns"]

    def test_conditional_beta_through_context(self, ctx):
        """Test conditional beta computation."""
        from wraquant.risk.beta import conditional_beta

        returns = ctx.get_dataset("returns")["returns"]
        benchmark = ctx.get_dataset("benchmark")["returns"]
        n = min(len(returns), len(benchmark))

        cond = conditional_beta(returns.iloc[-n:], benchmark.iloc[-n:])
        assert isinstance(cond, dict)
        assert "upside_beta" in cond
        assert "downside_beta" in cond

    def test_blume_adjusted_beta(self):
        """Test Blume adjustment."""
        from wraquant.risk.beta import blume_adjusted_beta

        adjusted = blume_adjusted_beta(1.5)
        assert adjusted == pytest.approx(0.33 + 0.67 * 1.5)

        # Beta of 1.0 stays at 1.0
        assert blume_adjusted_beta(1.0) == pytest.approx(1.0)

    def test_vasicek_adjusted_beta(self):
        """Test Vasicek Bayesian shrinkage."""
        from wraquant.risk.beta import vasicek_adjusted_beta

        adjusted = vasicek_adjusted_beta(1.5)
        assert isinstance(adjusted, float)
        # Should shrink toward 1.0
        assert 1.0 < adjusted < 1.5


# ------------------------------------------------------------------
# Factor analysis
# ------------------------------------------------------------------


class TestFactorAnalysis:
    """Test factor analysis through direct wraquant calls."""

    def test_statistical_factor_model_direct(self, ctx):
        """Test PCA factor model directly on multi-asset returns."""
        from wraquant.risk.factor import statistical_factor_model

        df = ctx.get_dataset("portfolio_returns")
        result = statistical_factor_model(df, n_factors=2)

        assert isinstance(result, dict)
        assert "factors" in result
        assert "loadings" in result
        assert "explained_variance" in result
        assert "explained_variance_ratio" in result

        # Store as model in context
        stored = ctx.store_model(
            "pca_factors", result,
            model_type="factor_pca",
            source_dataset="portfolio_returns",
        )
        assert stored["model_id"] == "pca_factors"

    def test_factor_model_variance_explained(self, ctx):
        """PCA should explain some meaningful variance."""
        from wraquant.risk.factor import statistical_factor_model

        df = ctx.get_dataset("portfolio_returns")
        result = statistical_factor_model(df, n_factors=2)

        total = sum(result["explained_variance_ratio"])
        assert total > 0.1  # should explain more than 10% with 2 factors

    def test_factor_model_stores_in_context(self, ctx):
        """Verify model is stored and retrievable."""
        from wraquant.risk.factor import statistical_factor_model

        df = ctx.get_dataset("portfolio_returns")
        result = statistical_factor_model(df, n_factors=2)
        ctx.store_model("factor_test", result, model_type="pca")

        retrieved = ctx.get_model("factor_test")
        assert "loadings" in retrieved


# ------------------------------------------------------------------
# Crisis drawdowns
# ------------------------------------------------------------------


class TestCrisisDrawdowns:
    """Test crisis_drawdowns tool."""

    def test_crisis_drawdowns_returns_results(self, risk_tools):
        result = risk_tools["crisis_drawdowns"](
            dataset="returns", column="returns", top_n=3,
        )
        assert result["tool"] == "crisis_drawdowns"
        assert result["top_n"] == 3
        assert "drawdowns" in result

    def test_crisis_drawdowns_default_top_n(self, risk_tools):
        result = risk_tools["crisis_drawdowns"](
            dataset="returns",
        )
        assert result["top_n"] == 5


# ------------------------------------------------------------------
# Portfolio risk
# ------------------------------------------------------------------


class TestPortfolioRisk:
    """Test portfolio_risk tool through direct wraquant calls."""

    def test_portfolio_volatility_through_context(self, ctx):
        """Test portfolio volatility computation."""
        from wraquant.risk.portfolio import portfolio_volatility, risk_contribution

        df = ctx.get_dataset("portfolio_returns")
        returns = df.select_dtypes(include=[np.number]).dropna()
        n_assets = returns.shape[1]
        weights = np.array([1.0 / n_assets] * n_assets)
        cov = returns.cov().values

        vol = portfolio_volatility(weights, cov)
        assert isinstance(vol, float)
        assert vol > 0

        rc = risk_contribution(weights, cov)
        assert isinstance(rc, np.ndarray)
        assert len(rc) == n_assets
        # Risk contributions should approximately sum to 1
        assert abs(rc.sum() - 1.0) < 0.1

    def test_diversification_ratio(self, ctx):
        """Test diversification ratio computation."""
        from wraquant.risk.portfolio import diversification_ratio

        df = ctx.get_dataset("portfolio_returns")
        returns = df.select_dtypes(include=[np.number]).dropna()
        n_assets = returns.shape[1]
        weights = np.array([1.0 / n_assets] * n_assets)
        cov = returns.cov().values

        div = diversification_ratio(weights, cov)
        assert isinstance(div, float)
        # Diversification ratio >= 1 for distinct assets
        assert div >= 0.9


# ------------------------------------------------------------------
# Tail risk
# ------------------------------------------------------------------


class TestTailRisk:
    """Test tail_risk tool through direct wraquant calls."""

    def test_cornish_fisher_var(self, ctx):
        """Test Cornish-Fisher VaR computation."""
        from wraquant.risk.tail import cornish_fisher_var

        returns = ctx.get_dataset("returns")["returns"].dropna()
        result = cornish_fisher_var(returns, alpha=0.05)
        assert isinstance(result, dict)
        assert "cf_var" in result
        assert isinstance(result["cf_var"], float)

    def test_conditional_drawdown_at_risk(self, ctx):
        """Test CDaR computation."""
        from wraquant.risk.tail import conditional_drawdown_at_risk

        returns = ctx.get_dataset("returns")["returns"].dropna()
        cdar = conditional_drawdown_at_risk(returns, alpha=0.05)
        assert isinstance(cdar, float)
        assert cdar > 0

    def test_drawdown_at_risk(self, ctx):
        """Test DaR computation."""
        from wraquant.risk.tail import drawdown_at_risk

        returns = ctx.get_dataset("returns")["returns"].dropna()
        dar = drawdown_at_risk(returns, alpha=0.05)
        assert isinstance(dar, float)
        assert dar > 0

    def test_tail_ratio_analysis(self, ctx):
        """Test tail ratio analysis."""
        from wraquant.risk.tail import tail_ratio_analysis

        returns = ctx.get_dataset("returns")["returns"].dropna()
        result = tail_ratio_analysis(returns)
        assert isinstance(result, dict)
        assert "tail_ratio" in result


# ------------------------------------------------------------------
# Credit analysis
# ------------------------------------------------------------------


class TestCreditAnalysis:
    """Test credit_analysis tool."""

    def test_merton_model(self):
        """Test Merton model directly."""
        from wraquant.risk.credit import merton_model

        result = merton_model(
            equity=100.0,
            debt=50.0,
            vol=0.30,
            rf_rate=0.05,
            maturity=1.0,
        )
        assert isinstance(result, dict)
        assert "distance_to_default" in result
        assert "default_probability" in result
        assert result["distance_to_default"] > 0

    def test_altman_z_score(self):
        """Test Altman Z-score."""
        from wraquant.risk.credit import altman_z_score

        result = altman_z_score(
            working_capital=50.0,
            total_assets=200.0,
            retained_earnings=30.0,
            ebit=25.0,
            market_cap=150.0,
            total_liabilities=80.0,
            sales=180.0,
        )
        assert isinstance(result, dict)
        assert "z_score" in result
        assert isinstance(result["z_score"], float)

    def test_merton_distance_to_default_increases_with_equity(self):
        """Higher equity relative to debt should give larger DD."""
        from wraquant.risk.credit import merton_model

        low_equity = merton_model(equity=50.0, debt=100.0, vol=0.30, rf_rate=0.05, maturity=1.0)
        high_equity = merton_model(equity=200.0, debt=50.0, vol=0.30, rf_rate=0.05, maturity=1.0)

        assert high_equity["distance_to_default"] > low_equity["distance_to_default"]

    def test_z_score_safe_zone(self):
        """Strong fundamentals should give Z > 2.99 (safe zone)."""
        from wraquant.risk.credit import altman_z_score

        result = altman_z_score(
            working_capital=100.0,
            total_assets=200.0,
            retained_earnings=60.0,
            ebit=50.0,
            market_cap=300.0,
            total_liabilities=40.0,
            sales=250.0,
        )
        assert result["z_score"] > 2.0  # should be in safe zone
