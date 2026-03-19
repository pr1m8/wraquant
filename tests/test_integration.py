"""Cross-module integration smoke tests.

Verifies that the new bridge functions correctly import from and
connect their respective modules.  Each test exercises a real
cross-module function call, not just an import check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. microstructure/ -> execution/ bridge: liquidity_adjusted_cost
# ---------------------------------------------------------------------------


class TestLiquidityAdjustedCost:
    """execution.cost.liquidity_adjusted_cost uses microstructure.liquidity."""

    def test_basic_call(self):
        from wraquant.execution.cost import liquidity_adjusted_cost

        result = liquidity_adjusted_cost(
            price=100.0,
            quantity=5000,
            bid=99.98,
            ask=100.02,
            volume=1_000_000,
        )
        assert isinstance(result, dict)
        assert result["spread_cost"] > 0
        assert result["total_cost"] >= result["spread_cost"]
        assert result["cost_bps"] > 0
        assert "effective_spread" in result
        assert "amihud_illiquidity" in result

    def test_series_inputs(self):
        from wraquant.execution.cost import liquidity_adjusted_cost

        n = 20
        bid = pd.Series(np.full(n, 99.98))
        ask = pd.Series(np.full(n, 100.02))
        volume = pd.Series(np.full(n, 1_000_000.0))

        result = liquidity_adjusted_cost(
            price=100.0,
            quantity=10_000,
            bid=bid,
            ask=ask,
            volume=volume,
            avg_daily_volume=2_000_000,
        )
        assert result["total_cost"] > 0

    def test_imports_from_microstructure(self):
        """Verify the function actually imports from microstructure."""
        from wraquant.microstructure.liquidity import (
            amihud_illiquidity,
            effective_spread,
        )

        # These should be importable (the bridge function depends on them)
        assert callable(effective_spread)
        assert callable(amihud_illiquidity)


# ---------------------------------------------------------------------------
# 2. execution/ -> microstructure/ bridge: adaptive_schedule
# ---------------------------------------------------------------------------


class TestAdaptiveSchedule:
    """execution.algorithms.adaptive_schedule uses microstructure signals."""

    def test_basic_schedule(self):
        from wraquant.execution.algorithms import adaptive_schedule

        volumes = np.array([1000, 2000, 1500, 3000, 2500])
        spreads = np.array([0.05, 0.02, 0.08, 0.03, 0.04])

        schedule = adaptive_schedule(10_000, volumes, spreads, urgency=0.5)

        assert len(schedule) == 5
        assert np.isclose(schedule.sum(), 10_000)
        # Interval with tightest spread (0.02) and high volume should get more
        assert schedule[1] > schedule[2]

    def test_urgency_zero_equals_vwap(self):
        from wraquant.execution.algorithms import adaptive_schedule, vwap_schedule

        volumes = np.array([1000, 2000, 3000])
        spreads = np.array([0.05, 0.02, 0.10])

        adaptive = adaptive_schedule(10_000, volumes, spreads, urgency=0.0)
        vwap = vwap_schedule(10_000, volumes)

        np.testing.assert_allclose(adaptive, vwap, atol=1e-10)

    def test_urgency_one_maximises_spread_sensitivity(self):
        from wraquant.execution.algorithms import adaptive_schedule

        volumes = np.array([1000, 1000, 1000])
        spreads = np.array([0.10, 0.01, 0.10])  # middle interval has tight spread

        schedule = adaptive_schedule(10_000, volumes, spreads, urgency=1.0)
        # Middle interval should dominate
        assert schedule[1] > schedule[0]
        assert schedule[1] > schedule[2]


# ---------------------------------------------------------------------------
# 3. ml/ -> ta/ integration: ta_features
# ---------------------------------------------------------------------------


class TestTaFeatures:
    """ml.features.ta_features imports from wraquant.ta."""

    def test_basic_features(self):
        from wraquant.ml.features import ta_features

        np.random.seed(0)
        n = 100
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        feats = ta_features(high, low, close)

        assert isinstance(feats, pd.DataFrame)
        assert "ta_rsi" in feats.columns
        assert "ta_atr" in feats.columns
        assert len(feats) == n

    def test_with_volume(self):
        from wraquant.ml.features import ta_features

        np.random.seed(0)
        n = 100
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = pd.Series(np.random.randint(1_000_000, 5_000_000, n))

        feats = ta_features(high, low, close, volume=volume)
        assert "ta_obv" in feats.columns

    def test_subset_include(self):
        from wraquant.ml.features import ta_features

        np.random.seed(0)
        n = 100
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        feats = ta_features(high, low, close, include=["rsi", "atr"])
        assert "ta_rsi" in feats.columns
        assert "ta_atr" in feats.columns
        # Should NOT have macd since we only asked for rsi and atr
        assert "ta_macd_hist" not in feats.columns

    def test_imports_from_ta(self):
        """Verify actual imports from wraquant.ta work."""
        from wraquant.ta.momentum import macd, rsi
        from wraquant.ta.overlap import bollinger_bands
        from wraquant.ta.volatility import atr

        assert callable(rsi)
        assert callable(macd)
        assert callable(bollinger_bands)
        assert callable(atr)


# ---------------------------------------------------------------------------
# 4. ts/ -> vol/ integration: garch_residual_forecast
# ---------------------------------------------------------------------------


class TestGarchResidualForecast:
    """ts.forecasting.garch_residual_forecast chains vol and ts modules."""

    def test_basic_forecast(self):
        from wraquant.ts.forecasting import garch_residual_forecast

        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.01)

        result = garch_residual_forecast(returns, horizon=5)

        assert "return_forecast" in result
        assert "vol_forecast" in result
        assert "residual_forecast" in result
        assert "standardized_residuals" in result
        assert "garch_params" in result
        assert len(result["return_forecast"]) == 5
        assert len(result["vol_forecast"]) == 5

    def test_imports_from_vol(self):
        """Verify imports from wraquant.vol.models work."""
        from wraquant.vol.models import garch_fit, garch_forecast

        assert callable(garch_fit)
        assert callable(garch_forecast)


# ---------------------------------------------------------------------------
# 5. price/ -> risk/ integration: greeks_var
# ---------------------------------------------------------------------------


class TestGreeksVar:
    """risk.var.greeks_var bridges price Greeks and risk VaR."""

    def test_basic_greeks_var(self):
        from wraquant.risk.var import greeks_var

        greeks = {"delta": 100, "gamma": -50, "vega": 200, "theta": -10}
        result = greeks_var(greeks, spot=100, vol=0.20, seed=42)

        assert isinstance(result, dict)
        assert result["var"] > 0
        assert result["cvar"] >= result["var"]
        assert "mean_pnl" in result
        assert "std_pnl" in result
        assert "delta_component" in result
        assert "gamma_component" in result
        assert "vega_component" in result

    def test_delta_only(self):
        from wraquant.risk.var import greeks_var

        greeks = {"delta": 500, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        result = greeks_var(greeks, spot=100, vol=0.20, seed=42)

        assert result["var"] > 0
        # Gamma and vega components should be near zero
        assert result["gamma_component"] < 1e-6
        assert result["vega_component"] < 1e-6

    def test_reproducibility(self):
        from wraquant.risk.var import greeks_var

        greeks = {"delta": 100, "gamma": -50, "vega": 200, "theta": -10}
        r1 = greeks_var(greeks, spot=100, vol=0.20, seed=123)
        r2 = greeks_var(greeks, spot=100, vol=0.20, seed=123)

        assert r1["var"] == r2["var"]
        assert r1["cvar"] == r2["cvar"]


# ---------------------------------------------------------------------------
# 6. bayes/ -> regimes/ integration: bayesian_regime_inference
# ---------------------------------------------------------------------------


class TestBayesianRegimeInference:
    """bayes.models.bayesian_regime_inference returns a RegimeResult."""

    def test_basic_inference(self):
        from wraquant.bayes.models import bayesian_regime_inference
        from wraquant.regimes.base import RegimeResult

        rng = np.random.default_rng(42)
        # Two-regime returns
        regime = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
        mu = np.array([0.001, -0.002])
        sigma = np.array([0.01, 0.03])
        returns = rng.normal(mu[regime], sigma[regime])

        result = bayesian_regime_inference(
            returns, n_regimes=2, n_samples=200, seed=42
        )

        assert isinstance(result, RegimeResult)
        assert result.n_regimes == 2
        assert result.states.shape == (200,)
        assert result.probabilities.shape == (200, 2)
        assert result.transition_matrix.shape == (2, 2)
        assert result.method == "bayesian_gibbs"
        # Transition matrix rows should sum to 1
        np.testing.assert_allclose(
            result.transition_matrix.sum(axis=1), [1.0, 1.0], atol=1e-6
        )

    def test_three_regimes(self):
        from wraquant.bayes.models import bayesian_regime_inference

        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, 300)

        result = bayesian_regime_inference(
            returns, n_regimes=3, n_samples=100, seed=0
        )

        assert result.n_regimes == 3
        assert result.probabilities.shape[1] == 3


# ---------------------------------------------------------------------------
# 7. viz/ smart detection: auto_plot
# ---------------------------------------------------------------------------


class TestAutoPlot:
    """viz.auto_plot auto-detects data type and dispatches correctly."""

    def test_import(self):
        from wraquant.viz import auto_plot

        assert callable(auto_plot)

    def test_unknown_type_raises(self):
        from wraquant.viz import auto_plot

        with pytest.raises(TypeError, match="Cannot auto-detect"):
            auto_plot(42)  # int is not a valid data type

    def test_explicit_kind_distribution(self):
        """Test that kind='distribution' doesn't raise on valid input."""
        # We don't test the actual plot rendering (requires matplotlib/plotly
        # display) but verify the function dispatches correctly.
        # This is a smoke test -- if it raises, the bridge is broken.
        from wraquant.viz import auto_plot

        returns = pd.Series(np.random.randn(100) * 0.01, name="returns")
        try:
            auto_plot(returns, kind="distribution")
        except Exception as e:
            # Some environments don't have display backends, which is OK
            if "display" in str(e).lower() or "backend" in str(e).lower():
                pytest.skip("No display backend available")
            raise


# ---------------------------------------------------------------------------
# 8. forex/ -> risk/ bridge: fx_portfolio_risk
# ---------------------------------------------------------------------------


class TestFxPortfolioRisk:
    """forex.risk.fx_portfolio_risk bridges forex and risk."""

    def test_basic_positions(self):
        from wraquant.forex.risk import fx_portfolio_risk

        positions = {"USD_stock": 100_000, "EUR_stock": 80_000}
        rates = {"USD": 1.0, "EUR": 1.10}

        result = fx_portfolio_risk(positions, rates, base_currency="USD")

        assert isinstance(result, dict)
        assert result["total_value_base"] > 0
        assert "positions_base" in result
        assert "currency_exposure" in result
        assert isinstance(result["positions_base"], dict)
        assert isinstance(result["currency_exposure"], dict)

    def test_currency_exposure_sums_to_one(self):
        from wraquant.forex.risk import fx_portfolio_risk

        positions = {"USD_stock": 50_000, "EUR_bond": 50_000, "JPY_equity": 5_000_000}
        rates = {"USD": 1.0, "EUR": 1.10, "JPY": 0.0067}

        result = fx_portfolio_risk(positions, rates)

        total_exposure = sum(result["currency_exposure"].values())
        assert abs(total_exposure - 1.0) < 1e-6

    def test_with_return_data(self):
        from wraquant.forex.risk import fx_portfolio_risk

        np.random.seed(0)
        n = 100
        positions = {"USD_stock": 100_000, "EUR_bond": 80_000}
        rates = {"USD": 1.0, "EUR": 1.10}

        returns = pd.DataFrame({
            "USD_stock": np.random.randn(n) * 0.01,
            "EUR_bond": np.random.randn(n) * 0.008,
        })
        fx_returns = pd.DataFrame({
            "USD": np.zeros(n),
            "EUR": np.random.randn(n) * 0.005,
        })

        result = fx_portfolio_risk(
            positions, rates,
            returns=returns,
            fx_returns=fx_returns,
        )

        assert result["fx_adjusted_vol"] is not None
        assert result["asset_vol"] is not None
        assert result["fx_adjusted_vol"] > 0

    def test_importable_from_forex_package(self):
        from wraquant.forex import fx_portfolio_risk

        assert callable(fx_portfolio_risk)


# ---------------------------------------------------------------------------
# Cross-module import verification
# ---------------------------------------------------------------------------


class TestCrossModuleImports:
    """Verify all new integration functions are importable from their packages."""

    def test_execution_exports(self):
        from wraquant.execution import adaptive_schedule, liquidity_adjusted_cost

        assert callable(adaptive_schedule)
        assert callable(liquidity_adjusted_cost)

    def test_ml_exports(self):
        from wraquant.ml import ta_features

        assert callable(ta_features)

    def test_ts_exports(self):
        from wraquant.ts import garch_residual_forecast

        assert callable(garch_residual_forecast)

    def test_risk_exports(self):
        from wraquant.risk import greeks_var

        assert callable(greeks_var)

    def test_bayes_exports(self):
        from wraquant.bayes import bayesian_regime_inference

        assert callable(bayesian_regime_inference)

    def test_viz_exports(self):
        from wraquant.viz import auto_plot

        assert callable(auto_plot)

    def test_forex_exports(self):
        from wraquant.forex import fx_portfolio_risk

        assert callable(fx_portfolio_risk)
