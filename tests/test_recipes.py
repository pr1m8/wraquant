"""Smoke tests for wraquant.recipes — integrated workflow pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synth_returns() -> pd.Series:
    """500-observation return series with mild regime structure."""
    rng = np.random.default_rng(42)
    r = rng.normal(0.0003, 0.012, 500)
    return pd.Series(r, index=pd.bdate_range("2022-01-03", periods=500), name="ret")


@pytest.fixture
def synth_prices(synth_returns: pd.Series) -> pd.Series:
    """Price series derived from synth_returns."""
    return (1 + synth_returns).cumprod() * 100


@pytest.fixture
def multi_returns(synth_returns: pd.Series) -> pd.DataFrame:
    """Multi-asset return DataFrame (3 assets)."""
    rng = np.random.default_rng(99)
    n = len(synth_returns)
    return pd.DataFrame(
        {
            "A": synth_returns.values,
            "B": rng.normal(0.0001, 0.005, n),
            "C": rng.normal(0.0002, 0.008, n),
        },
        index=synth_returns.index,
    )


# ---------------------------------------------------------------------------
# analyze()
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Tests for wraquant.recipes.analyze."""

    def test_returns_expected_keys(self, synth_returns: pd.Series) -> None:
        from wraquant.recipes import analyze

        result = analyze(synth_returns)

        # Core sections always present
        assert "descriptive" in result
        assert "risk" in result
        assert "distribution" in result
        assert "stationarity" in result

    def test_risk_metrics_populated(self, synth_returns: pd.Series) -> None:
        from wraquant.recipes import analyze

        result = analyze(synth_returns)

        assert "sharpe" in result["risk"]
        assert "sortino" in result["risk"]
        assert "max_drawdown" in result["risk"]
        assert isinstance(result["risk"]["sharpe"], float)

    def test_descriptive_populated(self, synth_returns: pd.Series) -> None:
        from wraquant.recipes import analyze

        result = analyze(synth_returns)

        desc = result["descriptive"]
        assert "mean" in desc
        assert "std" in desc
        assert "skew" in desc
        assert desc["count"] == len(synth_returns)

    def test_stationarity(self, synth_returns: pd.Series) -> None:
        from wraquant.recipes import analyze

        result = analyze(synth_returns)

        stat = result["stationarity"]
        assert "p_value" in stat
        assert "test_statistic" in stat

    def test_with_benchmark(self, synth_returns: pd.Series) -> None:
        from wraquant.recipes import analyze

        rng = np.random.default_rng(123)
        bench = pd.Series(
            rng.normal(0.0002, 0.01, len(synth_returns)),
            index=synth_returns.index,
        )
        result = analyze(synth_returns, benchmark=bench)

        assert "relative" in result
        assert "information_ratio" in result["relative"]
        assert "beta" in result["relative"]

    def test_dataframe_input(self, multi_returns: pd.DataFrame) -> None:
        from wraquant.recipes import analyze

        result = analyze(multi_returns)
        assert "descriptive" in result
        assert "risk" in result

    def test_top_level_import(self, synth_returns: pd.Series) -> None:
        """analyze() is accessible from wraquant top level."""
        import wraquant as wq

        result = wq.analyze(synth_returns)
        assert "descriptive" in result


# ---------------------------------------------------------------------------
# regime_aware_backtest()
# ---------------------------------------------------------------------------


class TestRegimeAwareBacktest:
    """Tests for wraquant.recipes.regime_aware_backtest."""

    def test_runs_without_error(self, synth_prices: pd.Series) -> None:
        from wraquant.recipes import regime_aware_backtest

        result = regime_aware_backtest(synth_prices, n_regimes=2)

        assert "regime_result" in result
        assert "strategy_returns" in result
        assert "tearsheet" in result
        assert "regime_stats" in result
        assert "risk_metrics" in result

    def test_strategy_returns_shape(self, synth_prices: pd.Series) -> None:
        from wraquant.recipes import regime_aware_backtest

        result = regime_aware_backtest(synth_prices)

        sr = result["strategy_returns"]
        assert isinstance(sr, pd.Series)
        assert len(sr) > 0

    def test_risk_metrics_populated(self, synth_prices: pd.Series) -> None:
        from wraquant.recipes import regime_aware_backtest

        result = regime_aware_backtest(synth_prices)

        rm = result["risk_metrics"]
        assert "sharpe" in rm
        assert "sortino" in rm
        assert "max_drawdown" in rm
        assert isinstance(rm["sharpe"], float)


# ---------------------------------------------------------------------------
# garch_risk_pipeline()
# ---------------------------------------------------------------------------


class TestGarchRiskPipeline:
    """Tests for wraquant.recipes.garch_risk_pipeline."""

    def test_returns_expected_keys(self, synth_returns: pd.Series) -> None:
        from wraquant.recipes import garch_risk_pipeline

        result = garch_risk_pipeline(synth_returns, vol_model="GARCH", dist="normal")

        assert "garch" in result
        assert "var" in result
        assert "news_impact" in result
        assert "diagnostics" in result

    def test_diagnostics_populated(self, synth_returns: pd.Series) -> None:
        from wraquant.recipes import garch_risk_pipeline

        result = garch_risk_pipeline(synth_returns, vol_model="GARCH", dist="normal")

        diag = result["diagnostics"]
        assert "persistence" in diag
        assert "half_life" in diag
        assert "current_vol" in diag
        assert "breach_rate" in diag
        assert 0 <= diag["persistence"] <= 1.5  # persistence should be reasonable
        assert diag["current_vol"] > 0

    def test_var_breach_rate_reasonable(self, synth_returns: pd.Series) -> None:
        from wraquant.recipes import garch_risk_pipeline

        result = garch_risk_pipeline(
            synth_returns,
            vol_model="GARCH",
            dist="normal",
            var_alpha=0.05,
        )

        # Breach rate should be in a reasonable range (not exactly alpha
        # due to finite sample, but not wildly off)
        assert 0 <= result["diagnostics"]["breach_rate"] <= 0.30


# ---------------------------------------------------------------------------
# portfolio_construction_pipeline()
# ---------------------------------------------------------------------------


class TestPortfolioConstructionPipeline:
    """Tests for wraquant.recipes.portfolio_construction_pipeline."""

    def test_runs_without_error(self, multi_returns: pd.DataFrame) -> None:
        from wraquant.recipes import portfolio_construction_pipeline

        result = portfolio_construction_pipeline(
            multi_returns,
            method="risk_parity",
            regime_aware=False,
        )

        assert "weights" in result
        assert "optimization" in result
        assert "component_var" in result
        assert "diversification_ratio" in result
        assert "betas" in result

    def test_weights_sum_to_one(self, multi_returns: pd.DataFrame) -> None:
        from wraquant.recipes import portfolio_construction_pipeline

        result = portfolio_construction_pipeline(
            multi_returns,
            method="risk_parity",
            regime_aware=False,
        )

        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 0.01

    def test_all_assets_have_weights(self, multi_returns: pd.DataFrame) -> None:
        from wraquant.recipes import portfolio_construction_pipeline

        result = portfolio_construction_pipeline(
            multi_returns,
            method="risk_parity",
            regime_aware=False,
        )

        assert set(result["weights"].keys()) == set(multi_returns.columns)

    def test_diversification_ratio_positive(self, multi_returns: pd.DataFrame) -> None:
        from wraquant.recipes import portfolio_construction_pipeline

        result = portfolio_construction_pipeline(
            multi_returns,
            method="risk_parity",
            regime_aware=False,
        )

        assert result["diversification_ratio"] >= 1.0
