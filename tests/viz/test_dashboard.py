"""Smoke tests for dashboard and chart visualizations.

Verifies that each plotting function runs without exceptions and returns
a ``plotly.graph_objects.Figure`` with the expected number of traces.
Visual correctness is not tested; figures are never shown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_returns() -> pd.Series:
    """Simple daily return series spanning ~2 years."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=504)
    rets = rng.normal(0.0004, 0.015, size=504)
    return pd.Series(rets, index=dates, name="strategy")


@pytest.fixture
def sample_benchmark() -> pd.Series:
    """Benchmark return series."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2020-01-01", periods=504)
    rets = rng.normal(0.0003, 0.012, size=504)
    return pd.Series(rets, index=dates, name="benchmark")


@pytest.fixture
def sample_multi_returns() -> pd.DataFrame:
    """Multi-asset daily returns."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=504)
    data = rng.normal(0.0003, 0.015, size=(504, 5))
    return pd.DataFrame(data, index=dates, columns=["SPY", "AGG", "GLD", "VWO", "TLT"])


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Sample OHLCV data."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=252)))
    high = close * (1 + rng.uniform(0, 0.03, size=252))
    low = close * (1 - rng.uniform(0, 0.03, size=252))
    open_ = close * (1 + rng.normal(0, 0.01, size=252))
    volume = rng.integers(1_000_000, 10_000_000, size=252)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def sample_prices() -> pd.Series:
    """Close price series."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    returns = rng.normal(0.0005, 0.02, size=252)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="price")


# ---------------------------------------------------------------------------
# dashboard.py tests
# ---------------------------------------------------------------------------


class TestPortfolioDashboard:
    def test_basic(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.dashboard import portfolio_dashboard

        fig = portfolio_dashboard(sample_returns)
        assert isinstance(fig, go.Figure)
        # At minimum: cum returns, drawdown, rolling sharpe, rolling sortino,
        # histogram, normal overlay, heatmap, table
        assert len(fig.data) >= 6

    def test_with_benchmark(
        self, sample_returns: pd.Series, sample_benchmark: pd.Series
    ) -> None:
        from wraquant.viz.dashboard import portfolio_dashboard

        fig = portfolio_dashboard(sample_returns, benchmark=sample_benchmark)
        assert isinstance(fig, go.Figure)
        # Should have extra benchmark trace
        assert len(fig.data) >= 7

    def test_custom_title(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.dashboard import portfolio_dashboard

        fig = portfolio_dashboard(sample_returns, title="My Dashboard")
        assert fig.layout.title.text == "My Dashboard"

    def test_custom_rolling_window(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.dashboard import portfolio_dashboard

        fig = portfolio_dashboard(sample_returns, rolling_window=21)
        assert isinstance(fig, go.Figure)


class TestRegimeDashboard:
    def test_basic(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.dashboard import regime_dashboard

        rng = np.random.default_rng(42)
        states = pd.Series(
            rng.choice([0, 1, 2], size=len(sample_returns)),
            index=sample_returns.index,
        )
        fig = regime_dashboard(sample_returns, states)
        assert isinstance(fig, go.Figure)
        # Cum returns + 3 regime markers + 3 distribution histograms + table
        assert len(fig.data) >= 5

    def test_with_probabilities(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.dashboard import regime_dashboard

        rng = np.random.default_rng(42)
        states = pd.Series(
            rng.choice([0, 1], size=len(sample_returns)),
            index=sample_returns.index,
        )
        probs = pd.DataFrame(
            {"0": rng.uniform(0.3, 0.7, len(sample_returns))},
            index=sample_returns.index,
        )
        probs["1"] = 1 - probs["0"]
        fig = regime_dashboard(sample_returns, states, probabilities=probs)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 6

    def test_with_transition_matrix(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.dashboard import regime_dashboard

        rng = np.random.default_rng(42)
        states = pd.Series(
            rng.choice([0, 1], size=len(sample_returns)),
            index=sample_returns.index,
        )
        tm = np.array([[0.95, 0.05], [0.10, 0.90]])
        fig = regime_dashboard(sample_returns, states, transition_matrix=tm)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 6


class TestRiskDashboard:
    def test_multi_asset(self, sample_multi_returns: pd.DataFrame) -> None:
        from wraquant.viz.dashboard import risk_dashboard

        fig = risk_dashboard(sample_multi_returns)
        assert isinstance(fig, go.Figure)
        # Port returns + VaR + CVaR + corr heatmap + risk bar + possibly breaches
        assert len(fig.data) >= 5

    def test_single_asset(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.dashboard import risk_dashboard

        fig = risk_dashboard(sample_returns)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_with_stress_scenarios(self, sample_multi_returns: pd.DataFrame) -> None:
        from wraquant.viz.dashboard import risk_dashboard

        scenarios = {
            "2008 GFC": -0.38,
            "COVID Crash": -0.34,
            "Taper Tantrum": -0.08,
        }
        fig = risk_dashboard(sample_multi_returns, stress_scenarios=scenarios)
        assert isinstance(fig, go.Figure)
        # Should include stress bar chart
        assert len(fig.data) >= 6

    def test_custom_confidence(self, sample_multi_returns: pd.DataFrame) -> None:
        from wraquant.viz.dashboard import risk_dashboard

        fig = risk_dashboard(sample_multi_returns, var_confidence=0.99)
        assert isinstance(fig, go.Figure)


class TestTechnicalDashboard:
    def test_basic(self, sample_ohlcv: pd.DataFrame) -> None:
        from wraquant.viz.dashboard import technical_dashboard

        fig = technical_dashboard(sample_ohlcv)
        assert isinstance(fig, go.Figure)
        # OHLC + volume + SMA20 + SMA50 + BB(3) + RSI + MACD(3) = 11+
        assert len(fig.data) >= 8

    def test_minimal_indicators(self, sample_ohlcv: pd.DataFrame) -> None:
        from wraquant.viz.dashboard import technical_dashboard

        fig = technical_dashboard(sample_ohlcv, indicators=["sma20"])
        assert isinstance(fig, go.Figure)
        # OHLC + volume + SMA20
        assert len(fig.data) >= 3

    def test_rsi_only(self, sample_ohlcv: pd.DataFrame) -> None:
        from wraquant.viz.dashboard import technical_dashboard

        fig = technical_dashboard(sample_ohlcv, indicators=["rsi"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_macd_only(self, sample_ohlcv: pd.DataFrame) -> None:
        from wraquant.viz.dashboard import technical_dashboard

        fig = technical_dashboard(sample_ohlcv, indicators=["macd"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_all_indicators(self, sample_ohlcv: pd.DataFrame) -> None:
        from wraquant.viz.dashboard import technical_dashboard

        fig = technical_dashboard(
            sample_ohlcv,
            indicators=["sma20", "sma50", "ema12", "bb", "rsi", "macd"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 10

    def test_no_volume(self) -> None:
        from wraquant.viz.dashboard import technical_dashboard

        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=100)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=100)))
        df = pd.DataFrame({
            "open": close * 0.99, "high": close * 1.02,
            "low": close * 0.98, "close": close,
        }, index=dates)
        fig = technical_dashboard(df, indicators=["sma20"])
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# charts.py tests
# ---------------------------------------------------------------------------


class TestPlotMultiAsset:
    def test_basic(self, sample_multi_returns: pd.DataFrame) -> None:
        from wraquant.viz.charts import plot_multi_asset

        fig = plot_multi_asset(sample_multi_returns)
        assert isinstance(fig, go.Figure)
        # 5 performance lines + heatmap + rolling corr pairs
        assert len(fig.data) >= 6

    def test_custom_window(self, sample_multi_returns: pd.DataFrame) -> None:
        from wraquant.viz.charts import plot_multi_asset

        fig = plot_multi_asset(sample_multi_returns, rolling_corr_window=21)
        assert isinstance(fig, go.Figure)


class TestPlotVolSurface:
    def test_basic(self) -> None:
        from wraquant.viz.charts import plot_vol_surface

        strikes = np.linspace(80, 120, 20)
        mats = np.array([0.1, 0.25, 0.5, 1.0])
        rng = np.random.default_rng(42)
        iv = 0.2 + 0.05 * rng.standard_normal((len(mats), len(strikes)))
        fig = plot_vol_surface(strikes, mats, iv)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Single surface trace

    def test_custom_title(self) -> None:
        from wraquant.viz.charts import plot_vol_surface

        strikes = np.linspace(90, 110, 10)
        mats = np.array([0.25, 0.5, 1.0])
        rng = np.random.default_rng(42)
        iv = 0.2 + 0.03 * rng.standard_normal((3, 10))
        fig = plot_vol_surface(strikes, mats, iv, title="My Vol Surface")
        assert fig.layout.title.text == "My Vol Surface"


class TestPlotRegimeOverlay:
    def test_basic(self, sample_prices: pd.Series) -> None:
        from wraquant.viz.charts import plot_regime_overlay

        rng = np.random.default_rng(42)
        probs = pd.DataFrame(
            {"Bull": rng.uniform(0.3, 0.8, len(sample_prices))},
            index=sample_prices.index,
        )
        probs["Bear"] = 1 - probs["Bull"]
        fig = plot_regime_overlay(sample_prices, probs)
        assert isinstance(fig, go.Figure)
        # Price line + 2 prob stacked areas + 2 legend markers
        assert len(fig.data) >= 3


class TestPlotDistributionAnalysis:
    def test_basic(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.charts import plot_distribution_analysis

        fig = plot_distribution_analysis(sample_returns)
        assert isinstance(fig, go.Figure)
        # Histogram + KDE + Normal + best fit + QQ scatter + QQ line = 5-6 traces
        assert len(fig.data) >= 5

    def test_has_annotation(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.charts import plot_distribution_analysis

        fig = plot_distribution_analysis(sample_returns)
        assert len(fig.layout.annotations) >= 1

    def test_custom_bins(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.charts import plot_distribution_analysis

        fig = plot_distribution_analysis(sample_returns, bins=100)
        assert isinstance(fig, go.Figure)


class TestPlotCorrelationNetwork:
    def test_basic(self, sample_multi_returns: pd.DataFrame) -> None:
        from wraquant.viz.charts import plot_correlation_network

        fig = plot_correlation_network(sample_multi_returns, threshold=0.1)
        assert isinstance(fig, go.Figure)
        # At minimum: node trace + 2 legend entries
        assert len(fig.data) >= 3

    def test_with_mst(self, sample_multi_returns: pd.DataFrame) -> None:
        from wraquant.viz.charts import plot_correlation_network

        fig = plot_correlation_network(
            sample_multi_returns, threshold=0.1, show_mst=True
        )
        assert isinstance(fig, go.Figure)
        # Should have MST legend entry
        assert len(fig.data) >= 4

    def test_high_threshold(self, sample_multi_returns: pd.DataFrame) -> None:
        from wraquant.viz.charts import plot_correlation_network

        fig = plot_correlation_network(sample_multi_returns, threshold=0.99)
        assert isinstance(fig, go.Figure)


class TestPlotBacktestTearsheet:
    def test_basic(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.charts import plot_backtest_tearsheet

        fig = plot_backtest_tearsheet(sample_returns)
        assert isinstance(fig, go.Figure)
        # Equity + drawdown + heatmap + rolling sharpe + rolling vol + table
        assert len(fig.data) >= 5

    def test_with_benchmark(
        self, sample_returns: pd.Series, sample_benchmark: pd.Series
    ) -> None:
        from wraquant.viz.charts import plot_backtest_tearsheet

        fig = plot_backtest_tearsheet(sample_returns, benchmark=sample_benchmark)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 6

    def test_with_trades(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.charts import plot_backtest_tearsheet

        rng = np.random.default_rng(42)
        trades = pd.DataFrame({
            "pnl": rng.normal(0.001, 0.05, 50),
        })
        fig = plot_backtest_tearsheet(sample_returns, trades=trades)
        assert isinstance(fig, go.Figure)
        # Should have trade scatter panel
        assert len(fig.data) >= 6

    def test_custom_title(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.charts import plot_backtest_tearsheet

        fig = plot_backtest_tearsheet(sample_returns, title="My Tearsheet")
        assert fig.layout.title.text == "My Tearsheet"
