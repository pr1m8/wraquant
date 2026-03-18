"""Smoke tests for Plotly interactive visualizations.

Verifies that each plotting function runs without exceptions and returns
a ``plotly.graph_objects.Figure``.  Visual correctness is not tested;
figures are never shown.
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
    dates = pd.bdate_range("2020-01-01", periods=252)
    data = rng.normal(0.0003, 0.015, size=(252, 5))
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
# interactive.py tests
# ---------------------------------------------------------------------------


class TestPlotlyReturns:
    def test_basic(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.interactive import plotly_returns

        fig = plotly_returns(sample_returns)
        assert isinstance(fig, go.Figure)

    def test_with_benchmark(
        self, sample_returns: pd.Series, sample_benchmark: pd.Series
    ) -> None:
        from wraquant.viz.interactive import plotly_returns

        fig = plotly_returns(sample_returns, benchmark=sample_benchmark)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_custom_title(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.interactive import plotly_returns

        fig = plotly_returns(sample_returns, title="My Returns")
        assert fig.layout.title.text == "My Returns"


class TestPlotlyDrawdown:
    def test_basic(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.interactive import plotly_drawdown

        fig = plotly_drawdown(sample_returns)
        assert isinstance(fig, go.Figure)

    def test_has_annotation(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.interactive import plotly_drawdown

        fig = plotly_drawdown(sample_returns)
        assert len(fig.layout.annotations) >= 1


class TestPlotlyRollingStats:
    def test_basic(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.interactive import plotly_rolling_stats

        fig = plotly_rolling_stats(sample_returns, window=63)
        assert isinstance(fig, go.Figure)
        # Three traces (Sharpe, vol, beta)
        assert len(fig.data) == 3

    def test_custom_window(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.interactive import plotly_rolling_stats

        fig = plotly_rolling_stats(sample_returns, window=21)
        assert isinstance(fig, go.Figure)


class TestPlotlyDistribution:
    def test_basic(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.interactive import plotly_distribution

        fig = plotly_distribution(sample_returns)
        assert isinstance(fig, go.Figure)

    def test_no_normal_overlay(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.interactive import plotly_distribution

        fig = plotly_distribution(sample_returns, overlay_normal=False)
        assert isinstance(fig, go.Figure)
        # Should have histogram + KDE but no normal trace
        assert len(fig.data) == 2

    def test_custom_bins(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.interactive import plotly_distribution

        fig = plotly_distribution(sample_returns, bins=100)
        assert isinstance(fig, go.Figure)


class TestPlotlyCorrelationHeatmap:
    def test_basic(self, sample_multi_returns: pd.DataFrame) -> None:
        from wraquant.viz.interactive import plotly_correlation_heatmap

        fig = plotly_correlation_heatmap(sample_multi_returns)
        assert isinstance(fig, go.Figure)


class TestPlotlyEfficientFrontier:
    def test_basic(self) -> None:
        from wraquant.viz.interactive import plotly_efficient_frontier

        rng = np.random.default_rng(42)
        mu = rng.normal(0.08, 0.03, size=4)
        cov = np.diag(rng.uniform(0.01, 0.04, size=4))
        fig = plotly_efficient_frontier(mu, cov, n_portfolios=200)
        assert isinstance(fig, go.Figure)
        # Should have scatter cloud + max-Sharpe star
        assert len(fig.data) == 2


class TestPlotlyRiskReturnScatter:
    def test_basic(self, sample_multi_returns: pd.DataFrame) -> None:
        from wraquant.viz.interactive import plotly_risk_return_scatter

        fig = plotly_risk_return_scatter(sample_multi_returns)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# advanced.py tests
# ---------------------------------------------------------------------------


class TestPlotlyRegimeOverlay:
    def test_basic(self, sample_prices: pd.Series) -> None:
        from wraquant.viz.advanced import plotly_regime_overlay

        rng = np.random.default_rng(42)
        regimes = pd.Series(
            rng.choice([0, 1, 2], size=len(sample_prices)),
            index=sample_prices.index,
        )
        fig = plotly_regime_overlay(sample_prices, regimes)
        assert isinstance(fig, go.Figure)


class TestPlotlyVolSurface:
    def test_basic(self) -> None:
        from wraquant.viz.advanced import plotly_vol_surface

        strikes = np.linspace(80, 120, 20)
        expiries = np.array([0.1, 0.25, 0.5, 1.0])
        rng = np.random.default_rng(42)
        iv = 0.2 + 0.05 * rng.standard_normal((len(expiries), len(strikes)))
        fig = plotly_vol_surface(strikes, expiries, iv)
        assert isinstance(fig, go.Figure)


class TestPlotlyTermStructure:
    def test_basic(self) -> None:
        from wraquant.viz.advanced import plotly_term_structure

        mats = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
        rng = np.random.default_rng(42)
        yields_data = 0.03 + 0.01 * np.cumsum(
            rng.standard_normal((5, len(mats))), axis=1
        )
        fig = plotly_term_structure(mats, yields_data, dates=["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1"])
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5


class TestPlotlyCopulaScatter:
    def test_basic(self) -> None:
        from wraquant.viz.advanced import plotly_copula_scatter

        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, 500)
        v = rng.uniform(0, 1, 500)
        fig = plotly_copula_scatter(u, v, copula_type="Gaussian")
        assert isinstance(fig, go.Figure)


class TestPlotlyNetworkGraph:
    def test_array(self) -> None:
        from wraquant.viz.advanced import plotly_network_graph

        rng = np.random.default_rng(42)
        n = 6
        data = rng.standard_normal((100, n))
        corr = np.corrcoef(data, rowvar=False)
        fig = plotly_network_graph(corr, threshold=0.2)
        assert isinstance(fig, go.Figure)

    def test_dataframe(self, sample_multi_returns: pd.DataFrame) -> None:
        from wraquant.viz.advanced import plotly_network_graph

        corr = sample_multi_returns.corr()
        fig = plotly_network_graph(corr, threshold=0.1)
        assert isinstance(fig, go.Figure)


class TestPlotlySankeyFlow:
    def test_basic(self) -> None:
        from wraquant.viz.advanced import plotly_sankey_flow

        sectors = ["Tech", "Health", "Finance", "Energy"]
        before = [0.4, 0.3, 0.2, 0.1]
        after = [0.3, 0.25, 0.25, 0.2]
        fig = plotly_sankey_flow(sectors, before, after)
        assert isinstance(fig, go.Figure)


class TestPlotlyTreemap:
    def test_basic(self) -> None:
        from wraquant.viz.advanced import plotly_treemap

        sectors = ["Tech", "Health", "Finance", "Energy", "Utilities"]
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        returns = [0.05, -0.02, 0.03, -0.01, 0.01]
        fig = plotly_treemap(weights, sectors, returns)
        assert isinstance(fig, go.Figure)


class TestPlotlyRadar:
    def test_basic(self) -> None:
        from wraquant.viz.advanced import plotly_radar

        metrics = {
            "Portfolio A": {"Sharpe": 1.2, "Sortino": 1.5, "MaxDD": 0.15, "Vol": 0.12},
            "Portfolio B": {"Sharpe": 0.9, "Sortino": 1.1, "MaxDD": 0.20, "Vol": 0.18},
        }
        fig = plotly_radar(metrics)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2


# ---------------------------------------------------------------------------
# candlestick.py tests
# ---------------------------------------------------------------------------


class TestPlotlyCandlestick:
    def test_basic(self, sample_ohlcv: pd.DataFrame) -> None:
        from wraquant.viz.candlestick import plotly_candlestick

        fig = plotly_candlestick(sample_ohlcv)
        assert isinstance(fig, go.Figure)

    def test_with_overlays(self, sample_ohlcv: pd.DataFrame) -> None:
        from wraquant.viz.candlestick import plotly_candlestick

        fig = plotly_candlestick(sample_ohlcv, overlays=["sma20", "sma50", "bb"])
        assert isinstance(fig, go.Figure)
        # OHLC + Volume + SMA20 + SMA50 + BB upper + BB lower + BB mid = 7
        assert len(fig.data) >= 6

    def test_with_ema(self, sample_ohlcv: pd.DataFrame) -> None:
        from wraquant.viz.candlestick import plotly_candlestick

        fig = plotly_candlestick(sample_ohlcv, overlays=["ema20"])
        assert isinstance(fig, go.Figure)


class TestPlotlyMarketProfile:
    def test_basic(self, sample_ohlcv: pd.DataFrame) -> None:
        from wraquant.viz.candlestick import plotly_market_profile

        fig = plotly_market_profile(sample_ohlcv)
        assert isinstance(fig, go.Figure)


class TestPlotlyRenko:
    def test_basic(self, sample_prices: pd.Series) -> None:
        from wraquant.viz.candlestick import plotly_renko

        fig = plotly_renko(sample_prices)
        assert isinstance(fig, go.Figure)

    def test_custom_brick(self, sample_prices: pd.Series) -> None:
        from wraquant.viz.candlestick import plotly_renko

        fig = plotly_renko(sample_prices, brick_size=2.0)
        assert isinstance(fig, go.Figure)


class TestPlotlyHeikinAshi:
    def test_basic(self, sample_ohlcv: pd.DataFrame) -> None:
        from wraquant.viz.candlestick import plotly_heikin_ashi

        fig = plotly_heikin_ashi(sample_ohlcv)
        assert isinstance(fig, go.Figure)
