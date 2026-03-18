"""Smoke tests for wraquant.viz plotting functions.

Verifies that each plotting function runs without exceptions and returns
the expected matplotlib types.  Visual correctness is not tested.
"""

from __future__ import annotations

import matplotlib
import matplotlib.axes
import matplotlib.figure

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures() -> None:  # noqa: PT004
    """Close all matplotlib figures after every test."""
    yield
    plt.close("all")


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
def sample_weights() -> pd.Series:
    """Portfolio weights."""
    return pd.Series(
        [0.4, 0.25, 0.15, 0.1, 0.05, 0.05],
        index=["SPY", "AGG", "GLD", "VWO", "TLT", "TIPS"],
    )


@pytest.fixture
def sample_corr() -> pd.DataFrame:
    """Small correlation matrix."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(252, 4))
    df = pd.DataFrame(data, columns=["A", "B", "C", "D"])
    return df.corr()


# ---------------------------------------------------------------------------
# Theme tests
# ---------------------------------------------------------------------------


class TestThemes:
    def test_set_wraquant_style(self) -> None:
        from wraquant.viz.themes import set_wraquant_style

        set_wraquant_style()
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False

    def test_apply_theme(self) -> None:
        from wraquant.viz.themes import apply_theme

        fig, ax = plt.subplots()
        apply_theme(fig, ax)
        assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)

    def test_colors_dict(self) -> None:
        from wraquant.viz.themes import COLORS

        assert isinstance(COLORS, dict)
        assert "primary" in COLORS
        assert "positive" in COLORS
        assert "negative" in COLORS


# ---------------------------------------------------------------------------
# Returns plot tests
# ---------------------------------------------------------------------------


class TestReturnsPlots:
    def test_plot_cumulative_returns(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.returns import plot_cumulative_returns

        ax = plot_cumulative_returns(sample_returns)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_cumulative_returns_with_benchmark(
        self, sample_returns: pd.Series, sample_benchmark: pd.Series
    ) -> None:
        from wraquant.viz.returns import plot_cumulative_returns

        ax = plot_cumulative_returns(sample_returns, benchmark=sample_benchmark)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_cumulative_returns_custom_title(
        self, sample_returns: pd.Series
    ) -> None:
        from wraquant.viz.returns import plot_cumulative_returns

        ax = plot_cumulative_returns(sample_returns, title="My Title")
        assert ax.get_title() == "My Title"

    def test_plot_cumulative_returns_existing_ax(
        self, sample_returns: pd.Series
    ) -> None:
        from wraquant.viz.returns import plot_cumulative_returns

        fig, provided_ax = plt.subplots()
        ax = plot_cumulative_returns(sample_returns, ax=provided_ax)
        assert ax is provided_ax

    def test_plot_drawdowns(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.returns import plot_drawdowns

        ax = plot_drawdowns(sample_returns)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_drawdowns_top_n(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.returns import plot_drawdowns

        ax = plot_drawdowns(sample_returns, top_n=3)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_return_distribution(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.returns import plot_return_distribution

        ax = plot_return_distribution(sample_returns)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_return_distribution_no_fit(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.returns import plot_return_distribution

        ax = plot_return_distribution(sample_returns, fit_normal=False, bins=30)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_rolling_returns(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.returns import plot_rolling_returns

        ax = plot_rolling_returns(sample_returns, window=63)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_monthly_heatmap(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.returns import plot_monthly_heatmap

        ax = plot_monthly_heatmap(sample_returns)
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# Portfolio plot tests
# ---------------------------------------------------------------------------


class TestPortfolioPlots:
    def test_plot_weights_series(self, sample_weights: pd.Series) -> None:
        from wraquant.viz.portfolio import plot_weights

        ax = plot_weights(sample_weights)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_weights_array(self) -> None:
        from wraquant.viz.portfolio import plot_weights

        ax = plot_weights(np.array([0.5, 0.3, 0.2]), names=["A", "B", "C"])
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_weights_existing_ax(self, sample_weights: pd.Series) -> None:
        from wraquant.viz.portfolio import plot_weights

        fig, provided_ax = plt.subplots()
        ax = plot_weights(sample_weights, ax=provided_ax)
        assert ax is provided_ax

    def test_plot_efficient_frontier_line(self) -> None:
        from wraquant.viz.portfolio import plot_efficient_frontier

        vols = np.linspace(0.05, 0.3, 50)
        rets = 0.02 + 0.5 * vols
        ax = plot_efficient_frontier(rets, vols)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_efficient_frontier_scatter(self) -> None:
        from wraquant.viz.portfolio import plot_efficient_frontier

        rng = np.random.default_rng(42)
        vols = rng.uniform(0.05, 0.3, 200)
        rets = 0.02 + 0.5 * vols + rng.normal(0, 0.01, 200)
        sharpes = rets / vols
        ax = plot_efficient_frontier(rets, vols, sharpe_range=sharpes)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_efficient_frontier_optimal(self) -> None:
        from wraquant.viz.portfolio import plot_efficient_frontier

        vols = np.linspace(0.05, 0.3, 50)
        rets = 0.02 + 0.5 * vols
        ax = plot_efficient_frontier(rets, vols, optimal_point=(0.15, 0.095))
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_risk_contribution(self) -> None:
        from wraquant.viz.portfolio import plot_risk_contribution

        contribs = pd.Series(
            [0.3, 0.25, 0.2, 0.15, 0.1], index=["A", "B", "C", "D", "E"]
        )
        ax = plot_risk_contribution(contribs)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_correlation_matrix_dataframe(self, sample_corr: pd.DataFrame) -> None:
        from wraquant.viz.portfolio import plot_correlation_matrix

        ax = plot_correlation_matrix(sample_corr)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_correlation_matrix_array(self) -> None:
        from wraquant.viz.portfolio import plot_correlation_matrix

        corr = np.eye(3)
        ax = plot_correlation_matrix(corr, labels=["X", "Y", "Z"])
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# Time series plot tests
# ---------------------------------------------------------------------------


class TestTimeseriesPlots:
    def test_plot_series_single(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.timeseries import plot_series

        ax = plot_series(sample_returns, title="Returns", ylabel="Return")
        assert isinstance(ax, matplotlib.axes.Axes)
        assert ax.get_title() == "Returns"

    def test_plot_series_dataframe(self) -> None:
        from wraquant.viz.timeseries import plot_series

        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame(
            rng.normal(size=(100, 3)), index=dates, columns=["A", "B", "C"]
        )
        ax = plot_series(df)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_series_existing_ax(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.timeseries import plot_series

        fig, provided_ax = plt.subplots()
        ax = plot_series(sample_returns, ax=provided_ax)
        assert ax is provided_ax

    def test_plot_regime_overlay(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.timeseries import plot_regime_overlay

        rng = np.random.default_rng(42)
        regimes = pd.Series(
            rng.choice([0, 1, 2], size=len(sample_returns)),
            index=sample_returns.index,
        )
        ax = plot_regime_overlay(sample_returns, regimes)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_decomposition(self) -> None:
        from wraquant.viz.timeseries import plot_decomposition

        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=252)
        trend = pd.Series(np.linspace(100, 120, 252), index=dates)
        seasonal = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 252)), index=dates)
        residual = pd.Series(rng.normal(0, 0.5, 252), index=dates)

        result = plot_decomposition(trend, seasonal, residual)
        assert isinstance(result, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# Risk plot tests
# ---------------------------------------------------------------------------


class TestRiskPlots:
    def test_plot_var_backtest_scalar(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.risk import plot_var_backtest

        var_level = float(np.percentile(sample_returns.values, 5))
        ax = plot_var_backtest(sample_returns, var_level)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_var_backtest_series(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.risk import plot_var_backtest

        var_series = sample_returns.rolling(63).quantile(0.05)
        ax = plot_var_backtest(sample_returns, var_series, confidence=0.95)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_var_backtest_existing_ax(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.risk import plot_var_backtest

        fig, provided_ax = plt.subplots()
        ax = plot_var_backtest(sample_returns, -0.02, ax=provided_ax)
        assert ax is provided_ax

    def test_plot_rolling_volatility(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.risk import plot_rolling_volatility

        ax = plot_rolling_volatility(sample_returns, window=21)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_rolling_volatility_no_annualize(
        self, sample_returns: pd.Series
    ) -> None:
        from wraquant.viz.risk import plot_rolling_volatility

        ax = plot_rolling_volatility(sample_returns, window=21, annualize=False)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_tail_distribution(self, sample_returns: pd.Series) -> None:
        from wraquant.viz.risk import plot_tail_distribution

        ax = plot_tail_distribution(sample_returns, threshold_percentile=5)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_tail_distribution_custom_threshold(
        self, sample_returns: pd.Series
    ) -> None:
        from wraquant.viz.risk import plot_tail_distribution

        ax = plot_tail_distribution(sample_returns, threshold_percentile=10)
        assert isinstance(ax, matplotlib.axes.Axes)
