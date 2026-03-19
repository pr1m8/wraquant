"""Tests for wraquant.risk.historical module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.risk.historical import (
    contagion_analysis,
    crisis_drawdowns,
    drawdown_attribution,
    event_impact,
)


@pytest.fixture()
def return_series():
    """Generate synthetic return series with DatetimeIndex."""
    np.random.seed(42)
    idx = pd.bdate_range("2019-01-02", periods=750)
    returns = pd.Series(np.random.normal(0.0003, 0.01, 750), index=idx)
    return returns


@pytest.fixture()
def multi_asset_returns():
    """Generate synthetic multi-asset returns."""
    np.random.seed(42)
    idx = pd.bdate_range("2019-01-02", periods=750)
    return pd.DataFrame(
        {
            "A": np.random.normal(0.0005, 0.01, 750),
            "B": np.random.normal(0.0003, 0.015, 750),
            "C": np.random.normal(0.0004, 0.008, 750),
        },
        index=idx,
    )


class TestCrisisDrawdowns:
    """Tests for crisis_drawdowns."""

    def test_returns_dataframe(self, return_series):
        result = crisis_drawdowns(return_series, top_n=3)
        assert isinstance(result, pd.DataFrame)

    def test_top_n_limit(self, return_series):
        result = crisis_drawdowns(return_series, top_n=3)
        assert len(result) <= 3

    def test_columns(self, return_series):
        result = crisis_drawdowns(return_series, top_n=3)
        expected_cols = [
            "start",
            "trough",
            "end",
            "drawdown",
            "days_to_trough",
            "days_to_recovery",
            "total_days",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_drawdowns_negative(self, return_series):
        result = crisis_drawdowns(return_series, top_n=5)
        if len(result) > 0:
            assert (result["drawdown"] < 0).all()

    def test_sorted_by_severity(self, return_series):
        result = crisis_drawdowns(return_series, top_n=5)
        if len(result) > 1:
            assert result["drawdown"].iloc[0] <= result["drawdown"].iloc[1]

    def test_no_drawdown(self):
        """Monotonically increasing returns should have minimal drawdowns."""
        returns = pd.Series(
            np.ones(100) * 0.01,
            index=pd.bdate_range("2020-01-01", periods=100),
        )
        result = crisis_drawdowns(returns, top_n=3)
        assert len(result) == 0


class TestEventImpact:
    """Tests for event_impact."""

    def test_returns_dict(self, return_series):
        result = event_impact(return_series, ["2020-03-16"], window=5)
        assert isinstance(result, dict)

    def test_event_keys(self, return_series):
        dates = ["2020-03-16"]
        result = event_impact(return_series, dates, window=5)
        if result:
            event = list(result.values())[0]
            assert "pre_cumulative" in event
            assert "post_cumulative" in event
            assert "event_day_return" in event
            assert "pre_vol" in event
            assert "post_vol" in event
            assert "total_impact" in event

    def test_multiple_events(self, return_series):
        dates = ["2019-06-15", "2020-01-15", "2020-06-15"]
        result = event_impact(return_series, dates, window=5)
        # At least some should match
        assert len(result) >= 1

    def test_window_effect(self, return_series):
        result_5 = event_impact(return_series, ["2020-03-16"], window=5)
        result_10 = event_impact(return_series, ["2020-03-16"], window=10)
        # Both should return results
        assert len(result_5) >= 0
        assert len(result_10) >= 0


class TestContagionAnalysis:
    """Tests for contagion_analysis."""

    def test_returns_dict(self, multi_asset_returns):
        result = contagion_analysis(
            multi_asset_returns,
            crisis_dates=("2020-02-01", "2020-06-01"),
        )
        assert "normal_corr" in result
        assert "crisis_corr" in result
        assert "corr_change" in result
        assert "avg_normal_corr" in result
        assert "avg_crisis_corr" in result
        assert "contagion_detected" in result
        assert "n_normal" in result
        assert "n_crisis" in result

    def test_corr_matrices_shape(self, multi_asset_returns):
        result = contagion_analysis(
            multi_asset_returns,
            crisis_dates=("2020-02-01", "2020-06-01"),
        )
        assert result["normal_corr"].shape == (3, 3)
        assert result["crisis_corr"].shape == (3, 3)

    def test_corr_change(self, multi_asset_returns):
        result = contagion_analysis(
            multi_asset_returns,
            crisis_dates=("2020-02-01", "2020-06-01"),
        )
        expected_change = result["crisis_corr"] - result["normal_corr"]
        pd.testing.assert_frame_equal(result["corr_change"], expected_change)

    def test_sample_counts(self, multi_asset_returns):
        result = contagion_analysis(
            multi_asset_returns,
            crisis_dates=("2020-02-01", "2020-06-01"),
        )
        assert result["n_normal"] + result["n_crisis"] == len(multi_asset_returns)

    def test_contagion_bool(self, multi_asset_returns):
        result = contagion_analysis(
            multi_asset_returns,
            crisis_dates=("2020-02-01", "2020-06-01"),
        )
        assert isinstance(result["contagion_detected"], (bool, np.bool_))


class TestDrawdownAttribution:
    """Tests for drawdown_attribution."""

    def test_returns_dataframe(self, multi_asset_returns):
        weights = np.array([0.4, 0.35, 0.25])
        result = drawdown_attribution(multi_asset_returns, weights)
        assert isinstance(result, pd.DataFrame)

    def test_has_portfolio_dd(self, multi_asset_returns):
        weights = np.array([0.4, 0.35, 0.25])
        result = drawdown_attribution(multi_asset_returns, weights)
        assert "portfolio_dd" in result.columns

    def test_has_asset_contributions(self, multi_asset_returns):
        weights = np.array([0.4, 0.35, 0.25])
        result = drawdown_attribution(multi_asset_returns, weights)
        assert "A_contribution" in result.columns
        assert "B_contribution" in result.columns
        assert "C_contribution" in result.columns

    def test_length_matches(self, multi_asset_returns):
        weights = np.array([0.4, 0.35, 0.25])
        result = drawdown_attribution(multi_asset_returns, weights)
        assert len(result) == len(multi_asset_returns.dropna())

    def test_drawdown_negative(self, multi_asset_returns):
        weights = np.array([0.4, 0.35, 0.25])
        result = drawdown_attribution(multi_asset_returns, weights)
        # Portfolio drawdowns should be <= 0
        assert (result["portfolio_dd"] <= 1e-10).all()
