"""Tests for event study methodology."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.econometrics.event_study import (
    buy_and_hold_abnormal_return,
    cumulative_abnormal_return,
    event_study,
)


def _make_return_series(
    n: int = 500,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Generate daily stock and market return series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    market = pd.Series(rng.normal(0.0005, 0.01, n), index=dates, name="market")
    stock = 0.001 + 1.2 * market + pd.Series(
        rng.normal(0, 0.005, n), index=dates
    )
    stock.name = "stock"
    return stock, market


class TestEventStudy:
    def test_output_structure(self) -> None:
        stock, market = _make_return_series()
        event_dates = [stock.index[300]]

        result = event_study(
            stock,
            event_dates,
            estimation_window=(-250, -10),
            event_window=(-5, 5),
            market_returns=market,
        )

        assert "abnormal_returns" in result
        assert "car" in result
        assert "mean_car" in result
        assert "t_stat" in result
        assert "p_value" in result
        assert "event_dates" in result
        assert "n_events" in result
        assert result["n_events"] == 1

    def test_ar_shape(self) -> None:
        stock, market = _make_return_series()
        event_dates = [stock.index[300]]

        result = event_study(
            stock,
            event_dates,
            event_window=(-5, 5),
            market_returns=market,
        )

        # Event window of (-5, 5) = 11 days
        ar = result["abnormal_returns"]
        assert ar.shape[1] == 11

    def test_multiple_events(self) -> None:
        stock, market = _make_return_series()
        event_dates = [stock.index[300], stock.index[400]]

        result = event_study(
            stock,
            event_dates,
            event_window=(-3, 3),
            market_returns=market,
        )

        assert result["n_events"] == 2
        assert len(result["car"]) == 2

    def test_constant_mean_model(self) -> None:
        """Event study without market returns uses constant-mean model."""
        stock, _ = _make_return_series()
        event_dates = [stock.index[300]]

        result = event_study(
            stock,
            event_dates,
            estimation_window=(-200, -10),
            event_window=(-3, 3),
        )

        assert result["n_events"] == 1
        assert "abnormal_returns" in result

    def test_abnormal_returns_near_zero_for_random_data(self) -> None:
        """For random data with no event, AR should be centered near zero."""
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)
        returns = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)
        market = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)

        # Pick several "event" dates in random data
        event_dates = [dates[300], dates[350], dates[400]]
        result = event_study(
            returns,
            event_dates,
            event_window=(-2, 2),
            market_returns=market,
        )

        # Mean CAR should be close to zero
        assert abs(result["mean_car"]) < 0.1

    def test_empty_result_for_no_valid_events(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=50)
        returns = pd.Series(rng.normal(0, 0.01, 50), index=dates)

        # Event date outside the range
        result = event_study(
            returns,
            [pd.Timestamp("2025-01-01")],
            estimation_window=(-250, -10),
            event_window=(-5, 5),
        )

        assert result["n_events"] == 0


class TestCumulativeAbnormalReturn:
    def test_with_known_values(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=5)
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=dates)
        expected = pd.Series([0.005, 0.005, 0.005, 0.005, 0.005], index=dates)

        car = cumulative_abnormal_return(returns, expected)

        # AR = [0.005, 0.015, -0.015, 0.025, 0.005]
        # CAR = [0.005, 0.020, 0.005, 0.030, 0.035]
        expected_car = [0.005, 0.020, 0.005, 0.030, 0.035]
        np.testing.assert_allclose(car.values, expected_car, atol=1e-10)

    def test_with_event_window(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=10)
        returns = pd.Series(np.ones(10) * 0.01, index=dates)
        expected = pd.Series(np.ones(10) * 0.005, index=dates)

        car = cumulative_abnormal_return(returns, expected, event_window=(2, 5))
        assert len(car) == 4


class TestBuyAndHoldAbnormalReturn:
    def test_known_values(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=3)
        returns = pd.Series([0.10, 0.05, -0.02], index=dates)
        benchmark = pd.Series([0.02, 0.01, 0.01], index=dates)

        bhar = buy_and_hold_abnormal_return(returns, benchmark)

        # BHAR = (1.1 * 1.05 * 0.98) - (1.02 * 1.01 * 1.01)
        expected = (1.10 * 1.05 * 0.98) - (1.02 * 1.01 * 1.01)
        assert abs(bhar - expected) < 1e-10

    def test_with_event_window(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=5)
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=dates)
        benchmark = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01], index=dates)

        bhar = buy_and_hold_abnormal_return(returns, benchmark, event_window=(1, 3))

        expected = np.prod(1 + returns.iloc[1:4].values) - np.prod(
            1 + benchmark.iloc[1:4].values
        )
        assert abs(bhar - expected) < 1e-10
