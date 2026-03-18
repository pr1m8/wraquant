"""Tests for execution algorithm schedules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.execution.algorithms import (
    arrival_price_benchmark,
    implementation_shortfall,
    participation_rate_schedule,
    twap_schedule,
    vwap_schedule,
)


class TestTWAPSchedule:
    def test_equal_slices(self) -> None:
        result = twap_schedule(1000.0, 10)
        np.testing.assert_allclose(result, 100.0)

    def test_sums_to_total(self) -> None:
        result = twap_schedule(5000.0, 7)
        np.testing.assert_allclose(np.sum(result), 5000.0)

    def test_length(self) -> None:
        result = twap_schedule(1000.0, 25)
        assert len(result) == 25

    def test_invalid_intervals(self) -> None:
        with pytest.raises(ValueError):
            twap_schedule(1000.0, 0)


class TestVWAPSchedule:
    def test_proportional_to_volume(self) -> None:
        profile = np.array([100, 200, 300, 400])
        result = vwap_schedule(1000.0, profile)
        np.testing.assert_allclose(result, [100, 200, 300, 400])

    def test_sums_to_total(self) -> None:
        rng = np.random.default_rng(42)
        profile = rng.uniform(100, 1000, 20)
        result = vwap_schedule(5000.0, profile)
        np.testing.assert_allclose(np.sum(result), 5000.0)

    def test_zero_volume_raises(self) -> None:
        with pytest.raises(ValueError):
            vwap_schedule(1000.0, np.zeros(5))

    def test_pandas_input(self) -> None:
        profile = pd.Series([100, 200, 300])
        result = vwap_schedule(600.0, profile)
        np.testing.assert_allclose(result, [100, 200, 300])


class TestImplementationShortfall:
    def test_zero_shortfall(self) -> None:
        exec_prices = np.array([100.0, 100.0, 100.0])
        result = implementation_shortfall(exec_prices, 100.0, 100.0)
        assert result["total_is"] == pytest.approx(0.0)

    def test_positive_shortfall_for_buys(self) -> None:
        exec_prices = np.array([101.0, 102.0, 103.0])
        result = implementation_shortfall(exec_prices, 100.0, 105.0)
        assert result["total_is"] > 0

    def test_decomposition_sums(self) -> None:
        exec_prices = np.array([101.0, 102.0, 103.0])
        result = implementation_shortfall(exec_prices, 100.0, 105.0)
        approx_sum = result["delay_cost"] + result["trading_impact"]
        assert result["total_is"] == pytest.approx(approx_sum, rel=1e-10)


class TestParticipationRateSchedule:
    def test_respects_rate(self) -> None:
        expected_vol = np.array([1000, 2000, 1500, 500])
        result = participation_rate_schedule(500.0, 0.1, expected_vol)
        # Each interval capped at rate * expected_vol
        assert np.all(result <= 0.1 * expected_vol + 1e-10)

    def test_sums_to_at_most_total(self) -> None:
        expected_vol = np.array([10000, 20000, 15000])
        result = participation_rate_schedule(1000.0, 0.5, expected_vol)
        assert np.sum(result) <= 1000.0 + 1e-10

    def test_invalid_rate(self) -> None:
        with pytest.raises(ValueError):
            participation_rate_schedule(1000.0, 0.0, np.ones(5))


class TestArrivalPriceBenchmark:
    def test_zero_cost_at_arrival(self) -> None:
        exec_prices = np.array([100.0, 100.0])
        volumes = np.array([500, 500])
        result = arrival_price_benchmark(exec_prices, volumes, 100.0)
        assert result["arrival_cost"] == pytest.approx(0.0)

    def test_positive_cost(self) -> None:
        exec_prices = np.array([101.0, 102.0])
        volumes = np.array([500, 500])
        result = arrival_price_benchmark(exec_prices, volumes, 100.0)
        assert result["arrival_cost"] > 0
        assert result["arrival_cost_bps"] > 0
