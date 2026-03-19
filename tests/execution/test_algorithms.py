"""Tests for execution algorithm schedules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.execution.algorithms import (
    arrival_price_benchmark,
    close_auction_allocation,
    implementation_shortfall,
    is_schedule,
    participation_rate_schedule,
    pov_schedule,
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


class TestISSchedule:
    def test_sums_to_total(self) -> None:
        volumes = np.array([1000, 2000, 1500, 3000, 2500])
        schedule = is_schedule(10_000, volumes, alpha=0.5)
        np.testing.assert_allclose(schedule.sum(), 10_000)

    def test_alpha_zero_is_vwap(self) -> None:
        volumes = np.array([100, 200, 300, 400])
        schedule = is_schedule(1000.0, volumes, alpha=0.0)
        expected = vwap_schedule(1000.0, volumes)
        np.testing.assert_allclose(schedule, expected)

    def test_alpha_one_is_twap(self) -> None:
        volumes = np.array([100, 200, 300, 400])
        schedule = is_schedule(1000.0, volumes, alpha=1.0)
        expected = twap_schedule(1000.0, 4)
        np.testing.assert_allclose(schedule, expected)

    def test_length_matches_volumes(self) -> None:
        volumes = np.array([1000, 2000, 1500])
        schedule = is_schedule(5000, volumes)
        assert len(schedule) == 3

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError):
            is_schedule(1000, np.ones(5), alpha=1.5)


class TestPOVSchedule:
    def test_respects_rate(self) -> None:
        volumes = np.array([5000, 8000, 6000, 10000, 7000])
        schedule = pov_schedule(2000, volumes, pov_rate=0.10)
        assert schedule[0] == pytest.approx(500.0)

    def test_does_not_exceed_total(self) -> None:
        volumes = np.array([5000, 8000, 6000, 10000, 7000])
        schedule = pov_schedule(2000, volumes, pov_rate=0.10)
        assert schedule.sum() <= 2000.0 + 1e-10

    def test_invalid_rate(self) -> None:
        with pytest.raises(ValueError):
            pov_schedule(1000, np.ones(5), pov_rate=0.0)

    def test_exhausts_when_enough_volume(self) -> None:
        volumes = np.array([50000, 50000, 50000])
        schedule = pov_schedule(1000, volumes, pov_rate=0.50)
        assert schedule.sum() == pytest.approx(1000.0)


class TestCloseAuctionAllocation:
    def test_basic_split(self) -> None:
        result = close_auction_allocation(10_000, 0.15)
        assert result["close_quantity"] == pytest.approx(1500.0)
        assert result["continuous_quantity"] == pytest.approx(8500.0)

    def test_sums_to_total(self) -> None:
        result = close_auction_allocation(10_000, 0.20)
        total = result["close_quantity"] + result["continuous_quantity"]
        assert total == pytest.approx(10_000.0)

    def test_zero_close(self) -> None:
        result = close_auction_allocation(10_000, 0.0)
        assert result["close_quantity"] == pytest.approx(0.0)

    def test_full_close(self) -> None:
        result = close_auction_allocation(10_000, 1.0)
        assert result["close_quantity"] == pytest.approx(10_000.0)

    def test_invalid_pct(self) -> None:
        with pytest.raises(ValueError):
            close_auction_allocation(10_000, 1.5)

    def test_returns_dict_keys(self) -> None:
        result = close_auction_allocation(10_000)
        assert "continuous_quantity" in result
        assert "close_quantity" in result
        assert "close_pct" in result
