"""Tests for stationarity transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.ts.stationarity import detrend, difference, fractional_difference


def _make_series(n: int = 200, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    values = np.cumsum(rng.normal(0, 1, n))
    return pd.Series(values, index=dates, name="x")


class TestDifference:
    def test_first_order(self) -> None:
        data = _make_series()
        result = difference(data, order=1)
        # First difference of a random walk should remove the unit root
        assert len(result) == len(data) - 1

    def test_second_order(self) -> None:
        data = _make_series()
        result = difference(data, order=2)
        assert len(result) == len(data) - 2

    def test_no_nans(self) -> None:
        data = _make_series()
        result = difference(data, order=1)
        assert not result.isna().any()


class TestFractionalDifference:
    def test_returns_series(self) -> None:
        data = _make_series()
        result = fractional_difference(data, d=0.5)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_d_zero_approximates_original(self) -> None:
        data = _make_series()
        # d=0 should keep values close to original (identity transform)
        result = fractional_difference(data, d=0.0, threshold=1e-5)
        # With d=0 the only weight is 1.0 so result equals original
        assert len(result) == len(data)
        np.testing.assert_allclose(result.values, data.values, atol=1e-10)

    def test_shorter_than_original(self) -> None:
        data = _make_series()
        result = fractional_difference(data, d=0.5, threshold=1e-3)
        # fracdiff reduces length because of the window
        assert len(result) <= len(data)


class TestDetrend:
    def test_linear_detrend(self) -> None:
        # Linearly trending series should be centered after detrend
        t = np.arange(100, dtype=float)
        data = pd.Series(3 * t + 10 + np.random.default_rng(42).normal(0, 0.1, 100))
        result = detrend(data, method="linear")
        assert abs(result.mean()) < 1.0

    def test_constant_detrend(self) -> None:
        data = pd.Series([5.0, 6.0, 7.0, 8.0, 9.0])
        result = detrend(data, method="constant")
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-10)

    def test_preserves_length(self) -> None:
        data = _make_series()
        result = detrend(data, method="linear")
        assert len(result) == len(data)


# ---------------------------------------------------------------------------
# ADF Test
# ---------------------------------------------------------------------------

from wraquant.ts.stationarity import (
    adf_test,
    kpss_test,
    optimal_differencing,
    phillips_perron,
    variance_ratio_test,
)


class TestADFTest:
    def test_detects_nonstationary_random_walk(self) -> None:
        """A random walk should fail the ADF test (non-stationary)."""
        rng = np.random.default_rng(42)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
        result = adf_test(rw)
        assert result["is_stationary"] is False
        assert result["p_value"] > 0.05

    def test_detects_stationary_white_noise(self) -> None:
        """White noise should pass the ADF test (stationary)."""
        rng = np.random.default_rng(42)
        wn = pd.Series(rng.normal(0, 1, 500))
        result = adf_test(wn)
        assert result["is_stationary"] is True
        assert result["p_value"] < 0.05

    def test_output_keys(self) -> None:
        data = _make_series()
        result = adf_test(data)
        assert "test_statistic" in result
        assert "p_value" in result
        assert "critical_values" in result
        assert "optimal_lag" in result
        assert "is_stationary" in result
        assert "interpretation" in result

    def test_critical_values_present(self) -> None:
        data = _make_series()
        result = adf_test(data)
        cv = result["critical_values"]
        assert "1%" in cv
        assert "5%" in cv
        assert "10%" in cv


# ---------------------------------------------------------------------------
# KPSS Test
# ---------------------------------------------------------------------------


class TestKPSSTest:
    def test_detects_stationary_white_noise(self) -> None:
        """White noise should pass KPSS (null = stationary)."""
        rng = np.random.default_rng(42)
        wn = pd.Series(rng.normal(0, 1, 500))
        result = kpss_test(wn)
        assert result["is_stationary"] is True

    def test_detects_nonstationary(self) -> None:
        """A random walk should fail KPSS."""
        rng = np.random.default_rng(42)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
        result = kpss_test(rw)
        assert result["is_stationary"] is False

    def test_output_keys(self) -> None:
        data = _make_series()
        result = kpss_test(data)
        assert "test_statistic" in result
        assert "p_value" in result
        assert "n_lags" in result
        assert "interpretation" in result


# ---------------------------------------------------------------------------
# Optimal Differencing
# ---------------------------------------------------------------------------


class TestOptimalDifferencing:
    def test_random_walk_needs_d1(self) -> None:
        """A random walk should require d=1."""
        rng = np.random.default_rng(42)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
        result = optimal_differencing(rw)
        assert result["optimal_d"] == 1
        assert result["is_stationary"] is True

    def test_stationary_needs_d0(self) -> None:
        """White noise should need d=0."""
        rng = np.random.default_rng(42)
        wn = pd.Series(rng.normal(0, 1, 500))
        result = optimal_differencing(wn)
        assert result["optimal_d"] == 0

    def test_test_results_per_d(self) -> None:
        rng = np.random.default_rng(42)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
        result = optimal_differencing(rw)
        assert 0 in result["test_results"]
        assert 1 in result["test_results"]


# ---------------------------------------------------------------------------
# Phillips-Perron Test
# ---------------------------------------------------------------------------


class TestPhillipsPerron:
    def test_detects_stationary(self) -> None:
        rng = np.random.default_rng(42)
        wn = pd.Series(rng.normal(0, 1, 500))
        result = phillips_perron(wn)
        assert result["is_stationary"] is True

    def test_output_keys(self) -> None:
        data = _make_series()
        result = phillips_perron(data)
        assert "test_statistic" in result
        assert "p_value" in result
        assert "n_lags" in result
        assert "interpretation" in result


# ---------------------------------------------------------------------------
# Variance Ratio Test
# ---------------------------------------------------------------------------


class TestVarianceRatioTest:
    def test_random_walk_vr_near_one(self) -> None:
        rng = np.random.default_rng(42)
        # Use exp(cumsum) to get positive price-like data
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000))))
        result = variance_ratio_test(prices, lags=2)
        assert 0.5 < result["variance_ratio"] < 1.5

    def test_output_keys(self) -> None:
        rng = np.random.default_rng(42)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
        result = variance_ratio_test(prices, lags=4)
        assert "variance_ratio" in result
        assert "z_statistic" in result
        assert "z_robust" in result
        assert "p_value" in result
        assert "is_random_walk" in result
