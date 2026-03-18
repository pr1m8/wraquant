"""Tests for wraquant.math.signals."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.math.signals import (
    exponential_smooth,
    hodrick_prescott,
    kalman_smooth,
    median_filter,
    savitzky_golay,
)


class TestSavitzkyGolay:
    """Tests for savitzky_golay."""

    def test_smooths_noisy_sine(self) -> None:
        """Smoothed output should be closer to the true signal than raw."""
        rng = np.random.default_rng(42)
        n = 200
        t = np.linspace(0, 4 * np.pi, n)
        true_signal = np.sin(t)
        noisy = true_signal + rng.normal(0, 0.3, n)

        smoothed = savitzky_golay(noisy, window=15, polyorder=3)

        error_raw = np.mean((noisy - true_signal) ** 2)
        error_smooth = np.mean((smoothed - true_signal) ** 2)
        assert error_smooth < error_raw

    def test_output_shape(self) -> None:
        data = np.random.default_rng(0).standard_normal(100)
        smoothed = savitzky_golay(data)
        assert smoothed.shape == data.shape


class TestKalmanSmooth:
    """Tests for kalman_smooth."""

    def test_output_shape_matches_input(self) -> None:
        data = np.random.default_rng(42).standard_normal(200)
        smoothed = kalman_smooth(data)
        assert smoothed.shape == data.shape

    def test_reduces_noise(self) -> None:
        rng = np.random.default_rng(42)
        true_val = 5.0
        data = true_val + rng.normal(0, 1, 500)
        smoothed = kalman_smooth(data, process_var=1e-5, measurement_var=1.0)
        # The tail of smoothed should be closer to true_val
        assert (
            abs(smoothed[-1] - true_val) < abs(data[-1] - true_val)
            or abs(smoothed[-1] - true_val) < 0.5
        )


class TestMedianFilter:
    """Tests for median_filter."""

    def test_removes_spikes(self) -> None:
        """Median filter should remove isolated spikes."""
        data = np.ones(100)
        data[50] = 100.0  # spike
        filtered = median_filter(data, kernel_size=5)
        assert filtered[50] == pytest.approx(1.0)

    def test_output_shape(self) -> None:
        data = np.random.default_rng(0).standard_normal(100)
        filtered = median_filter(data)
        assert filtered.shape == data.shape


class TestExponentialSmooth:
    """Tests for exponential_smooth."""

    def test_output_shape(self) -> None:
        data = np.random.default_rng(0).standard_normal(100)
        smoothed = exponential_smooth(data)
        assert smoothed.shape == data.shape

    def test_first_element_unchanged(self) -> None:
        data = np.array([10.0, 20.0, 30.0])
        smoothed = exponential_smooth(data, alpha=0.5)
        assert smoothed[0] == pytest.approx(10.0)


class TestHodrickPrescott:
    """Tests for hodrick_prescott."""

    def test_trend_plus_cycle_equals_original(self) -> None:
        """Trend + cycle should reconstruct the original data."""
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.standard_normal(200))
        trend, cycle = hodrick_prescott(data)
        np.testing.assert_allclose(trend + cycle, data, atol=1e-8)

    def test_output_shapes(self) -> None:
        data = np.random.default_rng(0).standard_normal(100)
        trend, cycle = hodrick_prescott(data)
        assert trend.shape == data.shape
        assert cycle.shape == data.shape

    def test_smooth_trend(self) -> None:
        """The trend should be smoother than the original."""
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.standard_normal(200))
        trend, _ = hodrick_prescott(data, lamb=1600)
        # Second differences of trend should be small
        d2_data = np.diff(data, n=2)
        d2_trend = np.diff(trend, n=2)
        assert np.std(d2_trend) < np.std(d2_data)
