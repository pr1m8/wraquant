"""Tests for seasonality detection and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.ts.seasonality import (
    detect_seasonality,
    fourier_features,
    multi_fourier_features,
    multi_seasonal_decompose,
    seasonal_strength,
)


def _make_seasonal_series(n: int = 400, period: int = 20, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    trend = 0.02 * t
    seasonal = 5 * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0, 0.3, n)
    return pd.Series(trend + seasonal + noise, name="y")


# ---------------------------------------------------------------------------
# Fourier Features
# ---------------------------------------------------------------------------


class TestFourierFeatures:
    def test_correct_shape(self) -> None:
        idx = pd.date_range("2020-01-01", periods=365, freq="D")
        df = fourier_features(idx, period=7, n_harmonics=3)
        assert df.shape == (365, 6)  # 3 sin + 3 cos

    def test_columns_named_correctly(self) -> None:
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        df = fourier_features(idx, period=12, n_harmonics=2)
        assert "sin_1" in df.columns
        assert "cos_1" in df.columns
        assert "sin_2" in df.columns
        assert "cos_2" in df.columns

    def test_values_bounded(self) -> None:
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        df = fourier_features(idx, period=7, n_harmonics=5)
        assert df.min().min() >= -1.0 - 1e-10
        assert df.max().max() <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Multi-Fourier Features
# ---------------------------------------------------------------------------


class TestMultiFourierFeatures:
    def test_correct_shape_multiple_periods(self) -> None:
        idx = pd.date_range("2020-01-01", periods=365, freq="D")
        df = multi_fourier_features(idx, periods=[7, 365], n_harmonics=3)
        # 2 periods * 3 harmonics * 2 (sin+cos) = 12
        assert df.shape == (365, 12)

    def test_column_naming(self) -> None:
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        df = multi_fourier_features(idx, periods=[7, 30], n_harmonics=2)
        assert "sin_P7_H1" in df.columns
        assert "cos_P30_H2" in df.columns

    def test_per_period_harmonics(self) -> None:
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        df = multi_fourier_features(idx, periods=[7, 30], n_harmonics=[2, 4])
        # 2*2 + 4*2 = 12
        assert df.shape[1] == 12


# ---------------------------------------------------------------------------
# Seasonal Strength
# ---------------------------------------------------------------------------


class TestSeasonalStrength:
    def test_strong_seasonal_near_one(self) -> None:
        """A pure seasonal signal should have strength near 1."""
        t = np.arange(200, dtype=float)
        pure = pd.Series(10 * np.sin(2 * np.pi * t / 20))
        strength = seasonal_strength(pure, period=20)
        assert strength > 0.9

    def test_noise_near_zero(self) -> None:
        """Pure white noise should have low seasonal strength."""
        rng = np.random.default_rng(42)
        noise = pd.Series(rng.normal(0, 1, 500))
        strength = seasonal_strength(noise, period=20)
        assert strength < 0.5

    def test_returns_float(self) -> None:
        data = _make_seasonal_series()
        strength = seasonal_strength(data, period=20)
        assert isinstance(strength, float)
        assert 0.0 <= strength <= 1.0


# ---------------------------------------------------------------------------
# Multi-Seasonal Decomposition
# ---------------------------------------------------------------------------


class TestMultiSeasonalDecompose:
    def test_returns_all_keys(self) -> None:
        t = np.arange(500, dtype=float)
        weekly = 3 * np.sin(2 * np.pi * t / 7)
        trend = 0.01 * t
        data = pd.Series(trend + weekly)
        result = multi_seasonal_decompose(data, periods=[7])
        assert "trend" in result
        assert "seasonal" in result
        assert "residual" in result
        assert 7 in result["seasonal"]

    def test_multiple_periods(self) -> None:
        # Need >2*365=730 observations for yearly seasonality
        t = np.arange(1100, dtype=float)
        weekly = 3 * np.sin(2 * np.pi * t / 7)
        yearly = 5 * np.sin(2 * np.pi * t / 365)
        data = pd.Series(0.01 * t + weekly + yearly)
        result = multi_seasonal_decompose(data, periods=[7, 365])
        assert 7 in result["seasonal"]
        assert 365 in result["seasonal"]
