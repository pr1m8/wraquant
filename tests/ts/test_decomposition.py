"""Tests for time series decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.ts.decomposition import seasonal_decompose, stl_decompose, trend_filter


def _make_seasonal_series(n: int = 200, period: int = 20, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    trend = 0.05 * t
    seasonal = 2 * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0, 0.5, n)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(trend + seasonal + noise, index=dates, name="y")


class TestSeasonalDecompose:
    def test_has_components(self) -> None:
        data = _make_seasonal_series()
        result = seasonal_decompose(data, period=20)
        assert hasattr(result, "trend")
        assert hasattr(result, "seasonal")
        assert hasattr(result, "resid")

    def test_trend_length(self) -> None:
        data = _make_seasonal_series()
        result = seasonal_decompose(data, period=20)
        assert len(result.trend) == len(data)


class TestSTLDecompose:
    def test_has_components(self) -> None:
        data = _make_seasonal_series()
        result = stl_decompose(data, period=20)
        assert hasattr(result, "trend")
        assert hasattr(result, "seasonal")
        assert hasattr(result, "resid")


class TestTrendFilter:
    def test_hp_filter(self) -> None:
        data = _make_seasonal_series()
        trend = trend_filter(data, method="hp")
        assert len(trend) == len(data)
        assert isinstance(trend, pd.Series)

    def test_unknown_method(self) -> None:
        data = _make_seasonal_series()
        try:
            trend_filter(data, method="unknown")
            raise AssertionError("Should have raised ValueError")  # noqa: TRY301
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# SSA Decomposition
# ---------------------------------------------------------------------------

from wraquant.ts.decomposition import emd_decompose, ssa_decompose


class TestSSADecompose:
    def test_components_sum_to_original(self) -> None:
        """All SSA components must sum back to the original series."""
        data = _make_seasonal_series(n=200, period=20)
        result = ssa_decompose(data, window=40)
        reconstructed = sum(result["components"].values())
        np.testing.assert_allclose(
            reconstructed.values, data.values, atol=1e-8,
        )

    def test_singular_values_decreasing(self) -> None:
        data = _make_seasonal_series(n=200, period=20)
        result = ssa_decompose(data, window=40, n_components=10)
        sv = result["singular_values"]
        assert len(sv) == 10
        # Singular values should be non-negative and sorted descending
        assert np.all(sv >= 0)
        assert np.all(np.diff(sv) <= 1e-10)  # non-increasing

    def test_explained_variance_sums_near_one(self) -> None:
        data = _make_seasonal_series(n=200, period=20)
        result = ssa_decompose(data, window=40)
        ev = result["explained_variance"]
        assert abs(ev.sum() - 1.0) < 1e-8

    def test_grouped_components(self) -> None:
        data = _make_seasonal_series(n=200, period=20)
        groups = {"trend": [0], "oscillatory": [1, 2]}
        result = ssa_decompose(data, window=40, groups=groups)
        assert "trend" in result["components"]
        assert "oscillatory" in result["components"]


# ---------------------------------------------------------------------------
# EMD Decomposition
# ---------------------------------------------------------------------------


class TestEMDDecompose:
    def test_imfs_finite(self) -> None:
        """All IMFs must contain finite values."""
        data = _make_seasonal_series(n=300, period=20)
        result = emd_decompose(data, max_imfs=5)
        assert result["n_imfs"] >= 1
        assert np.all(np.isfinite(result["imfs"]))
        assert np.all(np.isfinite(result["residual"]))

    def test_imfs_plus_residual_sum_to_original(self) -> None:
        data = _make_seasonal_series(n=300, period=20)
        result = emd_decompose(data, max_imfs=10)
        reconstructed = result["imfs"].sum(axis=0) + result["residual"]
        np.testing.assert_allclose(
            reconstructed, data.dropna().values, atol=1e-6,
        )

    def test_returns_at_least_one_imf(self) -> None:
        data = _make_seasonal_series(n=200, period=20)
        result = emd_decompose(data, max_imfs=3)
        assert result["imfs"].shape[0] >= 1
        assert result["imfs"].shape[1] == len(data.dropna())
