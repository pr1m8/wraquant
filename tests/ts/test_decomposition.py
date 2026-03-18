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
