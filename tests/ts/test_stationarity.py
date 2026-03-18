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
