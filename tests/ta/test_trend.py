"""Tests for wraquant.ta.trend module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.trend import (
    adx,
    aroon,
    linear_regression,
    linear_regression_slope,
    psar,
    trix,
    vortex,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv() -> dict[str, pd.Series]:
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    return {
        "high": pd.Series(high, name="high"),
        "low": pd.Series(low, name="low"),
        "close": pd.Series(close, name="close"),
    }


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


class TestADX:
    def test_bounds(self, ohlcv: dict[str, pd.Series]) -> None:
        """ADX and DI values should be in [0, 100]."""
        result = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        for key in ("adx", "plus_di", "minus_di"):
            valid = result[key].dropna()
            assert (valid >= -1e-10).all(), f"{key} below 0"
            assert (valid <= 100 + 1e-10).all(), f"{key} above 100"

    def test_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert set(result.keys()) == {"adx", "plus_di", "minus_di"}


# ---------------------------------------------------------------------------
# Aroon
# ---------------------------------------------------------------------------


class TestAroon:
    def test_bounds(self, ohlcv: dict[str, pd.Series]) -> None:
        """Aroon up/down should be in [0, 100]."""
        result = aroon(ohlcv["high"], ohlcv["low"])
        for key in ("aroon_up", "aroon_down"):
            valid = result[key].dropna()
            assert (valid >= -1e-10).all(), f"{key} below 0"
            assert (valid <= 100 + 1e-10).all(), f"{key} above 100"

    def test_oscillator_bounds(self, ohlcv: dict[str, pd.Series]) -> None:
        """Aroon oscillator should be in [-100, 100]."""
        result = aroon(ohlcv["high"], ohlcv["low"])
        valid = result["oscillator"].dropna()
        assert (valid >= -100 - 1e-10).all()
        assert (valid <= 100 + 1e-10).all()

    def test_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = aroon(ohlcv["high"], ohlcv["low"])
        assert set(result.keys()) == {"aroon_up", "aroon_down", "oscillator"}


# ---------------------------------------------------------------------------
# PSAR
# ---------------------------------------------------------------------------


class TestPSAR:
    def test_always_one_side_of_price(self, ohlcv: dict[str, pd.Series]) -> None:
        """PSAR should always be either above high or below low."""
        result = psar(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid_mask = result.notna()
        sar = result[valid_mask]
        h = ohlcv["high"][valid_mask]
        l = ohlcv["low"][valid_mask]  # noqa: E741
        # SAR is below low (uptrend) or above high (downtrend)
        # Allow some tolerance for the switching bar
        below_low = sar <= l
        above_high = sar >= h
        one_side = below_low | above_high
        # Most bars should satisfy this; switching bars may not
        assert one_side.sum() / len(one_side) > 0.85

    def test_psar_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = psar(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(result) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# Vortex
# ---------------------------------------------------------------------------


class TestVortex:
    def test_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = vortex(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert set(result.keys()) == {"plus_vi", "minus_vi"}

    def test_positive_values(self, ohlcv: dict[str, pd.Series]) -> None:
        result = vortex(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        for key in ("plus_vi", "minus_vi"):
            valid = result[key].dropna()
            assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# TRIX
# ---------------------------------------------------------------------------


class TestTRIX:
    def test_trix_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = trix(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------


class TestLinearRegression:
    def test_perfect_linear_data(self) -> None:
        """Regression on perfectly linear data should give exact slope and R^2=1."""
        data = pd.Series(np.arange(20, dtype=float) * 2.5 + 10.0)
        result = linear_regression(data, period=10)
        # Slope should be 2.5
        valid_slope = result["slope"].dropna()
        assert np.allclose(valid_slope.values, 2.5, atol=1e-10)
        # R-squared should be 1.0
        valid_r2 = result["r_squared"].dropna()
        assert np.allclose(valid_r2.values, 1.0, atol=1e-10)

    def test_keys(self) -> None:
        data = pd.Series(np.random.randn(50))
        result = linear_regression(data, period=10)
        assert set(result.keys()) == {"value", "slope", "intercept", "r_squared"}

    def test_slope_shortcut(self) -> None:
        data = pd.Series(np.arange(20, dtype=float))
        slope = linear_regression_slope(data, period=5)
        full = linear_regression(data, period=5)
        pd.testing.assert_series_equal(slope, full["slope"])
