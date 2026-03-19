"""Tests for wraquant.ta.trend module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.trend import (
    adx,
    aroon,
    fractal_adaptive_ma,
    guppy_mma,
    heikin_ashi,
    hull_ma,
    linear_regression,
    linear_regression_slope,
    mcginley_dynamic,
    psar,
    rainbow_ma,
    schaff_trend_cycle,
    tilson_t3,
    trix,
    vidya,
    vortex,
    zero_lag_ema,
    zigzag,
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
    open_ = close + np.random.randn(n) * 0.2
    return {
        "open": pd.Series(open_, name="open"),
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


# ---------------------------------------------------------------------------
# ZigZag
# ---------------------------------------------------------------------------


class TestZigZag:
    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = zigzag(ohlcv["close"], pct_change=2.0)
        assert len(result) == len(ohlcv["close"])

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = zigzag(ohlcv["close"], pct_change=2.0)
        assert isinstance(result, pd.Series)

    def test_known_pivots(self) -> None:
        """Given a clear up-down-up pattern, zigzag should interpolate."""
        data = pd.Series([100, 110, 120, 110, 100, 110, 120])
        result = zigzag(data, pct_change=5.0)
        # First pivot is at index 0, should not all be NaN
        assert result.notna().any()

    def test_short_series(self) -> None:
        result = zigzag(pd.Series([100.0]), pct_change=5.0)
        assert len(result) == 1

    def test_invalid_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="pct_change"):
            zigzag(pd.Series([100, 110, 120]), pct_change=-1.0)

    def test_accepts_list_input(self) -> None:
        result = zigzag(list(range(100, 200)))
        assert isinstance(result, (pd.Series, pd.DataFrame, dict))


# ---------------------------------------------------------------------------
# Heikin-Ashi
# ---------------------------------------------------------------------------


class TestHeikinAshi:
    def test_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = heikin_ashi(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert set(result.keys()) == {"ha_open", "ha_high", "ha_low", "ha_close"}

    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = heikin_ashi(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        for key in result:
            assert len(result[key]) == len(ohlcv["close"])

    def test_output_types(self, ohlcv: dict[str, pd.Series]) -> None:
        result = heikin_ashi(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        for key in result:
            assert isinstance(result[key], pd.Series)

    def test_ha_high_ge_ha_low(self, ohlcv: dict[str, pd.Series]) -> None:
        """HA high should always be >= HA low."""
        result = heikin_ashi(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert (result["ha_high"] >= result["ha_low"] - 1e-10).all()

    def test_ha_close_is_ohlc_avg(self, ohlcv: dict[str, pd.Series]) -> None:
        """HA close = (O + H + L + C) / 4."""
        result = heikin_ashi(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        expected = (
            ohlcv["open"] + ohlcv["high"] + ohlcv["low"] + ohlcv["close"]
        ) / 4.0
        pd.testing.assert_series_equal(
            result["ha_close"], expected, check_names=False
        )


# ---------------------------------------------------------------------------
# McGinley Dynamic
# ---------------------------------------------------------------------------


class TestMcGinleyDynamic:
    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = mcginley_dynamic(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = mcginley_dynamic(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_tracks_price(self, ohlcv: dict[str, pd.Series]) -> None:
        """McGinley Dynamic should roughly track the close price."""
        result = mcginley_dynamic(ohlcv["close"], period=14)
        valid = result.dropna()
        close_valid = ohlcv["close"].loc[valid.index]
        # Should be correlated with price
        assert valid.corr(close_valid) > 0.8

    def test_constant_price(self) -> None:
        """On constant data, MD should converge to that constant."""
        data = pd.Series([50.0] * 50)
        result = mcginley_dynamic(data, period=10)
        assert np.allclose(result.dropna().values, 50.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Schaff Trend Cycle
# ---------------------------------------------------------------------------


class TestSchaffTrendCycle:
    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = schaff_trend_cycle(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = schaff_trend_cycle(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_bounded(self, ohlcv: dict[str, pd.Series]) -> None:
        """STC should be in [0, 100]."""
        result = schaff_trend_cycle(ohlcv["close"])
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 100 + 1e-10).all()


# ---------------------------------------------------------------------------
# Guppy Multiple Moving Average
# ---------------------------------------------------------------------------


class TestGuppyMMA:
    def test_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = guppy_mma(ohlcv["close"])
        expected_keys = {
            "short_3", "short_5", "short_8", "short_10", "short_12", "short_15",
            "long_30", "long_35", "long_40", "long_45", "long_50", "long_60",
        }
        assert set(result.keys()) == expected_keys

    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = guppy_mma(ohlcv["close"])
        for key in result:
            assert len(result[key]) == len(ohlcv["close"])

    def test_output_types(self, ohlcv: dict[str, pd.Series]) -> None:
        result = guppy_mma(ohlcv["close"])
        for key in result:
            assert isinstance(result[key], pd.Series)


# ---------------------------------------------------------------------------
# Rainbow MA
# ---------------------------------------------------------------------------


class TestRainbowMA:
    def test_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = rainbow_ma(ohlcv["close"], period=10, levels=10)
        expected_keys = {f"sma_{i}" for i in range(1, 11)}
        assert set(result.keys()) == expected_keys

    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = rainbow_ma(ohlcv["close"])
        for key in result:
            assert len(result[key]) == len(ohlcv["close"])

    def test_output_types(self, ohlcv: dict[str, pd.Series]) -> None:
        result = rainbow_ma(ohlcv["close"])
        for key in result:
            assert isinstance(result[key], pd.Series)

    def test_smoothing_order(self, ohlcv: dict[str, pd.Series]) -> None:
        """Higher levels should be smoother (lower std of diff)."""
        result = rainbow_ma(ohlcv["close"], period=10, levels=5)
        std1 = result["sma_1"].diff().dropna().std()
        std5 = result["sma_5"].diff().dropna().std()
        assert std5 < std1

    def test_invalid_levels(self) -> None:
        with pytest.raises(ValueError, match="levels"):
            rainbow_ma(pd.Series([1, 2, 3]), period=2, levels=0)


# ---------------------------------------------------------------------------
# Hull MA
# ---------------------------------------------------------------------------


class TestHullMA:
    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = hull_ma(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = hull_ma(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_tracks_price(self, ohlcv: dict[str, pd.Series]) -> None:
        result = hull_ma(ohlcv["close"], period=16)
        valid = result.dropna()
        close_valid = ohlcv["close"].loc[valid.index]
        assert valid.corr(close_valid) > 0.9

    def test_linear_data(self) -> None:
        """On perfectly linear data, HMA should closely track."""
        data = pd.Series(np.arange(50, dtype=float))
        result = hull_ma(data, period=9)
        valid = result.dropna()
        expected = data.loc[valid.index]
        # Should be very close to the linear data
        assert np.allclose(valid.values, expected.values, atol=5.0)


# ---------------------------------------------------------------------------
# Zero-Lag EMA
# ---------------------------------------------------------------------------


class TestZeroLagEMA:
    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = zero_lag_ema(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = zero_lag_ema(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_constant_price(self) -> None:
        """On constant data, ZLEMA should equal the constant."""
        data = pd.Series([42.0] * 50)
        result = zero_lag_ema(data, period=10)
        valid = result.dropna()
        assert np.allclose(valid.values, 42.0, atol=1e-10)


# ---------------------------------------------------------------------------
# VIDYA
# ---------------------------------------------------------------------------


class TestVIDYA:
    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = vidya(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = vidya(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_tracks_price(self, ohlcv: dict[str, pd.Series]) -> None:
        result = vidya(ohlcv["close"], period=14)
        valid = result.dropna()
        close_valid = ohlcv["close"].loc[valid.index]
        assert valid.corr(close_valid) > 0.8

    def test_short_series(self) -> None:
        """Should handle series shorter than period gracefully."""
        data = pd.Series([1.0, 2.0, 3.0])
        result = vidya(data, period=10)
        # All NaN since period > len(data)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# Tilson T3
# ---------------------------------------------------------------------------


class TestTilsonT3:
    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = tilson_t3(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = tilson_t3(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_tracks_price(self, ohlcv: dict[str, pd.Series]) -> None:
        result = tilson_t3(ohlcv["close"], period=5)
        valid = result.dropna()
        close_valid = ohlcv["close"].loc[valid.index]
        assert valid.corr(close_valid) > 0.8

    def test_constant_price(self) -> None:
        """On constant data, T3 should converge to that constant."""
        data = pd.Series([30.0] * 100)
        result = tilson_t3(data, period=5)
        valid = result.dropna()
        assert np.allclose(valid.values, 30.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Fractal Adaptive MA
# ---------------------------------------------------------------------------


class TestFractalAdaptiveMA:
    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = fractal_adaptive_ma(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = fractal_adaptive_ma(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_tracks_price(self, ohlcv: dict[str, pd.Series]) -> None:
        result = fractal_adaptive_ma(ohlcv["close"], period=16)
        valid = result.dropna()
        close_valid = ohlcv["close"].loc[valid.index]
        assert valid.corr(close_valid) > 0.8

    def test_short_series(self) -> None:
        """Should return all NaN for series shorter than period."""
        data = pd.Series([1.0, 2.0, 3.0])
        result = fractal_adaptive_ma(data, period=16)
        assert result.isna().all()

    def test_odd_period_rounds_up(self) -> None:
        """Odd period should be made even (period+1) internally."""
        data = pd.Series(np.random.randn(100).cumsum() + 100)
        result = fractal_adaptive_ma(data, period=15)
        # Should work without error and produce values
        assert result.notna().any()
