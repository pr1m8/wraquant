"""Tests for wraquant.ta.volatility module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.volatility import (
    acceleration_bands,
    atr,
    bbwidth,
    chaikin_volatility,
    close_to_close,
    garman_klass,
    historical_volatility,
    kc_width,
    mass_index,
    natr,
    parkinson,
    relative_volatility_index,
    rogers_satchell,
    standard_deviation,
    true_range,
    ulcer_index,
    variance,
    yang_zhang,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv() -> dict[str, pd.Series]:
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, size=n).astype(float)
    return {
        "open": pd.Series(open_, name="open"),
        "high": pd.Series(high, name="high"),
        "low": pd.Series(low, name="low"),
        "close": pd.Series(close, name="close"),
        "volume": pd.Series(volume, name="volume"),
    }


# ---------------------------------------------------------------------------
# True Range
# ---------------------------------------------------------------------------


class TestTrueRange:
    def test_manual_calculation(self) -> None:
        """Verify TR against a hand calculation."""
        high = pd.Series([12.0, 13.0, 14.0])
        low = pd.Series([10.0, 11.0, 12.0])
        close = pd.Series([11.0, 12.0, 13.0])
        result = true_range(high, low, close)
        # Bar 1: max(13-11, |13-11|, |11-11|) = max(2, 2, 0) = 2
        assert abs(result.iloc[1] - 2.0) < 1e-10
        # Bar 2: max(14-12, |14-12|, |12-12|) = max(2, 2, 0) = 2
        assert abs(result.iloc[2] - 2.0) < 1e-10

    def test_always_positive(self, ohlcv: dict[str, pd.Series]) -> None:
        result = true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


class TestATR:
    def test_always_positive(self, ohlcv: dict[str, pd.Series]) -> None:
        """ATR must always be >= 0."""
        result = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_atr_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(result) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# NATR
# ---------------------------------------------------------------------------


class TestNATR:
    def test_is_percentage(self, ohlcv: dict[str, pd.Series]) -> None:
        """NATR should be ATR / close * 100."""
        natr_result = natr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        atr_result = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        expected = (atr_result / ohlcv["close"]) * 100.0
        valid_mask = natr_result.notna() & expected.notna()
        pd.testing.assert_series_equal(
            natr_result[valid_mask].rename(None),
            expected[valid_mask].rename(None),
            atol=1e-10,
        )

    def test_always_positive(self, ohlcv: dict[str, pd.Series]) -> None:
        result = natr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# BBWidth / KC Width
# ---------------------------------------------------------------------------


class TestBBWidth:
    def test_bbwidth_non_negative(self) -> None:
        data = pd.Series(np.random.randn(100) + 100)
        result = bbwidth(data, period=20)
        valid = result.dropna()
        assert (valid >= 0).all()


class TestKCWidth:
    def test_kc_width_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        result = kc_width(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# Others
# ---------------------------------------------------------------------------


class TestChaikinVolatility:
    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = chaikin_volatility(ohlcv["high"], ohlcv["low"])
        assert len(result) == len(ohlcv["close"])


class TestHistoricalVolatility:
    def test_annualized_greater_than_raw(self) -> None:
        np.random.seed(42)
        data = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
        ann = historical_volatility(data, period=21, annualize=True)
        raw = historical_volatility(data, period=21, annualize=False)
        valid_mask = ann.notna() & raw.notna()
        # Annualized should be sqrt(252) times raw
        ratio = ann[valid_mask] / raw[valid_mask]
        expected_ratio = np.sqrt(252)
        assert np.allclose(ratio.values, expected_ratio, rtol=1e-5)


class TestMassIndex:
    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = mass_index(ohlcv["high"], ohlcv["low"])
        assert len(result) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# Garman-Klass Volatility
# ---------------------------------------------------------------------------


class TestGarmanKlass:
    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = garman_klass(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"])
        assert len(result) == len(ohlcv["close"])

    def test_returns_series(self, ohlcv: dict[str, pd.Series]) -> None:
        result = garman_klass(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"])
        assert isinstance(result, pd.Series)

    def test_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        result = garman_klass(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"])
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_annualize_scales(self, ohlcv: dict[str, pd.Series]) -> None:
        ann = garman_klass(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"], annualize=True)
        raw = garman_klass(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"], annualize=False)
        valid = ann.notna() & raw.notna() & (raw > 0)
        ratio = ann[valid] / raw[valid]
        assert np.allclose(ratio.values, np.sqrt(252), rtol=1e-5)


# ---------------------------------------------------------------------------
# Parkinson Volatility
# ---------------------------------------------------------------------------


class TestParkinson:
    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = parkinson(ohlcv["high"], ohlcv["low"])
        assert len(result) == len(ohlcv["close"])

    def test_returns_series(self, ohlcv: dict[str, pd.Series]) -> None:
        result = parkinson(ohlcv["high"], ohlcv["low"])
        assert isinstance(result, pd.Series)

    def test_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        result = parkinson(ohlcv["high"], ohlcv["low"])
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            parkinson(pd.Series([1.0, 2.0]), pd.Series([0.5, 1.5]), period=0)


# ---------------------------------------------------------------------------
# Rogers-Satchell Volatility
# ---------------------------------------------------------------------------


class TestRogersSatchell:
    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = rogers_satchell(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"])
        assert len(result) == len(ohlcv["close"])

    def test_returns_series(self, ohlcv: dict[str, pd.Series]) -> None:
        result = rogers_satchell(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"])
        assert isinstance(result, pd.Series)

    def test_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        result = rogers_satchell(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"])
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# Yang-Zhang Volatility
# ---------------------------------------------------------------------------


class TestYangZhang:
    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = yang_zhang(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"])
        assert len(result) == len(ohlcv["close"])

    def test_returns_series(self, ohlcv: dict[str, pd.Series]) -> None:
        result = yang_zhang(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"])
        assert isinstance(result, pd.Series)

    def test_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        result = yang_zhang(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"])
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_annualize_scales(self, ohlcv: dict[str, pd.Series]) -> None:
        ann = yang_zhang(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"], annualize=True)
        raw = yang_zhang(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["open"], annualize=False)
        valid = ann.notna() & raw.notna() & (raw > 0)
        ratio = ann[valid] / raw[valid]
        assert np.allclose(ratio.values, np.sqrt(252), rtol=1e-5)


# ---------------------------------------------------------------------------
# Close-to-Close Volatility
# ---------------------------------------------------------------------------


class TestCloseToClose:
    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = close_to_close(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_returns_series(self, ohlcv: dict[str, pd.Series]) -> None:
        result = close_to_close(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_matches_historical_volatility(self, ohlcv: dict[str, pd.Series]) -> None:
        """close_to_close should match historical_volatility."""
        cc = close_to_close(ohlcv["close"], period=21, annualize=True)
        hv = historical_volatility(ohlcv["close"], period=21, annualize=True)
        valid = cc.notna() & hv.notna()
        pd.testing.assert_series_equal(
            cc[valid].rename(None),
            hv[valid].rename(None),
            atol=1e-10,
        )

    def test_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        result = close_to_close(ohlcv["close"])
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# Ulcer Index
# ---------------------------------------------------------------------------


class TestUlcerIndex:
    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = ulcer_index(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_returns_series(self, ohlcv: dict[str, pd.Series]) -> None:
        result = ulcer_index(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        result = ulcer_index(ohlcv["close"])
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_zero_for_constant_rising(self) -> None:
        """A strictly rising series should have ulcer index = 0."""
        data = pd.Series(np.arange(1.0, 51.0))
        result = ulcer_index(data, period=5)
        valid = result.dropna()
        assert np.allclose(valid.values, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Relative Volatility Index
# ---------------------------------------------------------------------------


class TestRelativeVolatilityIndex:
    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = relative_volatility_index(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_returns_series(self, ohlcv: dict[str, pd.Series]) -> None:
        result = relative_volatility_index(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_bounded(self, ohlcv: dict[str, pd.Series]) -> None:
        """RVI should oscillate between 0 and 100."""
        result = relative_volatility_index(ohlcv["close"])
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


# ---------------------------------------------------------------------------
# Acceleration Bands
# ---------------------------------------------------------------------------


class TestAccelerationBands:
    def test_returns_dict(self, ohlcv: dict[str, pd.Series]) -> None:
        result = acceleration_bands(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert isinstance(result, dict)
        assert set(result.keys()) == {"upper", "middle", "lower"}

    def test_upper_above_lower(self, ohlcv: dict[str, pd.Series]) -> None:
        result = acceleration_bands(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = result["upper"].notna() & result["lower"].notna()
        assert (result["upper"][valid] >= result["lower"][valid]).all()

    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = acceleration_bands(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(result["upper"]) == len(ohlcv["close"])
        assert len(result["middle"]) == len(ohlcv["close"])
        assert len(result["lower"]) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# Standard Deviation
# ---------------------------------------------------------------------------


class TestStandardDeviation:
    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = standard_deviation(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_returns_series(self, ohlcv: dict[str, pd.Series]) -> None:
        result = standard_deviation(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        result = standard_deviation(ohlcv["close"])
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_constant_series_zero(self) -> None:
        """Std dev of a constant series should be zero."""
        data = pd.Series([5.0] * 30)
        result = standard_deviation(data, period=10)
        valid = result.dropna()
        assert np.allclose(valid.values, 0.0, atol=1e-10)

    def test_accepts_list_input(self) -> None:
        result = standard_deviation(list(range(1, 30)))
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------


class TestVariance:
    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = variance(ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_returns_series(self, ohlcv: dict[str, pd.Series]) -> None:
        result = variance(ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        result = variance(ohlcv["close"])
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_variance_is_std_squared(self, ohlcv: dict[str, pd.Series]) -> None:
        """Variance should equal standard deviation squared."""
        sd = standard_deviation(ohlcv["close"], period=20)
        var = variance(ohlcv["close"], period=20)
        valid = sd.notna() & var.notna()
        pd.testing.assert_series_equal(
            var[valid].rename(None),
            (sd[valid] ** 2).rename(None),
            atol=1e-10,
        )

    def test_constant_series_zero(self) -> None:
        """Variance of a constant series should be zero."""
        data = pd.Series([5.0] * 30)
        result = variance(data, period=10)
        valid = result.dropna()
        assert np.allclose(valid.values, 0.0, atol=1e-10)
