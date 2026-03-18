"""Tests for wraquant.ta.volatility module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.volatility import (
    atr,
    bbwidth,
    chaikin_volatility,
    historical_volatility,
    kc_width,
    mass_index,
    natr,
    true_range,
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
    volume = np.random.randint(1000, 10000, size=n).astype(float)
    return {
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
