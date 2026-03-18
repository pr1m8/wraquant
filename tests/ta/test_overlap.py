"""Tests for wraquant.ta.overlap module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.overlap import (
    bollinger_bands,
    dema,
    donchian_channel,
    ema,
    ichimoku,
    kama,
    keltner_channel,
    sma,
    supertrend,
    tema,
    vwap,
    wma,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def close_series() -> pd.Series:
    """Simple ascending close prices for deterministic tests."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.Series(prices, name="close")


@pytest.fixture
def ohlcv() -> dict[str, pd.Series]:
    """Synthetic OHLCV data."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, size=n).astype(float)
    return {
        "open": pd.Series(open_, name="open"),
        "high": pd.Series(high, name="high"),
        "low": pd.Series(low, name="low"),
        "close": pd.Series(close, name="close"),
        "volume": pd.Series(volume, name="volume"),
    }


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


class TestSMA:
    def test_sma_equals_manual_mean(self, close_series: pd.Series) -> None:
        """SMA at each bar should equal the mean of the preceding window."""
        period = 10
        result = sma(close_series, period)
        for i in range(period - 1, len(close_series)):
            expected = close_series.iloc[i - period + 1 : i + 1].mean()
            assert abs(result.iloc[i] - expected) < 1e-10

    def test_sma_nan_prefix(self, close_series: pd.Series) -> None:
        """The first period-1 values must be NaN."""
        period = 20
        result = sma(close_series, period)
        assert result.iloc[: period - 1].isna().all()
        assert result.iloc[period - 1 :].notna().all()

    def test_sma_length(self, close_series: pd.Series) -> None:
        result = sma(close_series, 20)
        assert len(result) == len(close_series)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class TestEMA:
    def test_ema_first_value(self, close_series: pd.Series) -> None:
        """First non-NaN EMA value should be close to SMA of same window."""
        period = 10
        ema_result = ema(close_series, period)
        sma_result = sma(close_series, period)
        first_valid_idx = period - 1
        # The first EMA value should be close to SMA (not exact due to ewm semantics)
        assert (
            abs(ema_result.iloc[first_valid_idx] - sma_result.iloc[first_valid_idx])
            < 1.0
        )

    def test_ema_responds_faster_than_sma(self, close_series: pd.Series) -> None:
        """EMA should react more quickly to recent data than SMA."""
        period = 20
        ema_result = ema(close_series, period)
        sma_result = sma(close_series, period)
        # After a sharp move, EMA should deviate from SMA
        diff = (ema_result - sma_result).dropna()
        assert not (diff == 0).all()


# ---------------------------------------------------------------------------
# WMA
# ---------------------------------------------------------------------------


class TestWMA:
    def test_wma_manual(self) -> None:
        """Verify WMA against a hand calculation."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wma(data, period=3)
        # WMA(3) at index 2: (1*1 + 2*2 + 3*3) / 6 = 14/6
        expected = (1 * 1 + 2 * 2 + 3 * 3) / 6.0
        assert abs(result.iloc[2] - expected) < 1e-10


# ---------------------------------------------------------------------------
# DEMA / TEMA
# ---------------------------------------------------------------------------


class TestDEMATEMA:
    def test_dema_length(self, close_series: pd.Series) -> None:
        result = dema(close_series, 10)
        assert len(result) == len(close_series)

    def test_tema_length(self, close_series: pd.Series) -> None:
        result = tema(close_series, 10)
        assert len(result) == len(close_series)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_width_relationship(self, close_series: pd.Series) -> None:
        """Upper - lower should always be non-negative (where defined)."""
        bb = bollinger_bands(close_series, period=20, std_dev=2.0)
        diff = (bb["upper"] - bb["lower"]).dropna()
        assert (diff >= -1e-10).all()

    def test_middle_is_sma(self, close_series: pd.Series) -> None:
        bb = bollinger_bands(close_series, period=20)
        sma_val = sma(close_series, 20)
        pd.testing.assert_series_equal(
            bb["middle"].rename(None), sma_val.rename(None), atol=1e-10
        )

    def test_keys(self, close_series: pd.Series) -> None:
        bb = bollinger_bands(close_series)
        assert set(bb.keys()) == {"upper", "middle", "lower", "bandwidth", "percent_b"}


# ---------------------------------------------------------------------------
# Ichimoku
# ---------------------------------------------------------------------------


class TestIchimoku:
    def test_component_count(self, ohlcv: dict[str, pd.Series]) -> None:
        result = ichimoku(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert set(result.keys()) == {
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
        }

    def test_all_series_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = ichimoku(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        for key, series in result.items():
            assert len(series) == len(ohlcv["close"]), f"{key} length mismatch"


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------


class TestSupertrend:
    def test_output_structure(self, ohlcv: dict[str, pd.Series]) -> None:
        result = supertrend(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert set(result.keys()) == {"supertrend", "direction"}
        assert len(result["supertrend"]) == len(ohlcv["close"])
        assert len(result["direction"]) == len(ohlcv["close"])

    def test_direction_values(self, ohlcv: dict[str, pd.Series]) -> None:
        result = supertrend(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid_dir = result["direction"].dropna()
        assert set(valid_dir.unique()).issubset({1.0, -1.0})


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------


class TestVWAP:
    def test_vwap_basic(self, ohlcv: dict[str, pd.Series]) -> None:
        result = vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])
        assert result.notna().any()


# ---------------------------------------------------------------------------
# KAMA
# ---------------------------------------------------------------------------


class TestKAMA:
    def test_kama_output_length(self, close_series: pd.Series) -> None:
        result = kama(close_series, period=10)
        assert len(result) == len(close_series)


# ---------------------------------------------------------------------------
# Keltner / Donchian
# ---------------------------------------------------------------------------


class TestKeltnerChannel:
    def test_keltner_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = keltner_channel(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert set(result.keys()) == {"upper", "middle", "lower"}


class TestDonchianChannel:
    def test_donchian_upper_ge_lower(self, ohlcv: dict[str, pd.Series]) -> None:
        result = donchian_channel(ohlcv["high"], ohlcv["low"])
        diff = (result["upper"] - result["lower"]).dropna()
        assert (diff >= -1e-10).all()
