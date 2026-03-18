"""Tests for wraquant.ta.momentum module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.momentum import (
    awesome_oscillator,
    cci,
    cmo,
    dpo,
    macd,
    momentum,
    ppo,
    roc,
    rsi,
    stochastic,
    stochastic_rsi,
    tsi,
    ultimate_oscillator,
    williams_r,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def close_series() -> pd.Series:
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    return pd.Series(prices, name="close")


@pytest.fixture
def ohlcv() -> dict[str, pd.Series]:
    np.random.seed(42)
    n = 200
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
# RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_bounds(self, close_series: pd.Series) -> None:
        """RSI must be in [0, 100]."""
        result = rsi(close_series, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_nan_prefix(self, close_series: pd.Series) -> None:
        result = rsi(close_series, period=14)
        # At least the first 13 values should be NaN (need period values)
        assert result.iloc[:13].isna().all()

    def test_rsi_length(self, close_series: pd.Series) -> None:
        result = rsi(close_series, 14)
        assert len(result) == len(close_series)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


class TestMACD:
    def test_signal_is_ema_of_macd(self, close_series: pd.Series) -> None:
        """Signal line should be the EMA of the MACD line."""
        result = macd(close_series)
        macd_line = result["macd"]
        signal_line = result["signal"]
        # Compute EMA of MACD manually
        expected_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        # Compare where both are valid
        mask = signal_line.notna() & expected_signal.notna()
        pd.testing.assert_series_equal(
            signal_line[mask].rename(None),
            expected_signal[mask].rename(None),
            atol=1e-10,
        )

    def test_histogram_equals_macd_minus_signal(self, close_series: pd.Series) -> None:
        result = macd(close_series)
        hist_expected = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(
            result["histogram"].rename(None),
            hist_expected.rename(None),
            atol=1e-10,
        )

    def test_keys(self, close_series: pd.Series) -> None:
        result = macd(close_series)
        assert set(result.keys()) == {"macd", "signal", "histogram"}


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------


class TestStochastic:
    def test_bounds(self, ohlcv: dict[str, pd.Series]) -> None:
        """Both %K and %D must be in [0, 100]."""
        result = stochastic(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        for key in ("k", "d"):
            valid = result[key].dropna()
            assert (valid >= -1e-10).all(), f"{key} below 0"
            assert (valid <= 100 + 1e-10).all(), f"{key} above 100"

    def test_k_100_at_period_high(self) -> None:
        """If close == highest high in window, %K should be 100."""
        high = pd.Series([10, 20, 30, 40, 50.0])
        low = pd.Series([5, 15, 25, 35, 45.0])
        close = pd.Series([10, 20, 30, 40, 50.0])
        result = stochastic(high, low, close, k_period=3)
        # At index 4: close=50, highest=50, lowest=35 → %K = 100
        assert abs(result["k"].iloc[4] - 100.0) < 1e-10


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------


class TestWilliamsR:
    def test_bounds(self, ohlcv: dict[str, pd.Series]) -> None:
        """%R must be in [-100, 0]."""
        result = williams_r(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = result.dropna()
        assert (valid >= -100 - 1e-10).all()
        assert (valid <= 0 + 1e-10).all()


# ---------------------------------------------------------------------------
# CCI
# ---------------------------------------------------------------------------


class TestCCI:
    def test_cci_calculation(self, ohlcv: dict[str, pd.Series]) -> None:
        """CCI should return values and have correct length."""
        result = cci(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=20)
        assert len(result) == len(ohlcv["close"])
        # CCI is unbounded but should have valid values after warm-up
        assert result.dropna().shape[0] > 0

    def test_cci_nan_prefix(self, ohlcv: dict[str, pd.Series]) -> None:
        period = 20
        result = cci(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=period)
        assert result.iloc[: period - 1].isna().all()


# ---------------------------------------------------------------------------
# ROC / Momentum
# ---------------------------------------------------------------------------


class TestROCMomentum:
    def test_roc_manual(self) -> None:
        data = pd.Series([100, 110, 121, 133.1, 146.41])
        result = roc(data, period=1)
        # ROC at index 1: (110 - 100)/100 * 100 = 10
        assert abs(result.iloc[1] - 10.0) < 1e-10

    def test_momentum_manual(self) -> None:
        data = pd.Series([100, 110, 120, 130, 140.0])
        result = momentum(data, period=2)
        # momentum at index 2: 120 - 100 = 20
        assert abs(result.iloc[2] - 20.0) < 1e-10


# ---------------------------------------------------------------------------
# TSI / PPO / CMO / DPO
# ---------------------------------------------------------------------------


class TestOtherMomentum:
    def test_tsi_keys(self, close_series: pd.Series) -> None:
        result = tsi(close_series)
        assert set(result.keys()) == {"tsi", "signal"}

    def test_ppo_keys(self, close_series: pd.Series) -> None:
        result = ppo(close_series)
        assert set(result.keys()) == {"ppo", "signal", "histogram"}

    def test_cmo_bounds(self, close_series: pd.Series) -> None:
        result = cmo(close_series, period=14)
        valid = result.dropna()
        assert (valid >= -100 - 1e-10).all()
        assert (valid <= 100 + 1e-10).all()

    def test_dpo_length(self, close_series: pd.Series) -> None:
        result = dpo(close_series, period=20)
        assert len(result) == len(close_series)

    def test_ao_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = awesome_oscillator(ohlcv["high"], ohlcv["low"])
        assert len(result) == len(ohlcv["close"])

    def test_ultimate_oscillator_bounds(self, ohlcv: dict[str, pd.Series]) -> None:
        result = ultimate_oscillator(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 100 + 1e-10).all()

    def test_stochastic_rsi_keys(self, close_series: pd.Series) -> None:
        result = stochastic_rsi(close_series)
        assert set(result.keys()) == {"k", "d"}
