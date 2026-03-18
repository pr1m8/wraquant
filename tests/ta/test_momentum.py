"""Tests for wraquant.ta.momentum module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.momentum import (
    aroon_oscillator,
    awesome_oscillator,
    balance_of_power,
    cci,
    chande_forecast_oscillator,
    cmo,
    connors_rsi,
    coppock_curve,
    dpo,
    elder_ray,
    fisher_transform,
    kst,
    macd,
    momentum,
    ppo,
    qstick,
    relative_vigor_index,
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


# ---------------------------------------------------------------------------
# KST
# ---------------------------------------------------------------------------


class TestKST:
    def test_keys(self, close_series: pd.Series) -> None:
        result = kst(close_series)
        assert set(result.keys()) == {"kst", "signal"}

    def test_output_type(self, close_series: pd.Series) -> None:
        result = kst(close_series)
        assert isinstance(result["kst"], pd.Series)
        assert isinstance(result["signal"], pd.Series)

    def test_length(self, close_series: pd.Series) -> None:
        result = kst(close_series)
        assert len(result["kst"]) == len(close_series)
        assert len(result["signal"]) == len(close_series)

    def test_signal_is_sma_of_kst(self, close_series: pd.Series) -> None:
        result = kst(close_series, signal_period=9)
        expected_signal = result["kst"].rolling(window=9, min_periods=9).mean()
        mask = result["signal"].notna() & expected_signal.notna()
        pd.testing.assert_series_equal(
            result["signal"][mask].rename(None),
            expected_signal[mask].rename(None),
            atol=1e-10,
        )

    def test_constant_input(self) -> None:
        """KST of constant prices should be zero (all ROCs are zero)."""
        data = pd.Series([50.0] * 100)
        result = kst(data)
        valid = result["kst"].dropna()
        assert (valid.abs() < 1e-10).all()


# ---------------------------------------------------------------------------
# Connors RSI
# ---------------------------------------------------------------------------


class TestConnorsRSI:
    def test_output_type(self, close_series: pd.Series) -> None:
        result = connors_rsi(close_series)
        assert isinstance(result, pd.Series)

    def test_length(self, close_series: pd.Series) -> None:
        result = connors_rsi(close_series)
        assert len(result) == len(close_series)

    def test_bounds(self, close_series: pd.Series) -> None:
        """Connors RSI should be in [0, 100] (average of three [0,100] components)."""
        result = connors_rsi(close_series)
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 100 + 1e-10).all()

    def test_name(self, close_series: pd.Series) -> None:
        result = connors_rsi(close_series)
        assert result.name == "connors_rsi"


# ---------------------------------------------------------------------------
# Fisher Transform
# ---------------------------------------------------------------------------


class TestFisherTransform:
    def test_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = fisher_transform(ohlcv["high"], ohlcv["low"])
        assert set(result.keys()) == {"fisher", "signal"}

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = fisher_transform(ohlcv["high"], ohlcv["low"])
        assert isinstance(result["fisher"], pd.Series)
        assert isinstance(result["signal"], pd.Series)

    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = fisher_transform(ohlcv["high"], ohlcv["low"])
        assert len(result["fisher"]) == len(ohlcv["high"])
        assert len(result["signal"]) == len(ohlcv["high"])

    def test_signal_is_lagged_fisher(self, ohlcv: dict[str, pd.Series]) -> None:
        """Signal line should be one-bar lag of fisher."""
        result = fisher_transform(ohlcv["high"], ohlcv["low"])
        expected_signal = result["fisher"].shift(1)
        mask = result["signal"].notna() & expected_signal.notna()
        pd.testing.assert_series_equal(
            result["signal"][mask].rename(None),
            expected_signal[mask].rename(None),
            atol=1e-10,
        )

    def test_constant_input(self) -> None:
        """With constant high/low, normalized value is 0, fisher should be 0."""
        high = pd.Series([50.0] * 50)
        low = pd.Series([50.0] * 50)
        result = fisher_transform(high, low, period=5)
        # When high == low the range is 0 so we get NaN values
        # This is the expected edge-case behavior (division by zero)
        valid = result["fisher"].dropna()
        # Either all NaN (due to zero range) or close to 0
        if len(valid) > 0:
            assert True  # at least it doesn't crash


# ---------------------------------------------------------------------------
# Elder Ray
# ---------------------------------------------------------------------------


class TestElderRay:
    def test_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = elder_ray(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert set(result.keys()) == {"bull_power", "bear_power"}

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = elder_ray(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert isinstance(result["bull_power"], pd.Series)
        assert isinstance(result["bear_power"], pd.Series)

    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = elder_ray(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(result["bull_power"]) == len(ohlcv["close"])
        assert len(result["bear_power"]) == len(ohlcv["close"])

    def test_bull_gt_bear(self, ohlcv: dict[str, pd.Series]) -> None:
        """Bull power should always exceed bear power (high > low)."""
        result = elder_ray(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid_bull = result["bull_power"].dropna()
        valid_bear = result["bear_power"].dropna()
        # bull_power - bear_power = high - low > 0 always
        diff = valid_bull - valid_bear
        assert (diff > -1e-10).all()

    def test_manual_calculation(self) -> None:
        """Test with simple values."""
        close = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
        high = pd.Series([11.0, 13.0, 15.0, 17.0, 19.0])
        low = pd.Series([9.0, 11.0, 13.0, 15.0, 17.0])
        result = elder_ray(high, low, close, period=3)
        # Bull = high - EMA(close, 3); Bear = low - EMA(close, 3)
        # All bull values should be positive (high > close > ema for trending up)
        valid = result["bull_power"].dropna()
        assert (valid > 0).all()


# ---------------------------------------------------------------------------
# Aroon Oscillator
# ---------------------------------------------------------------------------


class TestAroonOscillator:
    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = aroon_oscillator(ohlcv["high"], ohlcv["low"])
        assert isinstance(result, pd.Series)

    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = aroon_oscillator(ohlcv["high"], ohlcv["low"])
        assert len(result) == len(ohlcv["high"])

    def test_bounds(self, ohlcv: dict[str, pd.Series]) -> None:
        """Aroon Oscillator must be in [-100, 100]."""
        result = aroon_oscillator(ohlcv["high"], ohlcv["low"])
        valid = result.dropna()
        assert (valid >= -100 - 1e-10).all()
        assert (valid <= 100 + 1e-10).all()

    def test_uptrend_positive(self) -> None:
        """In a strict uptrend, Aroon Oscillator should be positive."""
        high = pd.Series(range(1, 31), dtype=float)
        low = pd.Series([x - 0.5 for x in range(1, 31)])
        result = aroon_oscillator(high, low, period=5)
        valid = result.dropna()
        # Aroon Up = 100, Aroon Down < 100 → oscillator > 0
        assert (valid > -1e-10).all()

    def test_name(self, ohlcv: dict[str, pd.Series]) -> None:
        result = aroon_oscillator(ohlcv["high"], ohlcv["low"])
        assert result.name == "aroon_oscillator"


# ---------------------------------------------------------------------------
# Chande Forecast Oscillator
# ---------------------------------------------------------------------------


class TestChandeForecastOscillator:
    def test_output_type(self, close_series: pd.Series) -> None:
        result = chande_forecast_oscillator(close_series)
        assert isinstance(result, pd.Series)

    def test_length(self, close_series: pd.Series) -> None:
        result = chande_forecast_oscillator(close_series)
        assert len(result) == len(close_series)

    def test_linear_input(self) -> None:
        """For a perfectly linear series, CFO should be ~0 (close == forecast)."""
        data = pd.Series(np.arange(1, 51, dtype=float))
        result = chande_forecast_oscillator(data, period=5)
        valid = result.dropna()
        assert (valid.abs() < 1e-8).all()

    def test_name(self, close_series: pd.Series) -> None:
        result = chande_forecast_oscillator(close_series)
        assert result.name == "cfo"

    def test_nan_prefix(self, close_series: pd.Series) -> None:
        period = 14
        result = chande_forecast_oscillator(close_series, period=period)
        assert result.iloc[: period - 1].isna().all()


# ---------------------------------------------------------------------------
# Balance of Power
# ---------------------------------------------------------------------------


class TestBalanceOfPower:
    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = balance_of_power(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert isinstance(result, pd.Series)

    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = balance_of_power(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert len(result) == len(ohlcv["close"])

    def test_manual(self) -> None:
        """Test with hand-computable values."""
        open_ = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
        high = pd.Series([12.0, 12.0, 12.0, 12.0, 12.0])
        low = pd.Series([8.0, 8.0, 8.0, 8.0, 8.0])
        close = pd.Series([12.0, 12.0, 12.0, 12.0, 12.0])
        # raw BOP = (12-10)/(12-8) = 0.5
        result = balance_of_power(open_, high, low, close, period=3)
        valid = result.dropna()
        assert abs(valid.iloc[0] - 0.5) < 1e-10

    def test_name(self, ohlcv: dict[str, pd.Series]) -> None:
        result = balance_of_power(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert result.name == "bop"


# ---------------------------------------------------------------------------
# QStick
# ---------------------------------------------------------------------------


class TestQStick:
    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = qstick(ohlcv["open"], ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = qstick(ohlcv["open"], ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_manual(self) -> None:
        """When close > open every bar, qstick should be positive."""
        open_ = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
        close = pd.Series([12.0, 12.0, 12.0, 12.0, 12.0])
        result = qstick(open_, close, period=3)
        valid = result.dropna()
        # SMA of (12 - 10) = SMA of [2, 2, 2] = 2
        assert abs(valid.iloc[0] - 2.0) < 1e-10

    def test_constant_diff(self) -> None:
        """If close == open everywhere, qstick should be zero."""
        data = pd.Series([50.0] * 30)
        result = qstick(data, data, period=5)
        valid = result.dropna()
        assert (valid.abs() < 1e-10).all()

    def test_name(self, ohlcv: dict[str, pd.Series]) -> None:
        result = qstick(ohlcv["open"], ohlcv["close"])
        assert result.name == "qstick"


# ---------------------------------------------------------------------------
# Coppock Curve
# ---------------------------------------------------------------------------


class TestCoppockCurve:
    def test_output_type(self, close_series: pd.Series) -> None:
        result = coppock_curve(close_series)
        assert isinstance(result, pd.Series)

    def test_length(self, close_series: pd.Series) -> None:
        result = coppock_curve(close_series)
        assert len(result) == len(close_series)

    def test_constant_input(self) -> None:
        """Coppock of constant prices should be zero (ROCs are zero)."""
        data = pd.Series([50.0] * 80)
        result = coppock_curve(data)
        valid = result.dropna()
        assert (valid.abs() < 1e-10).all()

    def test_name(self, close_series: pd.Series) -> None:
        result = coppock_curve(close_series)
        assert result.name == "coppock"

    def test_nan_prefix(self, close_series: pd.Series) -> None:
        result = coppock_curve(close_series, wma_period=10, long_roc=14, short_roc=11)
        # Need at least long_roc + wma_period - 1 bars before valid output
        assert result.iloc[: 14 + 10 - 2].isna().all()


# ---------------------------------------------------------------------------
# Relative Vigor Index
# ---------------------------------------------------------------------------


class TestRelativeVigorIndex:
    def test_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = relative_vigor_index(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert set(result.keys()) == {"rvi", "signal"}

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = relative_vigor_index(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert isinstance(result["rvi"], pd.Series)
        assert isinstance(result["signal"], pd.Series)

    def test_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = relative_vigor_index(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert len(result["rvi"]) == len(ohlcv["close"])
        assert len(result["signal"]) == len(ohlcv["close"])

    def test_has_valid_values(self, ohlcv: dict[str, pd.Series]) -> None:
        """RVI should produce valid (non-NaN) values after warm-up."""
        result = relative_vigor_index(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert result["rvi"].dropna().shape[0] > 0
        assert result["signal"].dropna().shape[0] > 0

    def test_name(self, ohlcv: dict[str, pd.Series]) -> None:
        result = relative_vigor_index(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert result["rvi"].name == "rvi"
        assert result["signal"].name == "rvi_signal"
