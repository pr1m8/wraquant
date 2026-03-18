"""Tests for wraquant.ta.candles module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.candles import (
    average_candle_body,
    body_gap,
    body_to_range_ratio,
    candle_body_size,
    candle_direction,
    candle_momentum,
    candle_range,
    inside_bar,
    lower_shadow_ratio,
    outside_bar,
    pin_bar,
    upper_shadow_ratio,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlc() -> dict[str, pd.Series]:
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.2
    high = np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.3)
    low = np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.3)
    return {
        "open": pd.Series(open_, name="open"),
        "high": pd.Series(high, name="high"),
        "low": pd.Series(low, name="low"),
        "close": pd.Series(close, name="close"),
    }


# ---------------------------------------------------------------------------
# candle_body_size
# ---------------------------------------------------------------------------


class TestCandleBodySize:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = candle_body_size(ohlc["open"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = candle_body_size(ohlc["open"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_non_negative(self, ohlc: dict[str, pd.Series]) -> None:
        result = candle_body_size(ohlc["open"], ohlc["close"])
        assert (result >= 0).all()

    def test_manual(self) -> None:
        open_ = pd.Series([10.0, 20.0])
        close = pd.Series([15.0, 18.0])
        result = candle_body_size(open_, close)
        assert result.iloc[0] == pytest.approx(5.0)
        assert result.iloc[1] == pytest.approx(2.0)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            candle_body_size([1.0], [2.0])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# candle_range
# ---------------------------------------------------------------------------


class TestCandleRange:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = candle_range(ohlc["high"], ohlc["low"])
        assert len(result) == len(ohlc["close"])

    def test_non_negative(self, ohlc: dict[str, pd.Series]) -> None:
        result = candle_range(ohlc["high"], ohlc["low"])
        assert (result >= 0).all()

    def test_manual(self) -> None:
        high = pd.Series([20.0, 30.0])
        low = pd.Series([10.0, 25.0])
        result = candle_range(high, low)
        assert result.iloc[0] == pytest.approx(10.0)
        assert result.iloc[1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# upper_shadow_ratio
# ---------------------------------------------------------------------------


class TestUpperShadowRatio:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = upper_shadow_ratio(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = upper_shadow_ratio(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert (result >= 0).all()
        assert (result <= 1.0 + 1e-10).all()


# ---------------------------------------------------------------------------
# lower_shadow_ratio
# ---------------------------------------------------------------------------


class TestLowerShadowRatio:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = lower_shadow_ratio(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = lower_shadow_ratio(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert (result >= 0).all()
        assert (result <= 1.0 + 1e-10).all()


# ---------------------------------------------------------------------------
# body_to_range_ratio
# ---------------------------------------------------------------------------


class TestBodyToRangeRatio:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = body_to_range_ratio(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = body_to_range_ratio(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert (result >= 0).all()
        assert (result <= 1.0 + 1e-10).all()

    def test_ratios_sum(self, ohlc: dict[str, pd.Series]) -> None:
        """Body + upper + lower ratios should sum to 1 (within tolerance)."""
        br = body_to_range_ratio(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        ur = upper_shadow_ratio(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        lr = lower_shadow_ratio(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        total = br + ur + lr
        rng = ohlc["high"] - ohlc["low"]
        # Only check where range > 0
        mask = rng > 0
        np.testing.assert_allclose(total[mask].values, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# candle_direction
# ---------------------------------------------------------------------------


class TestCandleDirection:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = candle_direction(ohlc["open"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_values(self, ohlc: dict[str, pd.Series]) -> None:
        result = candle_direction(ohlc["open"], ohlc["close"])
        assert set(result.unique()).issubset({-1, 0, 1})

    def test_manual(self) -> None:
        open_ = pd.Series([10.0, 20.0, 15.0])
        close = pd.Series([15.0, 15.0, 15.0])
        result = candle_direction(open_, close)
        assert result.iloc[0] == 1   # bullish
        assert result.iloc[1] == -1  # bearish
        assert result.iloc[2] == 0   # doji


# ---------------------------------------------------------------------------
# average_candle_body
# ---------------------------------------------------------------------------


class TestAverageCandleBody:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = average_candle_body(ohlc["open"], ohlc["close"], period=14)
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = average_candle_body(ohlc["open"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_nan_initial(self, ohlc: dict[str, pd.Series]) -> None:
        result = average_candle_body(ohlc["open"], ohlc["close"], period=14)
        assert result.iloc[:13].isna().all()

    def test_invalid_period(self, ohlc: dict[str, pd.Series]) -> None:
        with pytest.raises(ValueError):
            average_candle_body(ohlc["open"], ohlc["close"], period=0)


# ---------------------------------------------------------------------------
# candle_momentum
# ---------------------------------------------------------------------------


class TestCandleMomentum:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = candle_momentum(ohlc["open"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = candle_momentum(ohlc["open"], ohlc["close"])
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# body_gap
# ---------------------------------------------------------------------------


class TestBodyGap:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = body_gap(ohlc["open"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_first_nan(self, ohlc: dict[str, pd.Series]) -> None:
        result = body_gap(ohlc["open"], ohlc["close"])
        assert pd.isna(result.iloc[0])

    def test_manual(self) -> None:
        open_ = pd.Series([10.0, 20.0])
        close = pd.Series([15.0, 25.0])
        result = body_gap(open_, close)
        # gap = open[1] - close[0] = 20 - 15 = 5
        assert result.iloc[1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# inside_bar
# ---------------------------------------------------------------------------


class TestInsideBar:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = inside_bar(ohlc["high"], ohlc["low"])
        assert len(result) == len(ohlc["close"])

    def test_dtype(self, ohlc: dict[str, pd.Series]) -> None:
        result = inside_bar(ohlc["high"], ohlc["low"])
        assert result.dtype == bool

    def test_manual(self) -> None:
        high = pd.Series([20.0, 18.0, 25.0])
        low = pd.Series([10.0, 12.0, 8.0])
        result = inside_bar(high, low)
        assert result.iloc[1] is np.bool_(True)  # 18<20 and 12>10
        assert result.iloc[2] is np.bool_(False)  # 25>18


# ---------------------------------------------------------------------------
# outside_bar
# ---------------------------------------------------------------------------


class TestOutsideBar:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = outside_bar(ohlc["high"], ohlc["low"])
        assert len(result) == len(ohlc["close"])

    def test_dtype(self, ohlc: dict[str, pd.Series]) -> None:
        result = outside_bar(ohlc["high"], ohlc["low"])
        assert result.dtype == bool

    def test_manual(self) -> None:
        high = pd.Series([20.0, 25.0])
        low = pd.Series([10.0, 8.0])
        result = outside_bar(high, low)
        assert result.iloc[1] is np.bool_(True)  # 25>20 and 8<10


# ---------------------------------------------------------------------------
# pin_bar
# ---------------------------------------------------------------------------


class TestPinBar:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = pin_bar(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_values(self, ohlc: dict[str, pd.Series]) -> None:
        result = pin_bar(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({-1, 0, 1})

    def test_bullish_pin(self) -> None:
        """Long lower shadow, short body near top."""
        open_ = pd.Series([19.0])
        high = pd.Series([20.0])
        low = pd.Series([10.0])
        close = pd.Series([19.5])
        result = pin_bar(open_, high, low, close)
        assert result.iloc[0] == 1

    def test_bearish_pin(self) -> None:
        """Long upper shadow, short body near bottom."""
        open_ = pd.Series([11.0])
        high = pd.Series([20.0])
        low = pd.Series([10.0])
        close = pd.Series([10.5])
        result = pin_bar(open_, high, low, close)
        assert result.iloc[0] == -1
