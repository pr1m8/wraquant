"""Tests for wraquant.ta.price_action module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.price_action import (
    gap_analysis,
    higher_highs_lows,
    key_reversal,
    narrow_range,
    pivot_reversal,
    range_expansion,
    swing_high,
    swing_low,
    trend_bars,
    wide_range_bar,
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
# higher_highs_lows
# ---------------------------------------------------------------------------


class TestHigherHighsLows:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = higher_highs_lows(ohlc["high"], ohlc["low"])
        assert len(result) == len(ohlc["close"])

    def test_values(self, ohlc: dict[str, pd.Series]) -> None:
        result = higher_highs_lows(ohlc["high"], ohlc["low"])
        assert set(result.unique()).issubset({-1, 0, 1})

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = higher_highs_lows(ohlc["high"], ohlc["low"])
        assert isinstance(result, pd.Series)

    def test_accepts_list_input(self) -> None:
        result = higher_highs_lows(list(range(10, 40)), list(range(1, 31)))
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# swing_high
# ---------------------------------------------------------------------------


class TestSwingHigh:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = swing_high(ohlc["high"])
        assert len(result) == len(ohlc["close"])

    def test_dtype(self, ohlc: dict[str, pd.Series]) -> None:
        result = swing_high(ohlc["high"])
        assert result.dtype == bool

    def test_manual(self) -> None:
        high = pd.Series([10.0, 12.0, 15.0, 13.0, 11.0])
        result = swing_high(high, lookback=2, lookahead=2)
        assert result.iloc[2] is np.bool_(True)  # 15 is highest

    def test_no_false_at_edges(self) -> None:
        high = pd.Series([10.0, 12.0, 15.0, 13.0, 11.0])
        result = swing_high(high, lookback=2, lookahead=2)
        assert result.iloc[0] is np.bool_(False)
        assert result.iloc[-1] is np.bool_(False)


# ---------------------------------------------------------------------------
# swing_low
# ---------------------------------------------------------------------------


class TestSwingLow:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = swing_low(ohlc["low"])
        assert len(result) == len(ohlc["close"])

    def test_dtype(self, ohlc: dict[str, pd.Series]) -> None:
        result = swing_low(ohlc["low"])
        assert result.dtype == bool

    def test_manual(self) -> None:
        low = pd.Series([15.0, 12.0, 8.0, 10.0, 13.0])
        result = swing_low(low, lookback=2, lookahead=2)
        assert result.iloc[2] is np.bool_(True)  # 8 is lowest


# ---------------------------------------------------------------------------
# trend_bars
# ---------------------------------------------------------------------------


class TestTrendBars:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = trend_bars(ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = trend_bars(ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_manual_up(self) -> None:
        close = pd.Series([10.0, 11.0, 12.0, 13.0])
        result = trend_bars(close)
        assert result.iloc[1] == 1
        assert result.iloc[2] == 2
        assert result.iloc[3] == 3

    def test_manual_down(self) -> None:
        close = pd.Series([13.0, 12.0, 11.0])
        result = trend_bars(close)
        assert result.iloc[1] == -1
        assert result.iloc[2] == -2


# ---------------------------------------------------------------------------
# gap_analysis
# ---------------------------------------------------------------------------


class TestGapAnalysis:
    def test_returns_dict(self, ohlc: dict[str, pd.Series]) -> None:
        result = gap_analysis(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, dict)
        assert "gap_size" in result
        assert "gap_direction" in result
        assert "gap_type" in result

    def test_lengths(self, ohlc: dict[str, pd.Series]) -> None:
        result = gap_analysis(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        for key in result:
            assert len(result[key]) == len(ohlc["close"])

    def test_direction_values(self, ohlc: dict[str, pd.Series]) -> None:
        result = gap_analysis(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result["gap_direction"].unique()).issubset({-1, 0, 1})

    def test_accepts_list_input(self) -> None:
        result = gap_analysis(
            list(range(5, 35)), list(range(10, 40)),
            list(range(1, 31)), list(range(5, 35)),
        )
        assert isinstance(result, (pd.DataFrame, dict))


# ---------------------------------------------------------------------------
# range_expansion
# ---------------------------------------------------------------------------


class TestRangeExpansion:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = range_expansion(ohlc["high"], ohlc["low"])
        assert len(result) == len(ohlc["close"])

    def test_dtype(self, ohlc: dict[str, pd.Series]) -> None:
        result = range_expansion(ohlc["high"], ohlc["low"])
        assert result.dtype == bool


# ---------------------------------------------------------------------------
# narrow_range
# ---------------------------------------------------------------------------


class TestNarrowRange:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = narrow_range(ohlc["high"], ohlc["low"])
        assert len(result) == len(ohlc["close"])

    def test_dtype(self, ohlc: dict[str, pd.Series]) -> None:
        result = narrow_range(ohlc["high"], ohlc["low"])
        assert result.dtype == bool

    def test_manual(self) -> None:
        high = pd.Series([20.0, 18.0, 15.0, 14.0])
        low = pd.Series([10.0, 12.0, 13.0, 13.5])
        result = narrow_range(high, low, period=4)
        # Ranges: 10, 6, 2, 0.5 — 0.5 is the narrowest
        assert result.iloc[3] is np.bool_(True)


# ---------------------------------------------------------------------------
# wide_range_bar
# ---------------------------------------------------------------------------


class TestWideRangeBar:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = wide_range_bar(ohlc["high"], ohlc["low"])
        assert len(result) == len(ohlc["close"])

    def test_dtype(self, ohlc: dict[str, pd.Series]) -> None:
        result = wide_range_bar(ohlc["high"], ohlc["low"])
        assert result.dtype == bool


# ---------------------------------------------------------------------------
# key_reversal
# ---------------------------------------------------------------------------


class TestKeyReversal:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = key_reversal(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_values(self, ohlc: dict[str, pd.Series]) -> None:
        result = key_reversal(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({-1, 0, 1})

    def test_bullish_manual(self) -> None:
        """New low + close above prev close = bullish key reversal."""
        open_ = pd.Series([50.0, 45.0])
        high = pd.Series([52.0, 55.0])
        low = pd.Series([48.0, 44.0])
        close = pd.Series([51.0, 53.0])
        result = key_reversal(open_, high, low, close)
        assert result.iloc[1] == 1

    def test_bearish_manual(self) -> None:
        """New high + close below prev close = bearish key reversal."""
        open_ = pd.Series([50.0, 55.0])
        high = pd.Series([52.0, 56.0])
        low = pd.Series([48.0, 47.0])
        close = pd.Series([51.0, 48.0])
        result = key_reversal(open_, high, low, close)
        assert result.iloc[1] == -1


# ---------------------------------------------------------------------------
# pivot_reversal
# ---------------------------------------------------------------------------


class TestPivotReversal:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = pivot_reversal(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_values(self, ohlc: dict[str, pd.Series]) -> None:
        result = pivot_reversal(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({-1, 0, 1})

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = pivot_reversal(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)
