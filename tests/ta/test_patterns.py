"""Tests for wraquant.ta.patterns module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.patterns import (
    abandoned_baby,
    belt_hold,
    dark_cloud_cover,
    doji,
    engulfing,
    evening_star,
    hammer,
    hanging_man,
    harami,
    inverted_hammer,
    kicking,
    marubozu,
    morning_star,
    piercing_pattern,
    shooting_star,
    spinning_top,
    three_black_crows,
    three_inside_down,
    three_inside_up,
    three_white_soldiers,
    tweezer_bottom,
    tweezer_top,
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
# Existing patterns (basic smoke tests)
# ---------------------------------------------------------------------------


class TestDoji:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = doji(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_values_binary(self, ohlc: dict[str, pd.Series]) -> None:
        result = doji(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, 1})


class TestHammer:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = hammer(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])


class TestEngulfing:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = engulfing(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])


class TestMorningStar:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = morning_star(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])


class TestEveningStar:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = evening_star(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])


class TestThreeWhiteSoldiers:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = three_white_soldiers(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])


class TestThreeBlackCrows:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = three_black_crows(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])


class TestHarami:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = harami(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])


class TestSpinningTop:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = spinning_top(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])


class TestMarubozu:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = marubozu(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])


# ---------------------------------------------------------------------------
# Piercing Pattern
# ---------------------------------------------------------------------------


class TestPiercingPattern:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = piercing_pattern(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = piercing_pattern(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = piercing_pattern(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, 1})

    def test_manual_detection(self) -> None:
        """Construct a known piercing pattern."""
        # Day 1: bearish (open=50, close=40), Day 2: opens below low, closes above midpoint
        open_ = pd.Series([50.0, 38.0])
        high = pd.Series([52.0, 48.0])
        low = pd.Series([39.0, 37.0])
        close = pd.Series([40.0, 46.0])
        result = piercing_pattern(open_, high, low, close)
        assert result.iloc[1] == 1

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            piercing_pattern([1.0], [2.0], [0.5], [1.5])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dark Cloud Cover
# ---------------------------------------------------------------------------


class TestDarkCloudCover:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = dark_cloud_cover(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = dark_cloud_cover(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = dark_cloud_cover(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, -1})

    def test_manual_detection(self) -> None:
        """Construct a known dark cloud cover."""
        # Day 1: bullish (open=40, close=50), Day 2: opens above high, closes below mid
        open_ = pd.Series([40.0, 53.0])
        high = pd.Series([51.0, 54.0])
        low = pd.Series([39.0, 43.0])
        close = pd.Series([50.0, 44.0])
        result = dark_cloud_cover(open_, high, low, close)
        assert result.iloc[1] == -1


# ---------------------------------------------------------------------------
# Hanging Man
# ---------------------------------------------------------------------------


class TestHangingMan:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = hanging_man(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = hanging_man(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = hanging_man(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, -1})


# ---------------------------------------------------------------------------
# Inverted Hammer
# ---------------------------------------------------------------------------


class TestInvertedHammer:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = inverted_hammer(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = inverted_hammer(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = inverted_hammer(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Shooting Star
# ---------------------------------------------------------------------------


class TestShootingStar:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = shooting_star(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = shooting_star(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = shooting_star(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, -1})


# ---------------------------------------------------------------------------
# Tweezer Top
# ---------------------------------------------------------------------------


class TestTweezerTop:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = tweezer_top(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = tweezer_top(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = tweezer_top(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, -1})

    def test_manual_detection(self) -> None:
        """Two candles with same highs: bullish then bearish."""
        open_ = pd.Series([48.0, 52.0])
        high = pd.Series([55.0, 55.0])
        low = pd.Series([47.0, 49.0])
        close = pd.Series([53.0, 50.0])
        result = tweezer_top(open_, high, low, close, tolerance=0.001)
        assert result.iloc[1] == -1


# ---------------------------------------------------------------------------
# Tweezer Bottom
# ---------------------------------------------------------------------------


class TestTweezerBottom:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = tweezer_bottom(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = tweezer_bottom(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = tweezer_bottom(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, 1})

    def test_manual_detection(self) -> None:
        """Two candles with same lows: bearish then bullish."""
        open_ = pd.Series([52.0, 46.0])
        high = pd.Series([53.0, 51.0])
        low = pd.Series([45.0, 45.0])
        close = pd.Series([47.0, 50.0])
        result = tweezer_bottom(open_, high, low, close, tolerance=0.001)
        assert result.iloc[1] == 1


# ---------------------------------------------------------------------------
# Three Inside Up
# ---------------------------------------------------------------------------


class TestThreeInsideUp:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = three_inside_up(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = three_inside_up(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = three_inside_up(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Three Inside Down
# ---------------------------------------------------------------------------


class TestThreeInsideDown:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = three_inside_down(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = three_inside_down(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = three_inside_down(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, -1})


# ---------------------------------------------------------------------------
# Abandoned Baby
# ---------------------------------------------------------------------------


class TestAbandonedBaby:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = abandoned_baby(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = abandoned_baby(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = abandoned_baby(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, 1, -1})

    def test_manual_bullish_detection(self) -> None:
        """Construct a known bullish abandoned baby."""
        # Day 1: bearish; Day 2: doji gapped below; Day 3: bullish gapped above
        open_ = pd.Series([50.0, 38.0, 42.0])
        high = pd.Series([51.0, 39.0, 48.0])
        low = pd.Series([40.0, 37.5, 41.0])
        close = pd.Series([41.0, 38.05, 47.0])
        result = abandoned_baby(open_, high, low, close)
        assert result.iloc[2] == 1


# ---------------------------------------------------------------------------
# Kicking
# ---------------------------------------------------------------------------


class TestKicking:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = kicking(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = kicking(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = kicking(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, 1, -1})


# ---------------------------------------------------------------------------
# Belt Hold
# ---------------------------------------------------------------------------


class TestBeltHold:
    def test_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = belt_hold(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert len(result) == len(ohlc["close"])

    def test_returns_series(self, ohlc: dict[str, pd.Series]) -> None:
        result = belt_hold(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert isinstance(result, pd.Series)

    def test_values_range(self, ohlc: dict[str, pd.Series]) -> None:
        result = belt_hold(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert set(result.unique()).issubset({0, 1, -1})

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            belt_hold([1.0], [2.0], [0.5], [1.5])  # type: ignore[arg-type]
