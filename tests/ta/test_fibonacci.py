"""Tests for wraquant.ta.fibonacci module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.fibonacci import (
    auto_fibonacci,
    fibonacci_extensions,
    fibonacci_fans,
    fibonacci_pivot_points,
    fibonacci_retracements,
    fibonacci_time_zones,
)


# ---------------------------------------------------------------------------
# Fibonacci Retracements
# ---------------------------------------------------------------------------


class TestFibonacciRetracements:
    def test_up_direction_levels(self) -> None:
        result = fibonacci_retracements(110.0, 100.0, direction="up")
        assert abs(result["0.0%"] - 110.0) < 1e-10
        assert abs(result["100.0%"] - 100.0) < 1e-10
        assert abs(result["50.0%"] - 105.0) < 1e-10

    def test_down_direction_levels(self) -> None:
        result = fibonacci_retracements(110.0, 100.0, direction="down")
        assert abs(result["0.0%"] - 100.0) < 1e-10
        assert abs(result["100.0%"] - 110.0) < 1e-10
        assert abs(result["50.0%"] - 105.0) < 1e-10

    def test_all_standard_levels_present(self) -> None:
        result = fibonacci_retracements(120.0, 100.0)
        expected_keys = {"0.0%", "23.6%", "38.2%", "50.0%", "61.8%", "78.6%", "100.0%"}
        assert set(result.keys()) == expected_keys

    def test_known_values(self) -> None:
        result = fibonacci_retracements(200.0, 100.0, direction="up")
        assert abs(result["23.6%"] - (200.0 - 0.236 * 100.0)) < 1e-10
        assert abs(result["38.2%"] - (200.0 - 0.382 * 100.0)) < 1e-10
        assert abs(result["61.8%"] - (200.0 - 0.618 * 100.0)) < 1e-10

    def test_invalid_swing(self) -> None:
        with pytest.raises(ValueError, match="swing_high"):
            fibonacci_retracements(100.0, 110.0)

    def test_equal_swing(self) -> None:
        with pytest.raises(ValueError, match="swing_high"):
            fibonacci_retracements(100.0, 100.0)

    def test_invalid_direction(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            fibonacci_retracements(110.0, 100.0, direction="sideways")


# ---------------------------------------------------------------------------
# Fibonacci Extensions
# ---------------------------------------------------------------------------


class TestFibonacciExtensions:
    def test_extension_keys(self) -> None:
        result = fibonacci_extensions(100.0, 110.0, 105.0)
        expected_keys = {"100.0%", "127.2%", "161.8%", "200.0%", "261.8%"}
        assert set(result.keys()) == expected_keys

    def test_known_values(self) -> None:
        result = fibonacci_extensions(100.0, 110.0, 105.0)
        diff = 10.0
        assert abs(result["100.0%"] - (105.0 + 1.0 * diff)) < 1e-10
        assert abs(result["161.8%"] - (105.0 + 1.618 * diff)) < 1e-10

    def test_invalid_swing(self) -> None:
        with pytest.raises(ValueError, match="swing_high"):
            fibonacci_extensions(110.0, 100.0, 105.0)


# ---------------------------------------------------------------------------
# Fibonacci Fans
# ---------------------------------------------------------------------------


class TestFibonacciFans:
    def test_fan_keys(self) -> None:
        result = fibonacci_fans(0, 100.0, 10, 110.0)
        expected_keys = {"38.2%", "50.0%", "61.8%"}
        assert set(result.keys()) == expected_keys

    def test_known_slopes(self) -> None:
        result = fibonacci_fans(0, 100.0, 10, 110.0)
        # 50% fan: y goes from 100 to 105 over 10 bars -> slope = 0.5
        assert abs(result["50.0%"] - 0.5) < 1e-10

    def test_zero_dx_raises(self) -> None:
        with pytest.raises(ValueError, match="different"):
            fibonacci_fans(5, 100.0, 5, 110.0)


# ---------------------------------------------------------------------------
# Fibonacci Time Zones
# ---------------------------------------------------------------------------


class TestFibonacciTimeZones:
    def test_from_zero(self) -> None:
        result = fibonacci_time_zones(0, 50)
        assert result == [1, 2, 3, 5, 8, 13, 21, 34]

    def test_from_offset(self) -> None:
        result = fibonacci_time_zones(10, 30)
        # 10+1=11, 10+2=12, 10+3=13, 10+5=15, 10+8=18, 10+13=23
        assert 11 in result
        assert 23 in result
        assert all(idx < 30 for idx in result)

    def test_invalid_range(self) -> None:
        with pytest.raises(ValueError, match="max_index"):
            fibonacci_time_zones(10, 5)

    def test_ascending(self) -> None:
        result = fibonacci_time_zones(0, 100)
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# Fibonacci Pivot Points
# ---------------------------------------------------------------------------


class TestFibonacciPivotPoints:
    @pytest.fixture
    def ohlc(self) -> dict[str, pd.Series]:
        return {
            "high": pd.Series([12, 13, 14, 13, 12], dtype=float),
            "low": pd.Series([10, 11, 12, 11, 10], dtype=float),
            "close": pd.Series([11, 12, 13, 12, 11], dtype=float),
        }

    def test_output_keys(self, ohlc: dict[str, pd.Series]) -> None:
        result = fibonacci_pivot_points(ohlc["high"], ohlc["low"], ohlc["close"])
        expected = {"pivot", "r1", "r2", "r3", "s1", "s2", "s3"}
        assert set(result.keys()) == expected

    def test_output_length(self, ohlc: dict[str, pd.Series]) -> None:
        result = fibonacci_pivot_points(ohlc["high"], ohlc["low"], ohlc["close"])
        for key in result:
            assert len(result[key]) == len(ohlc["close"])

    def test_resistance_above_support(self, ohlc: dict[str, pd.Series]) -> None:
        result = fibonacci_pivot_points(ohlc["high"], ohlc["low"], ohlc["close"])
        valid = result["r1"].notna() & result["s1"].notna()
        assert (result["r1"][valid] >= result["s1"][valid] - 1e-10).all()

    def test_accepts_list_input(self) -> None:
        result = fibonacci_pivot_points(
            list(range(10, 40)), list(range(1, 31)), list(range(5, 35))
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Auto Fibonacci
# ---------------------------------------------------------------------------


class TestAutoFibonacci:
    @pytest.fixture
    def close_series(self) -> pd.Series:
        np.random.seed(42)
        return pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5), name="close")

    def test_output_keys(self, close_series: pd.Series) -> None:
        result = auto_fibonacci(close_series, lookback=30)
        expected = {"swing_high", "swing_high_idx", "swing_low", "swing_low_idx", "levels"}
        assert set(result.keys()) == expected

    def test_levels_are_dict(self, close_series: pd.Series) -> None:
        result = auto_fibonacci(close_series, lookback=30)
        assert isinstance(result["levels"], dict)

    def test_swing_high_ge_swing_low(self, close_series: pd.Series) -> None:
        result = auto_fibonacci(close_series, lookback=30)
        assert result["swing_high"] >= result["swing_low"]

    def test_accepts_list_input(self) -> None:
        data = list(range(1, 100))
        result = auto_fibonacci(data, lookback=30)
        assert isinstance(result, dict)

    def test_small_lookback(self) -> None:
        with pytest.raises(ValueError, match="lookback"):
            auto_fibonacci(pd.Series([1.0, 2.0, 3.0]), lookback=1)
