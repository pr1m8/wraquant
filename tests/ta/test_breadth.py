"""Tests for wraquant.ta.breadth module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.breadth import (
    advance_decline_line,
    advance_decline_ratio,
    arms_index,
    bullish_percent,
    cumulative_volume_index,
    high_low_index,
    mcclellan_oscillator,
    mcclellan_summation,
    new_highs_lows,
    percent_above_ma,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def breadth_data() -> dict[str, pd.Series]:
    np.random.seed(42)
    n = 200
    advancing = pd.Series(np.random.randint(100, 400, size=n).astype(float), name="adv")
    declining = pd.Series(np.random.randint(100, 400, size=n).astype(float), name="dec")
    adv_volume = pd.Series(
        np.random.randint(1_000_000, 5_000_000, size=n).astype(float), name="adv_vol"
    )
    dec_volume = pd.Series(
        np.random.randint(1_000_000, 5_000_000, size=n).astype(float), name="dec_vol"
    )
    new_highs = pd.Series(np.random.randint(10, 100, size=n).astype(float), name="nh")
    new_lows = pd.Series(np.random.randint(10, 100, size=n).astype(float), name="nl")
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), name="close")
    volume = pd.Series(np.random.randint(1000, 10000, size=n).astype(float), name="vol")
    return {
        "advancing": advancing,
        "declining": declining,
        "adv_volume": adv_volume,
        "dec_volume": dec_volume,
        "new_highs": new_highs,
        "new_lows": new_lows,
        "close": close,
        "volume": volume,
    }


@pytest.fixture
def prices_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "A": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "B": 50 + np.cumsum(np.random.randn(n) * 0.3),
            "C": 200 + np.cumsum(np.random.randn(n) * 0.8),
        }
    )


# ---------------------------------------------------------------------------
# Advance/Decline Line
# ---------------------------------------------------------------------------


class TestAdvanceDeclineLine:
    def test_output_type(self, breadth_data: dict[str, pd.Series]) -> None:
        result = advance_decline_line(
            breadth_data["advancing"], breadth_data["declining"]
        )
        assert isinstance(result, pd.Series)

    def test_length(self, breadth_data: dict[str, pd.Series]) -> None:
        result = advance_decline_line(
            breadth_data["advancing"], breadth_data["declining"]
        )
        assert len(result) == len(breadth_data["advancing"])

    def test_cumulative(self) -> None:
        """A/D line should be a cumulative sum."""
        adv = pd.Series([200, 250, 180.0])
        dec = pd.Series([100, 150, 220.0])
        result = advance_decline_line(adv, dec)
        # (200-100) + (250-150) + (180-220) = 100 + 100 - 40 = 160
        assert abs(result.iloc[2] - 160.0) < 1e-10

    def test_name(self, breadth_data: dict[str, pd.Series]) -> None:
        result = advance_decline_line(
            breadth_data["advancing"], breadth_data["declining"]
        )
        assert result.name == "ad_line"

    def test_type_error(self) -> None:
        with pytest.raises(TypeError):
            advance_decline_line([1, 2, 3], pd.Series([1, 2, 3]))


# ---------------------------------------------------------------------------
# Advance/Decline Ratio
# ---------------------------------------------------------------------------


class TestAdvanceDeclineRatio:
    def test_output_type(self, breadth_data: dict[str, pd.Series]) -> None:
        result = advance_decline_ratio(
            breadth_data["advancing"], breadth_data["declining"]
        )
        assert isinstance(result, pd.Series)

    def test_manual(self) -> None:
        adv = pd.Series([200.0, 100.0])
        dec = pd.Series([100.0, 200.0])
        result = advance_decline_ratio(adv, dec)
        assert abs(result.iloc[0] - 2.0) < 1e-10
        assert abs(result.iloc[1] - 0.5) < 1e-10

    def test_name(self, breadth_data: dict[str, pd.Series]) -> None:
        result = advance_decline_ratio(
            breadth_data["advancing"], breadth_data["declining"]
        )
        assert result.name == "ad_ratio"

    def test_zero_declining(self) -> None:
        """Zero declining should produce NaN."""
        adv = pd.Series([100.0])
        dec = pd.Series([0.0])
        result = advance_decline_ratio(adv, dec)
        assert result.iloc[0] != result.iloc[0]  # NaN check


# ---------------------------------------------------------------------------
# McClellan Oscillator
# ---------------------------------------------------------------------------


class TestMcClellanOscillator:
    def test_output_type(self, breadth_data: dict[str, pd.Series]) -> None:
        result = mcclellan_oscillator(
            breadth_data["advancing"], breadth_data["declining"]
        )
        assert isinstance(result, pd.Series)

    def test_length(self, breadth_data: dict[str, pd.Series]) -> None:
        result = mcclellan_oscillator(
            breadth_data["advancing"], breadth_data["declining"]
        )
        assert len(result) == len(breadth_data["advancing"])

    def test_name(self, breadth_data: dict[str, pd.Series]) -> None:
        result = mcclellan_oscillator(
            breadth_data["advancing"], breadth_data["declining"]
        )
        assert result.name == "mcclellan_oscillator"

    def test_has_valid_values(self, breadth_data: dict[str, pd.Series]) -> None:
        result = mcclellan_oscillator(
            breadth_data["advancing"], breadth_data["declining"]
        )
        assert result.dropna().shape[0] > 0


# ---------------------------------------------------------------------------
# McClellan Summation
# ---------------------------------------------------------------------------


class TestMcClellanSummation:
    def test_output_type(self, breadth_data: dict[str, pd.Series]) -> None:
        result = mcclellan_summation(
            breadth_data["advancing"], breadth_data["declining"]
        )
        assert isinstance(result, pd.Series)

    def test_name(self, breadth_data: dict[str, pd.Series]) -> None:
        result = mcclellan_summation(
            breadth_data["advancing"], breadth_data["declining"]
        )
        assert result.name == "mcclellan_summation"


# ---------------------------------------------------------------------------
# Arms Index (TRIN)
# ---------------------------------------------------------------------------


class TestArmsIndex:
    def test_output_type(self, breadth_data: dict[str, pd.Series]) -> None:
        result = arms_index(
            breadth_data["advancing"],
            breadth_data["declining"],
            breadth_data["adv_volume"],
            breadth_data["dec_volume"],
        )
        assert isinstance(result, pd.Series)

    def test_manual(self) -> None:
        adv_i = pd.Series([200.0])
        dec_i = pd.Series([100.0])
        adv_v = pd.Series([1_000_000.0])
        dec_v = pd.Series([500_000.0])
        result = arms_index(adv_i, dec_i, adv_v, dec_v)
        # (200/100) / (1M/0.5M) = 2.0 / 2.0 = 1.0
        assert abs(result.iloc[0] - 1.0) < 1e-10

    def test_name(self, breadth_data: dict[str, pd.Series]) -> None:
        result = arms_index(
            breadth_data["advancing"],
            breadth_data["declining"],
            breadth_data["adv_volume"],
            breadth_data["dec_volume"],
        )
        assert result.name == "arms_index"


# ---------------------------------------------------------------------------
# New Highs - New Lows
# ---------------------------------------------------------------------------


class TestNewHighsLows:
    def test_output_type(self, breadth_data: dict[str, pd.Series]) -> None:
        result = new_highs_lows(breadth_data["new_highs"], breadth_data["new_lows"])
        assert isinstance(result, pd.Series)

    def test_manual(self) -> None:
        nh = pd.Series([50.0, 30.0])
        nl = pd.Series([20.0, 50.0])
        result = new_highs_lows(nh, nl)
        assert abs(result.iloc[0] - 30.0) < 1e-10
        assert abs(result.iloc[1] - (-20.0)) < 1e-10

    def test_name(self, breadth_data: dict[str, pd.Series]) -> None:
        result = new_highs_lows(breadth_data["new_highs"], breadth_data["new_lows"])
        assert result.name == "new_highs_lows"


# ---------------------------------------------------------------------------
# Percent Above MA
# ---------------------------------------------------------------------------


class TestPercentAboveMA:
    def test_output_type(self, prices_df: pd.DataFrame) -> None:
        result = percent_above_ma(prices_df, period=10)
        assert isinstance(result, pd.Series)

    def test_bounds(self, prices_df: pd.DataFrame) -> None:
        result = percent_above_ma(prices_df, period=10)
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 100 + 1e-10).all()

    def test_name(self, prices_df: pd.DataFrame) -> None:
        result = percent_above_ma(prices_df, period=10)
        assert result.name == "percent_above_ma"

    def test_type_error(self) -> None:
        with pytest.raises(TypeError):
            percent_above_ma(pd.Series([1, 2, 3]), period=2)

    def test_all_above(self) -> None:
        """When all components are strictly above their MA, result should be 100."""
        # Strictly increasing — every column is above its SMA after warm-up
        df = pd.DataFrame(
            {
                "A": np.arange(1.0, 21.0),
                "B": np.arange(1.0, 21.0) * 2,
            }
        )
        result = percent_above_ma(df, period=3)
        valid = result.dropna()
        # Skip warm-up period where SMA hasn't fully formed
        converged = valid.iloc[2:]
        assert (converged == 100.0).all()


# ---------------------------------------------------------------------------
# High-Low Index
# ---------------------------------------------------------------------------


class TestHighLowIndex:
    def test_output_type(self, breadth_data: dict[str, pd.Series]) -> None:
        result = high_low_index(breadth_data["new_highs"], breadth_data["new_lows"])
        assert isinstance(result, pd.Series)

    def test_bounds(self, breadth_data: dict[str, pd.Series]) -> None:
        result = high_low_index(breadth_data["new_highs"], breadth_data["new_lows"])
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 100 + 1e-10).all()

    def test_manual(self) -> None:
        nh = pd.Series([75.0])
        nl = pd.Series([25.0])
        result = high_low_index(nh, nl)
        assert abs(result.iloc[0] - 75.0) < 1e-10

    def test_name(self, breadth_data: dict[str, pd.Series]) -> None:
        result = high_low_index(breadth_data["new_highs"], breadth_data["new_lows"])
        assert result.name == "high_low_index"


# ---------------------------------------------------------------------------
# Bullish Percent
# ---------------------------------------------------------------------------


class TestBullishPercent:
    def test_output_type(self, prices_df: pd.DataFrame) -> None:
        result = bullish_percent(prices_df, period=10)
        assert isinstance(result, pd.Series)

    def test_name(self, prices_df: pd.DataFrame) -> None:
        result = bullish_percent(prices_df, period=10)
        assert result.name == "bullish_percent"

    def test_bounds(self, prices_df: pd.DataFrame) -> None:
        result = bullish_percent(prices_df, period=10)
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 100 + 1e-10).all()


# ---------------------------------------------------------------------------
# Cumulative Volume Index
# ---------------------------------------------------------------------------


class TestCumulativeVolumeIndex:
    def test_output_type(self, breadth_data: dict[str, pd.Series]) -> None:
        result = cumulative_volume_index(
            breadth_data["close"], breadth_data["volume"]
        )
        assert isinstance(result, pd.Series)

    def test_length(self, breadth_data: dict[str, pd.Series]) -> None:
        result = cumulative_volume_index(
            breadth_data["close"], breadth_data["volume"]
        )
        assert len(result) == len(breadth_data["close"])

    def test_name(self, breadth_data: dict[str, pd.Series]) -> None:
        result = cumulative_volume_index(
            breadth_data["close"], breadth_data["volume"]
        )
        assert result.name == "cvi"

    def test_manual(self) -> None:
        close = pd.Series([100, 102, 101, 103.0])
        volume = pd.Series([1000, 1500, 1200, 1800.0])
        result = cumulative_volume_index(close, volume)
        # idx 0: direction=0 → 0
        # idx 1: up → +1500 → 1500
        # idx 2: down → -1200 → 300
        # idx 3: up → +1800 → 2100
        assert abs(result.iloc[0] - 0.0) < 1e-10
        assert abs(result.iloc[1] - 1500.0) < 1e-10
        assert abs(result.iloc[2] - 300.0) < 1e-10
        assert abs(result.iloc[3] - 2100.0) < 1e-10
