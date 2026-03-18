"""Tests for wraquant.ta.custom module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.custom import (
    adaptive_rsi,
    anchored_vwap,
    ehlers_fisher,
    linear_regression_channel,
    market_structure,
    pivot_points,
    relative_strength,
    squeeze_momentum,
    swing_points,
    volume_weighted_macd,
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
# Squeeze Momentum
# ---------------------------------------------------------------------------


class TestSqueezeMomentum:
    def test_output_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = squeeze_momentum(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert set(result.keys()) == {"squeeze_on", "momentum"}

    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = squeeze_momentum(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(result["squeeze_on"]) == len(ohlcv["close"])
        assert len(result["momentum"]) == len(ohlcv["close"])

    def test_squeeze_on_binary(self, ohlcv: dict[str, pd.Series]) -> None:
        """squeeze_on should be 0 or 1."""
        result = squeeze_momentum(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = result["squeeze_on"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_type_error(self) -> None:
        with pytest.raises(TypeError):
            squeeze_momentum([1], [2], [3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Anchored VWAP
# ---------------------------------------------------------------------------


class TestAnchoredVWAP:
    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = anchored_vwap(ohlcv["close"], ohlcv["volume"], anchor_index=10)
        assert len(result) == len(ohlcv["close"])

    def test_nan_before_anchor(self, ohlcv: dict[str, pd.Series]) -> None:
        anchor = 10
        result = anchored_vwap(ohlcv["close"], ohlcv["volume"], anchor_index=anchor)
        assert result.iloc[:anchor].isna().all()

    def test_first_value_equals_close(self, ohlcv: dict[str, pd.Series]) -> None:
        """At the anchor point, VWAP should equal the close price."""
        anchor = 10
        result = anchored_vwap(ohlcv["close"], ohlcv["volume"], anchor_index=anchor)
        assert abs(result.iloc[anchor] - ohlcv["close"].iloc[anchor]) < 1e-10

    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = anchored_vwap(ohlcv["close"], ohlcv["volume"])
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Linear Regression Channel
# ---------------------------------------------------------------------------


class TestLinearRegressionChannel:
    def test_output_keys(self, close_series: pd.Series) -> None:
        result = linear_regression_channel(close_series, period=50)
        assert set(result.keys()) == {"middle", "upper", "lower"}

    def test_upper_above_lower(self, close_series: pd.Series) -> None:
        result = linear_regression_channel(close_series, period=50)
        valid_mask = result["upper"].notna() & result["lower"].notna()
        assert (result["upper"][valid_mask] >= result["lower"][valid_mask] - 1e-10).all()

    def test_output_length(self, close_series: pd.Series) -> None:
        result = linear_regression_channel(close_series, period=50)
        assert len(result["middle"]) == len(close_series)


# ---------------------------------------------------------------------------
# Pivot Points
# ---------------------------------------------------------------------------


class TestPivotPoints:
    def test_output_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pivot_points(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert set(result.keys()) == {"pivot", "r1", "r2", "s1", "s2"}

    def test_standard_method(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pivot_points(ohlcv["high"], ohlcv["low"], ohlcv["close"], method="standard")
        valid = result["pivot"].dropna()
        assert len(valid) > 0

    def test_fibonacci_method(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pivot_points(ohlcv["high"], ohlcv["low"], ohlcv["close"], method="fibonacci")
        valid = result["pivot"].dropna()
        assert len(valid) > 0

    def test_woodie_method(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pivot_points(ohlcv["high"], ohlcv["low"], ohlcv["close"], method="woodie")
        valid = result["pivot"].dropna()
        assert len(valid) > 0

    def test_invalid_method(self, ohlcv: dict[str, pd.Series]) -> None:
        with pytest.raises(ValueError, match="method must be"):
            pivot_points(ohlcv["high"], ohlcv["low"], ohlcv["close"], method="invalid")

    def test_resistance_above_support(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pivot_points(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid_mask = result["r1"].notna() & result["s1"].notna()
        assert (result["r1"][valid_mask] >= result["s1"][valid_mask] - 1e-10).all()

    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pivot_points(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(result["pivot"]) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# Market Structure
# ---------------------------------------------------------------------------


class TestMarketStructure:
    def test_output_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = market_structure(ohlcv["high"], ohlcv["low"], lookback=5)
        assert set(result.keys()) == {"swing_high", "swing_low", "structure"}

    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = market_structure(ohlcv["high"], ohlcv["low"], lookback=5)
        assert len(result["swing_high"]) == len(ohlcv["high"])
        assert len(result["swing_low"]) == len(ohlcv["low"])
        assert len(result["structure"]) == len(ohlcv["high"])

    def test_structure_values(self, ohlcv: dict[str, pd.Series]) -> None:
        """Structure should be -1, 0, or 1."""
        result = market_structure(ohlcv["high"], ohlcv["low"], lookback=5)
        valid = result["structure"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})


# ---------------------------------------------------------------------------
# Swing Points
# ---------------------------------------------------------------------------


class TestSwingPoints:
    def test_output_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = swing_points(ohlcv["high"], ohlcv["low"], lookback=5)
        assert set(result.keys()) == {"swing_high", "swing_low"}

    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = swing_points(ohlcv["high"], ohlcv["low"], lookback=5)
        assert len(result["swing_high"]) == len(ohlcv["high"])
        assert len(result["swing_low"]) == len(ohlcv["low"])

    def test_swing_highs_match_high_values(self, ohlcv: dict[str, pd.Series]) -> None:
        """Swing highs should equal the high values at those indices."""
        result = swing_points(ohlcv["high"], ohlcv["low"], lookback=3)
        sh = result["swing_high"]
        valid_idx = sh.dropna().index
        for idx in valid_idx:
            assert abs(sh.loc[idx] - ohlcv["high"].loc[idx]) < 1e-10

    def test_no_edge_swing_points(self, ohlcv: dict[str, pd.Series]) -> None:
        """No swing points at the edges (within lookback range)."""
        lookback = 5
        result = swing_points(ohlcv["high"], ohlcv["low"], lookback=lookback)
        assert result["swing_high"].iloc[:lookback].isna().all()
        assert result["swing_low"].iloc[:lookback].isna().all()


# ---------------------------------------------------------------------------
# Volume-Weighted MACD
# ---------------------------------------------------------------------------


class TestVolumeWeightedMACD:
    def test_output_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = volume_weighted_macd(ohlcv["close"], ohlcv["volume"])
        assert set(result.keys()) == {"macd", "signal", "histogram"}

    def test_histogram_equals_macd_minus_signal(self, ohlcv: dict[str, pd.Series]) -> None:
        result = volume_weighted_macd(ohlcv["close"], ohlcv["volume"])
        hist_expected = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(
            result["histogram"].rename(None),
            hist_expected.rename(None),
            atol=1e-10,
        )

    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = volume_weighted_macd(ohlcv["close"], ohlcv["volume"])
        assert len(result["macd"]) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# Ehlers Fisher
# ---------------------------------------------------------------------------


class TestEhlersFisher:
    def test_output_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = ehlers_fisher(ohlcv["high"], ohlcv["low"])
        assert set(result.keys()) == {"fisher", "trigger"}

    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = ehlers_fisher(ohlcv["high"], ohlcv["low"])
        assert len(result["fisher"]) == len(ohlcv["high"])

    def test_trigger_is_lagged(self, ohlcv: dict[str, pd.Series]) -> None:
        """Trigger is one-bar lag of fisher (where valid)."""
        result = ehlers_fisher(ohlcv["high"], ohlcv["low"], period=10)
        fisher = result["fisher"]
        trigger = result["trigger"]
        # After warmup, trigger[i] should equal fisher[i-1]
        valid_start = 10  # period
        for i in range(valid_start + 1, min(len(fisher), valid_start + 20)):
            if not np.isnan(fisher.iloc[i - 1]) and not np.isnan(trigger.iloc[i]):
                assert abs(trigger.iloc[i] - fisher.iloc[i - 1]) < 1e-10


# ---------------------------------------------------------------------------
# Adaptive RSI
# ---------------------------------------------------------------------------


class TestAdaptiveRSI:
    def test_bounds(self, close_series: pd.Series) -> None:
        result = adaptive_rsi(close_series)
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 100 + 1e-10).all()

    def test_output_length(self, close_series: pd.Series) -> None:
        result = adaptive_rsi(close_series)
        assert len(result) == len(close_series)

    def test_output_type(self, close_series: pd.Series) -> None:
        result = adaptive_rsi(close_series)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Relative Strength
# ---------------------------------------------------------------------------


class TestRelativeStrength:
    def test_output_length(self, close_series: pd.Series) -> None:
        benchmark = pd.Series(np.arange(len(close_series), dtype=float) + 100)
        result = relative_strength(close_series, benchmark)
        assert len(result) == len(close_series)

    def test_self_ratio_is_one(self) -> None:
        data = pd.Series([100, 110, 120, 130, 140], dtype=float)
        result = relative_strength(data, data)
        assert (abs(result - 1.0) < 1e-10).all()

    def test_output_type(self) -> None:
        data = pd.Series([100, 110, 120], dtype=float)
        benchmark = pd.Series([200, 220, 240], dtype=float)
        result = relative_strength(data, benchmark)
        assert isinstance(result, pd.Series)

    def test_known_value(self) -> None:
        data = pd.Series([100.0, 200.0])
        benchmark = pd.Series([50.0, 100.0])
        result = relative_strength(data, benchmark)
        assert abs(result.iloc[0] - 2.0) < 1e-10
        assert abs(result.iloc[1] - 2.0) < 1e-10
