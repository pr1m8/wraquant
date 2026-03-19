"""Tests for wraquant.ta.custom module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.custom import (
    adaptive_rsi,
    anchored_vwap,
    detrended_regression,
    ehlers_fisher,
    linear_regression_channel,
    linear_regression_forecast,
    market_structure,
    pivot_points,
    polynomial_regression,
    r_squared_indicator,
    raff_regression_channel,
    relative_strength,
    squeeze_momentum,
    standard_error_bands,
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

    def test_accepts_list_input(self) -> None:
        # Lists are now auto-coerced to pd.Series
        result = squeeze_momentum(
            list(range(1, 30)), list(range(1, 30)), list(range(1, 30))
        )
        assert isinstance(result, dict)


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


# ---------------------------------------------------------------------------
# Linear Regression Forecast
# ---------------------------------------------------------------------------


class TestLinearRegressionForecast:
    def test_output_type(self, close_series: pd.Series) -> None:
        result = linear_regression_forecast(close_series, period=20)
        assert isinstance(result, pd.Series)

    def test_output_length(self, close_series: pd.Series) -> None:
        result = linear_regression_forecast(close_series, period=20)
        assert len(result) == len(close_series)

    def test_nan_before_period(self, close_series: pd.Series) -> None:
        period = 20
        result = linear_regression_forecast(close_series, period=period)
        assert result.iloc[: period - 1].isna().all()

    def test_perfect_trend(self) -> None:
        """For a perfect linear series, forecast should be exact."""
        data = pd.Series(np.arange(50, dtype=float) * 2.0 + 10.0)
        result = linear_regression_forecast(data, period=10, forecast_bars=1)
        # At index 9 (end of first window), forecasting 1 bar ahead
        # should predict the next value in the linear sequence
        for i in range(9, len(data)):
            expected = data.iloc[i] + 2.0  # slope is 2.0
            assert abs(result.iloc[i] - expected) < 1e-8

    def test_accepts_list_input(self) -> None:
        result = linear_regression_forecast(list(range(1, 30)))
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Standard Error Bands
# ---------------------------------------------------------------------------


class TestStandardErrorBands:
    def test_output_keys(self, close_series: pd.Series) -> None:
        result = standard_error_bands(close_series, period=20, num_bands=3)
        expected = {"middle", "upper_1", "lower_1", "upper_2", "lower_2", "upper_3", "lower_3"}
        assert set(result.keys()) == expected

    def test_custom_num_bands(self, close_series: pd.Series) -> None:
        result = standard_error_bands(close_series, period=20, num_bands=2)
        expected = {"middle", "upper_1", "lower_1", "upper_2", "lower_2"}
        assert set(result.keys()) == expected

    def test_upper_above_lower(self, close_series: pd.Series) -> None:
        result = standard_error_bands(close_series, period=20)
        valid = result["upper_1"].notna() & result["lower_1"].notna()
        assert (result["upper_1"][valid] >= result["lower_1"][valid] - 1e-10).all()

    def test_output_length(self, close_series: pd.Series) -> None:
        result = standard_error_bands(close_series, period=20)
        assert len(result["middle"]) == len(close_series)


# ---------------------------------------------------------------------------
# R-Squared Indicator
# ---------------------------------------------------------------------------


class TestRSquaredIndicator:
    def test_output_type(self, close_series: pd.Series) -> None:
        result = r_squared_indicator(close_series, period=14)
        assert isinstance(result, pd.Series)

    def test_output_length(self, close_series: pd.Series) -> None:
        result = r_squared_indicator(close_series, period=14)
        assert len(result) == len(close_series)

    def test_bounds(self, close_series: pd.Series) -> None:
        result = r_squared_indicator(close_series, period=14)
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 1.0 + 1e-10).all()

    def test_perfect_trend(self) -> None:
        """R-squared of a perfect linear series should be 1.0."""
        data = pd.Series(np.arange(50, dtype=float) * 2.0 + 10.0)
        result = r_squared_indicator(data, period=10)
        valid = result.dropna()
        assert (abs(valid - 1.0) < 1e-8).all()

    def test_accepts_list_input(self) -> None:
        result = r_squared_indicator(list(range(1, 30)))
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Polynomial Regression
# ---------------------------------------------------------------------------


class TestPolynomialRegression:
    def test_output_type(self, close_series: pd.Series) -> None:
        result = polynomial_regression(close_series, period=20, degree=2)
        assert isinstance(result, pd.Series)

    def test_output_length(self, close_series: pd.Series) -> None:
        result = polynomial_regression(close_series, period=20, degree=2)
        assert len(result) == len(close_series)

    def test_nan_before_period(self, close_series: pd.Series) -> None:
        period = 20
        result = polynomial_regression(close_series, period=period)
        assert result.iloc[: period - 1].isna().all()

    def test_linear_matches_degree_1(self) -> None:
        """Degree 1 polynomial regression should match linear regression."""
        data = pd.Series(np.arange(30, dtype=float) * 3.0 + 5.0)
        poly_result = polynomial_regression(data, period=10, degree=1)
        valid = poly_result.dropna()
        # For a perfect linear series, the fitted value should be exact
        for i in range(9, len(data)):
            assert abs(poly_result.iloc[i] - data.iloc[i]) < 1e-6

    def test_invalid_degree(self) -> None:
        with pytest.raises(ValueError, match="degree"):
            polynomial_regression(pd.Series([1.0, 2.0, 3.0]), period=2, degree=0)


# ---------------------------------------------------------------------------
# Raff Regression Channel
# ---------------------------------------------------------------------------


class TestRaffRegressionChannel:
    def test_output_keys(self, close_series: pd.Series) -> None:
        result = raff_regression_channel(close_series, period=50)
        assert set(result.keys()) == {"center", "upper", "lower"}

    def test_upper_above_lower(self, close_series: pd.Series) -> None:
        result = raff_regression_channel(close_series, period=50)
        valid = result["upper"].notna() & result["lower"].notna()
        assert (result["upper"][valid] >= result["lower"][valid] - 1e-10).all()

    def test_output_length(self, close_series: pd.Series) -> None:
        result = raff_regression_channel(close_series, period=50)
        assert len(result["center"]) == len(close_series)

    def test_channel_width(self) -> None:
        """Upper and lower should be equidistant from center."""
        np.random.seed(42)
        data = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
        result = raff_regression_channel(data, period=30)
        valid = result["center"].notna()
        upper_dev = result["upper"][valid] - result["center"][valid]
        lower_dev = result["center"][valid] - result["lower"][valid]
        pd.testing.assert_series_equal(
            upper_dev.reset_index(drop=True),
            lower_dev.reset_index(drop=True),
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Detrended Regression
# ---------------------------------------------------------------------------


class TestDetrendedRegression:
    def test_output_type(self, close_series: pd.Series) -> None:
        result = detrended_regression(close_series, period=20)
        assert isinstance(result, pd.Series)

    def test_output_length(self, close_series: pd.Series) -> None:
        result = detrended_regression(close_series, period=20)
        assert len(result) == len(close_series)

    def test_nan_before_period(self, close_series: pd.Series) -> None:
        period = 20
        result = detrended_regression(close_series, period=period)
        assert result.iloc[: period - 1].isna().all()

    def test_perfect_trend_zero_residual(self) -> None:
        """For a perfect linear series, detrended values should be 0."""
        data = pd.Series(np.arange(50, dtype=float) * 2.0 + 10.0)
        result = detrended_regression(data, period=10)
        valid = result.dropna()
        assert (abs(valid) < 1e-8).all()

    def test_accepts_list_input(self) -> None:
        result = detrended_regression(list(range(1, 30)))
        assert isinstance(result, pd.Series)
