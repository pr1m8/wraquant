"""Tests for wraquant.ta.cycles module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.cycles import (
    bandpass_filter,
    decycler,
    even_better_sinewave,
    hilbert_instantaneous_phase,
    hilbert_transform_dominant_period,
    hilbert_transform_trend_mode,
    roofing_filter,
    sine_wave,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sine_series() -> pd.Series:
    """A clean sine wave for cycle detection."""
    t = np.linspace(0, 8 * np.pi, 300)
    return pd.Series(np.sin(t) * 10 + 100, name="sine_price")


@pytest.fixture
def trending_series() -> pd.Series:
    """A trending series with some noise."""
    np.random.seed(42)
    trend = np.linspace(100, 150, 300)
    noise = np.random.randn(300) * 0.5
    return pd.Series(trend + noise, name="trending_price")


@pytest.fixture
def close_series() -> pd.Series:
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(300) * 0.5)
    return pd.Series(prices, name="close")


# ---------------------------------------------------------------------------
# Hilbert Transform Dominant Period
# ---------------------------------------------------------------------------


class TestHilbertDominantPeriod:
    def test_output_length(self, close_series: pd.Series) -> None:
        result = hilbert_transform_dominant_period(close_series)
        assert len(result) == len(close_series)

    def test_output_type(self, close_series: pd.Series) -> None:
        result = hilbert_transform_dominant_period(close_series)
        assert isinstance(result, pd.Series)

    def test_period_bounds(self, close_series: pd.Series) -> None:
        """Period should be within [min_period, max_period]."""
        result = hilbert_transform_dominant_period(
            close_series, min_period=6, max_period=50
        )
        valid = result.dropna()
        # Skip warm-up period (first 50 values) where Hilbert hasn't converged
        converged = valid.iloc[50:] if len(valid) > 50 else valid
        if len(converged) > 0:
            assert (converged >= 6 - 1e-10).all()
            assert (converged <= 50 + 1e-10).all()

    def test_accepts_list_input(self) -> None:
        result = hilbert_transform_dominant_period(list(range(1, 100)))
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Hilbert Transform Trend Mode
# ---------------------------------------------------------------------------


class TestHilbertTrendMode:
    def test_output_length(self, close_series: pd.Series) -> None:
        result = hilbert_transform_trend_mode(close_series)
        assert len(result) == len(close_series)

    def test_binary_values(self, close_series: pd.Series) -> None:
        """Result should be 0 or 1 (or NaN)."""
        result = hilbert_transform_trend_mode(close_series)
        valid = result.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Hilbert Instantaneous Phase
# ---------------------------------------------------------------------------


class TestHilbertInstantaneousPhase:
    def test_output_length(self, close_series: pd.Series) -> None:
        result = hilbert_instantaneous_phase(close_series)
        assert len(result) == len(close_series)

    def test_output_type(self, close_series: pd.Series) -> None:
        result = hilbert_instantaneous_phase(close_series)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Sine Wave
# ---------------------------------------------------------------------------


class TestSineWave:
    def test_output_keys(self, close_series: pd.Series) -> None:
        result = sine_wave(close_series)
        assert set(result.keys()) == {"sine", "lead_sine"}

    def test_output_length(self, close_series: pd.Series) -> None:
        result = sine_wave(close_series)
        assert len(result["sine"]) == len(close_series)
        assert len(result["lead_sine"]) == len(close_series)


# ---------------------------------------------------------------------------
# Even Better Sinewave
# ---------------------------------------------------------------------------


class TestEvenBetterSinewave:
    def test_output_length(self, close_series: pd.Series) -> None:
        result = even_better_sinewave(close_series)
        assert len(result) == len(close_series)

    def test_bounds(self, close_series: pd.Series) -> None:
        """EBSW should be clamped to [-1, 1]."""
        result = even_better_sinewave(close_series)
        valid = result.dropna()
        assert (valid >= -1 - 1e-10).all()
        assert (valid <= 1 + 1e-10).all()

    def test_output_type(self, close_series: pd.Series) -> None:
        result = even_better_sinewave(close_series)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Roofing Filter
# ---------------------------------------------------------------------------


class TestRoofingFilter:
    def test_output_length(self, close_series: pd.Series) -> None:
        result = roofing_filter(close_series)
        assert len(result) == len(close_series)

    def test_output_type(self, close_series: pd.Series) -> None:
        result = roofing_filter(close_series)
        assert isinstance(result, pd.Series)

    def test_removes_trend(self, trending_series: pd.Series) -> None:
        """Roofing filter on a strong trend should produce small values."""
        result = roofing_filter(trending_series)
        # The filtered output should oscillate around zero
        valid = result.iloc[50:]  # Skip warmup
        assert abs(valid.mean()) < abs(trending_series.mean()) * 0.5


# ---------------------------------------------------------------------------
# Decycler
# ---------------------------------------------------------------------------


class TestDecycler:
    def test_output_length(self, close_series: pd.Series) -> None:
        result = decycler(close_series)
        assert len(result) == len(close_series)

    def test_output_type(self, close_series: pd.Series) -> None:
        result = decycler(close_series)
        assert isinstance(result, pd.Series)

    def test_preserves_trend(self, trending_series: pd.Series) -> None:
        """Decycler should preserve the general trend level."""
        result = decycler(trending_series, hp_period=125)
        # The mean of decycled series should be close to the original
        valid = result.iloc[10:]
        orig = trending_series.iloc[10:]
        assert abs(valid.mean() - orig.mean()) < 10.0


# ---------------------------------------------------------------------------
# Bandpass Filter
# ---------------------------------------------------------------------------


class TestBandpassFilter:
    def test_output_keys(self, close_series: pd.Series) -> None:
        result = bandpass_filter(close_series, period=20)
        assert set(result.keys()) == {"bp", "trigger"}

    def test_output_length(self, close_series: pd.Series) -> None:
        result = bandpass_filter(close_series, period=20)
        assert len(result["bp"]) == len(close_series)
        assert len(result["trigger"]) == len(close_series)

    def test_trigger_is_lagged_bp(self, close_series: pd.Series) -> None:
        """Trigger should be the one-bar lag of bp."""
        result = bandpass_filter(close_series, period=20)
        bp = result["bp"]
        trigger = result["trigger"]
        # trigger[i] == bp[i-1]
        pd.testing.assert_series_equal(
            trigger.iloc[1:].reset_index(drop=True).rename(None),
            bp.iloc[:-1].reset_index(drop=True).rename(None),
            atol=1e-10,
        )

    def test_oscillates_around_zero(self, close_series: pd.Series) -> None:
        """Bandpass filter output should oscillate around zero."""
        result = bandpass_filter(close_series, period=20)
        bp = result["bp"]
        valid = bp.iloc[30:]  # Skip warmup
        assert abs(valid.mean()) < 5.0

    def test_accepts_list_input(self) -> None:
        result = bandpass_filter(list(range(1, 100)))
        assert isinstance(result, dict)
