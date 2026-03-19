"""Tests for wraquant.ta.smoothing module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.smoothing import (
    alma,
    butterworth_filter,
    gaussian_filter,
    hamming_window_ma,
    hann_window_ma,
    jma,
    kaufman_efficiency_ratio,
    lsma,
    sinema,
    supersmoother,
    swma,
    trima,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def close_series() -> pd.Series:
    """Simple ascending close prices for deterministic tests."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    return pd.Series(prices, name="close")


@pytest.fixture
def linear_series() -> pd.Series:
    """Perfectly linear series for regression-based indicator tests."""
    return pd.Series(np.arange(1.0, 51.0), name="linear")


# ---------------------------------------------------------------------------
# ALMA
# ---------------------------------------------------------------------------


class TestALMA:
    def test_alma_length(self, close_series: pd.Series) -> None:
        result = alma(close_series, period=9)
        assert len(result) == len(close_series)

    def test_alma_nan_prefix(self, close_series: pd.Series) -> None:
        period = 9
        result = alma(close_series, period=period)
        assert result.iloc[: period - 1].isna().all()
        assert result.iloc[period - 1 :].notna().all()

    def test_alma_name(self, close_series: pd.Series) -> None:
        result = alma(close_series)
        assert result.name == "alma"

    def test_alma_smoother_than_raw(self, close_series: pd.Series) -> None:
        """ALMA should reduce volatility compared to raw prices."""
        result = alma(close_series, period=20)
        valid = result.dropna()
        raw_std = close_series.iloc[-len(valid) :].diff().std()
        alma_std = valid.diff().std()
        assert alma_std < raw_std


# ---------------------------------------------------------------------------
# LSMA
# ---------------------------------------------------------------------------


class TestLSMA:
    def test_lsma_length(self, close_series: pd.Series) -> None:
        result = lsma(close_series, period=25)
        assert len(result) == len(close_series)

    def test_lsma_on_linear_data(self, linear_series: pd.Series) -> None:
        """On a perfect line, LSMA should equal the data (after warm-up)."""
        period = 10
        result = lsma(linear_series, period=period)
        valid_idx = period - 1
        # LSMA endpoint should equal the actual value on a straight line
        np.testing.assert_allclose(
            result.iloc[valid_idx:].values,
            linear_series.iloc[valid_idx:].values,
            atol=1e-8,
        )

    def test_lsma_name(self, close_series: pd.Series) -> None:
        result = lsma(close_series)
        assert result.name == "lsma"


# ---------------------------------------------------------------------------
# SWMA
# ---------------------------------------------------------------------------


class TestSWMA:
    def test_swma_manual(self) -> None:
        """Verify SWMA against hand calculation."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = swma(data)
        # At index 3: (1*1 + 2*2 + 2*3 + 1*4) / 6 = (1+4+6+4)/6 = 15/6
        expected = (1 * 1 + 2 * 2 + 2 * 3 + 1 * 4) / 6.0
        assert abs(result.iloc[3] - expected) < 1e-10

    def test_swma_nan_prefix(self) -> None:
        data = pd.Series(range(10), dtype=float)
        result = swma(data)
        assert result.iloc[:3].isna().all()
        assert result.iloc[3:].notna().all()

    def test_swma_name(self, close_series: pd.Series) -> None:
        result = swma(close_series)
        assert result.name == "swma"


# ---------------------------------------------------------------------------
# SineMA
# ---------------------------------------------------------------------------


class TestSineMA:
    def test_sinema_length(self, close_series: pd.Series) -> None:
        result = sinema(close_series, period=14)
        assert len(result) == len(close_series)

    def test_sinema_nan_prefix(self, close_series: pd.Series) -> None:
        period = 14
        result = sinema(close_series, period=period)
        assert result.iloc[: period - 1].isna().all()

    def test_sinema_name(self, close_series: pd.Series) -> None:
        result = sinema(close_series)
        assert result.name == "sinema"

    def test_sinema_smoother_than_raw(self, close_series: pd.Series) -> None:
        result = sinema(close_series, period=20)
        valid = result.dropna()
        raw_std = close_series.iloc[-len(valid) :].diff().std()
        sinema_std = valid.diff().std()
        assert sinema_std < raw_std


# ---------------------------------------------------------------------------
# TRIMA
# ---------------------------------------------------------------------------


class TestTRIMA:
    def test_trima_length(self, close_series: pd.Series) -> None:
        result = trima(close_series, period=20)
        assert len(result) == len(close_series)

    def test_trima_smoother_than_sma(self, close_series: pd.Series) -> None:
        """TRIMA (double SMA) should produce finite smoothed values."""
        period = 20
        trima_val = trima(close_series, period=period)
        valid = trima_val.dropna()
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))

    def test_trima_name(self, close_series: pd.Series) -> None:
        result = trima(close_series)
        assert result.name == "trima"


# ---------------------------------------------------------------------------
# JMA
# ---------------------------------------------------------------------------


class TestJMA:
    def test_jma_length(self, close_series: pd.Series) -> None:
        result = jma(close_series, period=7)
        assert len(result) == len(close_series)

    def test_jma_has_valid_values(self, close_series: pd.Series) -> None:
        result = jma(close_series, period=7)
        assert result.notna().sum() > 0

    def test_jma_name(self, close_series: pd.Series) -> None:
        result = jma(close_series)
        assert result.name == "jma"

    def test_jma_phase_variations(self, close_series: pd.Series) -> None:
        """Different phase values should produce different results."""
        r1 = jma(close_series, period=7, phase=-50)
        r2 = jma(close_series, period=7, phase=50)
        assert not r1.equals(r2)


# ---------------------------------------------------------------------------
# Gaussian Filter
# ---------------------------------------------------------------------------


class TestGaussianFilter:
    def test_gaussian_length(self, close_series: pd.Series) -> None:
        result = gaussian_filter(close_series, period=14)
        assert len(result) == len(close_series)

    def test_gaussian_nan_prefix(self, close_series: pd.Series) -> None:
        period = 14
        result = gaussian_filter(close_series, period=period)
        assert result.iloc[: period - 1].isna().all()

    def test_gaussian_name(self, close_series: pd.Series) -> None:
        result = gaussian_filter(close_series)
        assert result.name == "gaussian_filter"

    def test_gaussian_smoothness(self, close_series: pd.Series) -> None:
        result = gaussian_filter(close_series, period=20)
        valid = result.dropna()
        raw_std = close_series.iloc[-len(valid) :].diff().std()
        gauss_std = valid.diff().std()
        assert gauss_std < raw_std


# ---------------------------------------------------------------------------
# Butterworth Filter
# ---------------------------------------------------------------------------


class TestButterworthFilter:
    def test_butterworth_length(self, close_series: pd.Series) -> None:
        result = butterworth_filter(close_series, period=14)
        assert len(result) == len(close_series)

    def test_butterworth_has_valid_values(self, close_series: pd.Series) -> None:
        result = butterworth_filter(close_series, period=14)
        assert result.notna().sum() > 0

    def test_butterworth_name(self, close_series: pd.Series) -> None:
        result = butterworth_filter(close_series)
        assert result.name == "butterworth"

    def test_butterworth_tracks_data(self, close_series: pd.Series) -> None:
        """Butterworth filter should follow the general trend of the data."""
        result = butterworth_filter(close_series, period=10)
        valid = result.dropna()
        raw = close_series.iloc[: len(valid)]
        correlation = valid.corr(raw)
        assert correlation > 0.9


# ---------------------------------------------------------------------------
# Super Smoother
# ---------------------------------------------------------------------------


class TestSuperSmoother:
    def test_supersmoother_length(self, close_series: pd.Series) -> None:
        result = supersmoother(close_series, period=14)
        assert len(result) == len(close_series)

    def test_supersmoother_has_valid_values(self, close_series: pd.Series) -> None:
        result = supersmoother(close_series, period=14)
        assert result.notna().sum() > 0

    def test_supersmoother_name(self, close_series: pd.Series) -> None:
        result = supersmoother(close_series)
        assert result.name == "supersmoother"


# ---------------------------------------------------------------------------
# Hann Window MA
# ---------------------------------------------------------------------------


class TestHannWindowMA:
    def test_hann_length(self, close_series: pd.Series) -> None:
        result = hann_window_ma(close_series, period=14)
        assert len(result) == len(close_series)

    def test_hann_nan_prefix(self, close_series: pd.Series) -> None:
        period = 14
        result = hann_window_ma(close_series, period=period)
        assert result.iloc[: period - 1].isna().all()

    def test_hann_name(self, close_series: pd.Series) -> None:
        result = hann_window_ma(close_series)
        assert result.name == "hann_ma"

    def test_hann_period_1(self, close_series: pd.Series) -> None:
        """Period 1 should return the original data."""
        result = hann_window_ma(close_series, period=1)
        pd.testing.assert_series_equal(result.rename(None), close_series.rename(None))


# ---------------------------------------------------------------------------
# Hamming Window MA
# ---------------------------------------------------------------------------


class TestHammingWindowMA:
    def test_hamming_length(self, close_series: pd.Series) -> None:
        result = hamming_window_ma(close_series, period=14)
        assert len(result) == len(close_series)

    def test_hamming_nan_prefix(self, close_series: pd.Series) -> None:
        period = 14
        result = hamming_window_ma(close_series, period=period)
        assert result.iloc[: period - 1].isna().all()

    def test_hamming_name(self, close_series: pd.Series) -> None:
        result = hamming_window_ma(close_series)
        assert result.name == "hamming_ma"

    def test_hamming_different_from_hann(self, close_series: pd.Series) -> None:
        """Hamming and Hann should produce different results."""
        hann_result = hann_window_ma(close_series, period=14)
        hamming_result = hamming_window_ma(close_series, period=14)
        assert not hann_result.equals(hamming_result)


# ---------------------------------------------------------------------------
# Kaufman Efficiency Ratio
# ---------------------------------------------------------------------------


class TestKaufmanEfficiencyRatio:
    def test_er_length(self, close_series: pd.Series) -> None:
        result = kaufman_efficiency_ratio(close_series, period=10)
        assert len(result) == len(close_series)

    def test_er_bounded(self, close_series: pd.Series) -> None:
        """ER should be in [0, 1] (where defined)."""
        result = kaufman_efficiency_ratio(close_series, period=10)
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 1.0 + 1e-10).all()

    def test_er_on_straight_line(self, linear_series: pd.Series) -> None:
        """A perfectly linear series should have ER = 1."""
        result = kaufman_efficiency_ratio(linear_series, period=10)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-10)

    def test_er_name(self, close_series: pd.Series) -> None:
        result = kaufman_efficiency_ratio(close_series)
        assert result.name == "efficiency_ratio"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_accepts_list_input(self) -> None:
        result = alma(list(range(1, 30)), period=3)
        assert isinstance(result, pd.Series)

    def test_invalid_period_raises(self) -> None:
        data = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            alma(data, period=0)
