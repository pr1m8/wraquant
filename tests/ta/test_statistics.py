"""Tests for wraquant.ta.statistics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.statistics import (
    beta,
    correlation,
    entropy,
    hurst_exponent,
    information_coefficient,
    kurtosis,
    mean_deviation,
    median,
    percentile_rank,
    r_squared,
    skewness,
    zscore,
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
def benchmark_series() -> pd.Series:
    np.random.seed(99)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    return pd.Series(prices, name="benchmark")


# ---------------------------------------------------------------------------
# Z-Score
# ---------------------------------------------------------------------------


class TestZScore:
    def test_output_length(self, close_series: pd.Series) -> None:
        result = zscore(close_series, period=20)
        assert len(result) == len(close_series)

    def test_output_type(self, close_series: pd.Series) -> None:
        result = zscore(close_series, period=20)
        assert isinstance(result, pd.Series)

    def test_nan_prefix(self, close_series: pd.Series) -> None:
        period = 20
        result = zscore(close_series, period=period)
        assert result.iloc[: period - 1].isna().all()

    def test_mean_near_zero(self) -> None:
        """Z-scores of a constant series should be NaN (zero std)."""
        data = pd.Series([5.0] * 30)
        result = zscore(data, period=10)
        valid = result.dropna()
        assert valid.isna().all() or (valid.abs() < 1e-10).all()

    def test_accepts_list_input(self) -> None:
        result = zscore(list(range(1, 30)))
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Percentile Rank
# ---------------------------------------------------------------------------


class TestPercentileRank:
    def test_bounds(self, close_series: pd.Series) -> None:
        result = percentile_rank(close_series, period=20)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_output_length(self, close_series: pd.Series) -> None:
        result = percentile_rank(close_series, period=20)
        assert len(result) == len(close_series)

    def test_max_value_is_100(self) -> None:
        """When the last value is the maximum, percentile rank should be 100."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = percentile_rank(data, period=5)
        assert abs(result.iloc[-1] - 100.0) < 1e-10


# ---------------------------------------------------------------------------
# Mean Deviation
# ---------------------------------------------------------------------------


class TestMeanDeviation:
    def test_non_negative(self, close_series: pd.Series) -> None:
        result = mean_deviation(close_series, period=20)
        valid = result.dropna()
        assert (valid >= -1e-10).all()

    def test_output_length(self, close_series: pd.Series) -> None:
        result = mean_deviation(close_series, period=20)
        assert len(result) == len(close_series)

    def test_constant_series(self) -> None:
        data = pd.Series([5.0] * 30)
        result = mean_deviation(data, period=10)
        valid = result.dropna()
        assert (valid.abs() < 1e-10).all()


# ---------------------------------------------------------------------------
# Median
# ---------------------------------------------------------------------------


class TestMedian:
    def test_output_length(self, close_series: pd.Series) -> None:
        result = median(close_series, period=20)
        assert len(result) == len(close_series)

    def test_nan_prefix(self, close_series: pd.Series) -> None:
        period = 20
        result = median(close_series, period=period)
        assert result.iloc[: period - 1].isna().all()

    def test_known_value(self) -> None:
        data = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
        result = median(data, period=3)
        # At index 2: median([1, 3, 2]) = 2
        assert abs(result.iloc[2] - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Skewness
# ---------------------------------------------------------------------------


class TestSkewness:
    def test_output_length(self, close_series: pd.Series) -> None:
        result = skewness(close_series, period=20)
        assert len(result) == len(close_series)

    def test_output_type(self, close_series: pd.Series) -> None:
        result = skewness(close_series, period=20)
        assert isinstance(result, pd.Series)

    def test_symmetric_distribution(self) -> None:
        """Symmetric data should have skewness near 0."""
        data = pd.Series(np.arange(50, dtype=float))
        result = skewness(data, period=20)
        valid = result.dropna()
        assert (valid.abs() < 0.5).all()


# ---------------------------------------------------------------------------
# Kurtosis
# ---------------------------------------------------------------------------


class TestKurtosis:
    def test_output_length(self, close_series: pd.Series) -> None:
        result = kurtosis(close_series, period=20)
        assert len(result) == len(close_series)

    def test_output_type(self, close_series: pd.Series) -> None:
        result = kurtosis(close_series, period=20)
        assert isinstance(result, pd.Series)

    def test_uniform_distribution_negative_kurtosis(self) -> None:
        """Uniform distribution has negative excess kurtosis."""
        data = pd.Series(np.arange(50, dtype=float))
        result = kurtosis(data, period=20)
        valid = result.dropna()
        assert (valid < 0).all()


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


class TestEntropy:
    def test_non_negative(self, close_series: pd.Series) -> None:
        result = entropy(close_series, period=20, bins=10)
        valid = result.dropna()
        assert (valid >= -1e-10).all()

    def test_output_length(self, close_series: pd.Series) -> None:
        result = entropy(close_series, period=20)
        assert len(result) == len(close_series)

    def test_constant_series_zero_entropy(self) -> None:
        """Constant price changes should have zero entropy."""
        data = pd.Series(np.arange(30, dtype=float))  # constant diff = 1
        result = entropy(data, period=20, bins=5)
        valid = result.dropna()
        # All changes are 1.0, so they all fall in one bin -> entropy should be 0
        assert (valid < 1e-10).all()


# ---------------------------------------------------------------------------
# Hurst Exponent
# ---------------------------------------------------------------------------


class TestHurstExponent:
    def test_output_length(self, close_series: pd.Series) -> None:
        result = hurst_exponent(close_series, period=100)
        assert len(result) == len(close_series)

    def test_output_type(self, close_series: pd.Series) -> None:
        result = hurst_exponent(close_series, period=100)
        assert isinstance(result, pd.Series)

    def test_valid_range(self, close_series: pd.Series) -> None:
        """Hurst exponent should be roughly in [0, 1]."""
        result = hurst_exponent(close_series, period=100)
        valid = result.dropna()
        if len(valid) > 0:
            assert (valid > -0.5).all()
            assert (valid < 1.5).all()


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


class TestCorrelation:
    def test_perfect_correlation(self) -> None:
        x = pd.Series(np.arange(20, dtype=float))
        y = pd.Series(np.arange(20, dtype=float) * 2)
        result = correlation(x, y, period=10)
        valid = result.dropna()
        assert (valid > 0.99).all()

    def test_bounds(self, close_series: pd.Series, benchmark_series: pd.Series) -> None:
        result = correlation(close_series, benchmark_series, period=20)
        valid = result.dropna()
        assert (valid >= -1 - 1e-10).all()
        assert (valid <= 1 + 1e-10).all()

    def test_output_length(self, close_series: pd.Series, benchmark_series: pd.Series) -> None:
        result = correlation(close_series, benchmark_series, period=20)
        assert len(result) == len(close_series)


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


class TestBeta:
    def test_output_length(self, close_series: pd.Series, benchmark_series: pd.Series) -> None:
        result = beta(close_series, benchmark_series, period=60)
        assert len(result) == len(close_series)

    def test_self_beta_is_one(self) -> None:
        """Beta of a series against itself should be 1."""
        np.random.seed(42)
        data = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
        result = beta(data, data, period=30)
        valid = result.dropna()
        if len(valid) > 0:
            assert (abs(valid - 1.0) < 1e-6).all()


# ---------------------------------------------------------------------------
# R-Squared
# ---------------------------------------------------------------------------


class TestRSquared:
    def test_bounds(self, close_series: pd.Series, benchmark_series: pd.Series) -> None:
        result = r_squared(close_series, benchmark_series, period=60)
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 1 + 1e-10).all()

    def test_perfect_correlation(self) -> None:
        x = pd.Series(100 + np.arange(50, dtype=float))
        y = pd.Series(200 + np.arange(50, dtype=float) * 2)
        result = r_squared(x, y, period=20)
        valid = result.dropna()
        assert (valid > 0.99).all()

    def test_output_length(self, close_series: pd.Series, benchmark_series: pd.Series) -> None:
        result = r_squared(close_series, benchmark_series, period=60)
        assert len(result) == len(close_series)


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------


class TestInformationCoefficient:
    def test_perfect_rank_correlation(self) -> None:
        x = pd.Series(np.arange(20, dtype=float))
        y = pd.Series(np.arange(20, dtype=float))
        result = information_coefficient(x, y, period=10)
        valid = result.dropna()
        assert (valid > 0.99).all()

    def test_bounds(self, close_series: pd.Series, benchmark_series: pd.Series) -> None:
        result = information_coefficient(close_series, benchmark_series, period=20)
        valid = result.dropna()
        assert (valid >= -1 - 1e-10).all()
        assert (valid <= 1 + 1e-10).all()

    def test_output_length(self, close_series: pd.Series, benchmark_series: pd.Series) -> None:
        result = information_coefficient(close_series, benchmark_series, period=20)
        assert len(result) == len(close_series)
