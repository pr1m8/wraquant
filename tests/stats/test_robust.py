"""Tests for robust statistical methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.stats.robust import (
    huber_mean,
    mad,
    outlier_detection,
    robust_zscore,
    trimmed_mean,
    trimmed_std,
    winsorize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normal_data(n: int = 500, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, size=n)


def _contaminated_data(n: int = 500, seed: int = 42) -> np.ndarray:
    """Normal data with a few extreme outliers appended."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, size=n)
    outliers = np.array([50.0, -60.0, 100.0])
    return np.concatenate([base, outliers])


# ---------------------------------------------------------------------------
# mad
# ---------------------------------------------------------------------------


class TestMAD:
    def test_returns_float(self) -> None:
        result = mad(_normal_data())
        assert isinstance(result, float)

    def test_positive(self) -> None:
        result = mad(_normal_data())
        assert result > 0

    def test_close_to_std_for_normal(self) -> None:
        data = _normal_data(n=10000)
        result = mad(data, scale="normal")
        # For normal data, MAD (scaled) should be close to std
        assert abs(result - np.std(data, ddof=1)) < 0.15

    def test_handles_pandas_series(self) -> None:
        data = pd.Series(_normal_data())
        result = mad(data)
        assert isinstance(result, float)

    def test_robust_to_outliers(self) -> None:
        clean = _normal_data(n=1000)
        contaminated = _contaminated_data(n=1000)
        mad_clean = mad(clean)
        mad_contaminated = mad(contaminated)
        # MAD should be relatively stable despite outliers
        assert abs(mad_contaminated - mad_clean) / mad_clean < 0.1


# ---------------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------------


class TestWinsorize:
    def test_returns_same_length(self) -> None:
        data = _normal_data(100)
        result = winsorize(data)
        assert len(result) == len(data)

    def test_returns_series_for_series_input(self) -> None:
        data = pd.Series(_normal_data(100))
        result = winsorize(data)
        assert isinstance(result, pd.Series)

    def test_returns_array_for_array_input(self) -> None:
        data = _normal_data(100)
        result = winsorize(data)
        assert isinstance(result, np.ndarray)

    def test_extremes_capped(self) -> None:
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = winsorize(data, lower=0.1, upper=0.1)
        assert result.max() < 100
        # The max should be winsorized to the 90th percentile value
        assert result.max() <= data[np.argsort(data)[-2]]

    def test_preserves_index(self) -> None:
        idx = pd.date_range("2020-01-01", periods=50)
        data = pd.Series(_normal_data(50), index=idx)
        result = winsorize(data)
        assert list(result.index) == list(idx)


# ---------------------------------------------------------------------------
# trimmed_mean
# ---------------------------------------------------------------------------


class TestTrimmedMean:
    def test_returns_float(self) -> None:
        result = trimmed_mean(_normal_data())
        assert isinstance(result, float)

    def test_more_robust_than_mean(self) -> None:
        contaminated = _contaminated_data()
        regular_mean = float(np.mean(contaminated))
        t_mean = trimmed_mean(contaminated, proportiontocut=0.05)
        # Trimmed mean should be closer to 0 than regular mean
        assert abs(t_mean) < abs(regular_mean)

    def test_zero_trim_equals_mean(self) -> None:
        data = _normal_data(100)
        result = trimmed_mean(data, proportiontocut=0.0)
        np.testing.assert_allclose(result, np.mean(data), atol=1e-10)


# ---------------------------------------------------------------------------
# trimmed_std
# ---------------------------------------------------------------------------


class TestTrimmedStd:
    def test_returns_float(self) -> None:
        result = trimmed_std(_normal_data())
        assert isinstance(result, float)

    def test_positive(self) -> None:
        result = trimmed_std(_normal_data())
        assert result > 0

    def test_less_than_contaminated_std(self) -> None:
        contaminated = _contaminated_data()
        t_std = trimmed_std(contaminated, proportiontocut=0.05)
        regular_std = float(np.std(contaminated, ddof=1))
        # Trimmed std should be smaller when outliers are removed
        assert t_std < regular_std


# ---------------------------------------------------------------------------
# robust_zscore
# ---------------------------------------------------------------------------


class TestRobustZscore:
    def test_returns_series(self) -> None:
        data = pd.Series(_normal_data())
        result = robust_zscore(data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_center_near_zero(self) -> None:
        data = _normal_data(5000)
        result = robust_zscore(data)
        # Median of robust z-scores should be near 0
        assert abs(np.median(result)) < 0.01

    def test_outlier_flagged(self) -> None:
        data = np.array([1.0, 2.0, 1.5, 1.8, 2.2, 50.0])
        result = robust_zscore(data)
        # The extreme value (50.0) should have the largest z-score
        assert np.argmax(np.abs(result)) == 5

    def test_accepts_numpy_array(self) -> None:
        data = _normal_data()
        result = robust_zscore(data)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# huber_mean
# ---------------------------------------------------------------------------


class TestHuberMean:
    def test_returns_float(self) -> None:
        result = huber_mean(_normal_data())
        assert isinstance(result, float)

    def test_close_to_median_for_contaminated(self) -> None:
        contaminated = _contaminated_data()
        h_mean = huber_mean(contaminated)
        median = float(np.median(contaminated))
        regular_mean = float(np.mean(contaminated))
        # Huber mean should be closer to median than to regular mean
        assert abs(h_mean - median) < abs(regular_mean - median)

    def test_close_to_mean_for_clean_data(self) -> None:
        data = _normal_data(1000)
        h_mean = huber_mean(data)
        regular_mean = float(np.mean(data))
        assert abs(h_mean - regular_mean) < 0.1


# ---------------------------------------------------------------------------
# outlier_detection
# ---------------------------------------------------------------------------


class TestOutlierDetection:
    def test_mad_method(self) -> None:
        contaminated = _contaminated_data()
        result = outlier_detection(contaminated, method="mad")
        assert "outliers" in result
        assert "n_outliers" in result
        assert "method" in result
        assert result["method"] == "mad"
        # Should detect the injected outliers
        assert result["n_outliers"] >= 2

    def test_iqr_method(self) -> None:
        contaminated = _contaminated_data()
        result = outlier_detection(contaminated, method="iqr")
        assert result["method"] == "iqr"
        assert result["n_outliers"] >= 2

    def test_grubbs_method(self) -> None:
        contaminated = _contaminated_data()
        result = outlier_detection(contaminated, method="grubbs")
        assert result["method"] == "grubbs"
        assert result["n_outliers"] >= 1

    def test_unknown_method_raises(self) -> None:
        data = _normal_data()
        with pytest.raises(ValueError, match="Unknown outlier detection"):
            outlier_detection(data, method="bogus")

    def test_outlier_array_is_bool(self) -> None:
        data = _normal_data()
        result = outlier_detection(data, method="mad")
        assert result["outliers"].dtype == bool

    def test_clean_data_few_outliers(self) -> None:
        data = _normal_data(1000)
        result = outlier_detection(data, method="mad", threshold=3.0)
        # For normal data with threshold=3, should flag very few
        assert result["n_outliers"] < 20

    def test_handles_pandas_series(self) -> None:
        data = pd.Series(_contaminated_data())
        result = outlier_detection(data, method="mad")
        assert result["n_outliers"] >= 2
