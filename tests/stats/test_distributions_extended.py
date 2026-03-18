"""Tests for extended distribution functions (tail_index, qqplot_data, jarque_bera, kolmogorov_smirnov)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.stats.distributions import (
    jarque_bera,
    kolmogorov_smirnov,
    qqplot_data,
    tail_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normal_data(n: int = 1000, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, size=n)


def _heavy_tailed_data(n: int = 1000, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).standard_t(df=3, size=n)


# ---------------------------------------------------------------------------
# tail_index
# ---------------------------------------------------------------------------


class TestTailIndex:
    def test_hill_returns_dict(self) -> None:
        data = _heavy_tailed_data()
        result = tail_index(data, method="hill")
        assert "tail_index" in result
        assert "method" in result
        assert "n_tail" in result
        assert result["method"] == "hill"

    def test_pickands_estimator(self) -> None:
        data = _heavy_tailed_data()
        result = tail_index(data, method="pickands")
        assert result["method"] == "pickands"
        assert np.isfinite(result["tail_index"])

    def test_moment_estimator(self) -> None:
        data = _heavy_tailed_data()
        result = tail_index(data, method="moment")
        assert result["method"] == "moment"
        assert np.isfinite(result["tail_index"])

    def test_unknown_method_raises(self) -> None:
        data = _normal_data()
        with pytest.raises(ValueError, match="Unknown method"):
            tail_index(data, method="bogus")

    def test_tail_index_positive_for_heavy_tails(self) -> None:
        data = _heavy_tailed_data(n=5000)
        result = tail_index(data, method="hill")
        assert result["tail_index"] > 0

    def test_handles_pandas_series(self) -> None:
        data = pd.Series(_normal_data())
        result = tail_index(data, method="hill")
        assert np.isfinite(result["tail_index"])


# ---------------------------------------------------------------------------
# qqplot_data
# ---------------------------------------------------------------------------


class TestQQPlotData:
    def test_output_keys(self) -> None:
        data = _normal_data()
        result = qqplot_data(data)
        assert "theoretical_quantiles" in result
        assert "sample_quantiles" in result
        assert "slope" in result
        assert "intercept" in result

    def test_lengths_match(self) -> None:
        data = _normal_data(n=200)
        result = qqplot_data(data)
        assert len(result["theoretical_quantiles"]) == 200
        assert len(result["sample_quantiles"]) == 200

    def test_sample_quantiles_sorted(self) -> None:
        data = _normal_data()
        result = qqplot_data(data)
        sq = result["sample_quantiles"]
        assert np.all(sq[:-1] <= sq[1:])

    def test_slope_near_one_for_normal(self) -> None:
        data = _normal_data(n=5000)
        result = qqplot_data(data, dist="norm")
        # For normal data against normal theoretical, slope ~ 1
        assert abs(result["slope"] - 1.0) < 0.15

    def test_t_distribution(self) -> None:
        data = _heavy_tailed_data()
        result = qqplot_data(data, dist="t")
        assert len(result["theoretical_quantiles"]) == len(data)


# ---------------------------------------------------------------------------
# jarque_bera
# ---------------------------------------------------------------------------


class TestJarqueBera:
    def test_output_keys(self) -> None:
        data = _normal_data()
        result = jarque_bera(data)
        assert "statistic" in result
        assert "p_value" in result
        assert "skewness" in result
        assert "kurtosis" in result

    def test_normal_data_high_pvalue(self) -> None:
        # Normal data should not reject H0 of normality
        data = _normal_data(n=5000)
        result = jarque_bera(data)
        assert result["p_value"] > 0.01

    def test_skewed_data_low_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.exponential(1.0, size=5000)
        result = jarque_bera(data)
        assert result["p_value"] < 0.05

    def test_skewness_sign(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.exponential(1.0, size=5000)
        result = jarque_bera(data)
        assert result["skewness"] > 0  # exponential is right-skewed

    def test_handles_pandas(self) -> None:
        data = pd.Series(_normal_data())
        result = jarque_bera(data)
        assert "statistic" in result


# ---------------------------------------------------------------------------
# kolmogorov_smirnov
# ---------------------------------------------------------------------------


class TestKolmogorovSmirnov:
    def test_output_keys(self) -> None:
        data = _normal_data()
        result = kolmogorov_smirnov(data)
        assert "statistic" in result
        assert "p_value" in result
        assert "dist" in result
        assert "params" in result

    def test_normal_data_high_pvalue(self) -> None:
        data = _normal_data(n=500)
        result = kolmogorov_smirnov(data, dist="norm")
        assert result["p_value"] > 0.01

    def test_wrong_distribution_low_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.exponential(1.0, size=500)
        result = kolmogorov_smirnov(data, dist="norm")
        # Exponential data should not fit normal well
        assert result["statistic"] > 0

    def test_t_distribution(self) -> None:
        data = _heavy_tailed_data()
        result = kolmogorov_smirnov(data, dist="t")
        assert result["dist"] == "t"
        assert result["p_value"] >= 0

    def test_handles_nan(self) -> None:
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
        result = kolmogorov_smirnov(data)
        assert np.isfinite(result["statistic"])
