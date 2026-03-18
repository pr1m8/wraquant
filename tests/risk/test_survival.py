"""Tests for survival analysis estimators."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.risk.survival import (
    cox_partial_likelihood,
    exponential_survival,
    hazard_rate,
    kaplan_meier,
    log_rank_test,
    median_survival_time,
    nelson_aalen,
    weibull_survival,
)


def _make_survival_data(
    n: int = 100, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic survival data with ~30% censoring."""
    rng = np.random.default_rng(seed)
    true_durations = rng.exponential(scale=10, size=n)
    censor_times = rng.uniform(5, 25, size=n)
    durations = np.minimum(true_durations, censor_times)
    event_observed = (true_durations <= censor_times).astype(int)
    return durations, event_observed


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------

class TestKaplanMeier:
    def test_returns_expected_keys(self) -> None:
        d, e = _make_survival_data()
        km = kaplan_meier(d, e)
        assert set(km.keys()) == {"timeline", "survival", "variance"}

    def test_survival_starts_at_one(self) -> None:
        d, e = _make_survival_data()
        km = kaplan_meier(d, e)
        # First event should still be close to 1
        assert km["survival"][0] <= 1.0

    def test_survival_non_increasing(self) -> None:
        d, e = _make_survival_data()
        km = kaplan_meier(d, e)
        diffs = np.diff(km["survival"])
        assert np.all(diffs <= 1e-12)

    def test_survival_between_0_and_1(self) -> None:
        d, e = _make_survival_data()
        km = kaplan_meier(d, e)
        assert np.all(km["survival"] >= 0)
        assert np.all(km["survival"] <= 1)

    def test_all_events(self) -> None:
        d = np.array([1, 2, 3, 4, 5], dtype=float)
        e = np.ones(5, dtype=int)
        km = kaplan_meier(d, e)
        # Last survival should be 0
        assert km["survival"][-1] == pytest.approx(0.0)

    def test_no_events(self) -> None:
        d = np.array([1, 2, 3], dtype=float)
        e = np.zeros(3, dtype=int)
        km = kaplan_meier(d, e)
        assert np.all(km["survival"] == 1.0)

    def test_variance_nonnegative(self) -> None:
        d, e = _make_survival_data()
        km = kaplan_meier(d, e)
        assert np.all(km["variance"] >= 0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            kaplan_meier(np.array([1, 2, 3]), np.array([1, 0]))


# ---------------------------------------------------------------------------
# Nelson-Aalen
# ---------------------------------------------------------------------------

class TestNelsonAalen:
    def test_returns_expected_keys(self) -> None:
        d, e = _make_survival_data()
        na = nelson_aalen(d, e)
        assert set(na.keys()) == {"timeline", "cumulative_hazard", "variance"}

    def test_cumulative_hazard_non_decreasing(self) -> None:
        d, e = _make_survival_data()
        na = nelson_aalen(d, e)
        diffs = np.diff(na["cumulative_hazard"])
        assert np.all(diffs >= -1e-12)

    def test_starts_nonnegative(self) -> None:
        d, e = _make_survival_data()
        na = nelson_aalen(d, e)
        assert na["cumulative_hazard"][0] >= 0


# ---------------------------------------------------------------------------
# Hazard rate
# ---------------------------------------------------------------------------

class TestHazardRate:
    def test_nonnegative(self) -> None:
        d, e = _make_survival_data()
        hr = hazard_rate(d, e)
        assert np.all(hr["hazard"] >= 0)

    def test_custom_bandwidth(self) -> None:
        d, e = _make_survival_data()
        hr = hazard_rate(d, e, bandwidth=2.0)
        assert len(hr["hazard"]) > 0


# ---------------------------------------------------------------------------
# Cox PH model
# ---------------------------------------------------------------------------

class TestCoxPartialLikelihood:
    def test_returns_expected_keys(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        d = rng.exponential(10, size=n)
        e = rng.binomial(1, 0.7, size=n)
        X = rng.standard_normal((n, 2))
        result = cox_partial_likelihood(d, e, X)
        assert "beta" in result
        assert "se" in result
        assert "log_partial_likelihood" in result

    def test_covariate_effect_direction(self) -> None:
        """A strong positive covariate should yield positive beta."""
        rng = np.random.default_rng(123)
        n = 200
        x = rng.standard_normal(n)
        # Higher x => shorter survival (positive hazard ratio)
        lam = np.exp(1.0 * x)
        d = rng.exponential(1.0 / lam)
        e = np.ones(n, dtype=int)
        result = cox_partial_likelihood(d, e, x.reshape(-1, 1))
        assert result["beta"][0] > 0

    def test_single_covariate(self) -> None:
        d, e = _make_survival_data(n=50)
        X = np.random.default_rng(7).standard_normal(50)
        result = cox_partial_likelihood(d, e, X)
        assert result["beta"].shape == (1,)


# ---------------------------------------------------------------------------
# Exponential survival
# ---------------------------------------------------------------------------

class TestExponentialSurvival:
    def test_at_zero(self) -> None:
        assert exponential_survival(0.1, 0.0) == pytest.approx(1.0)

    def test_decreasing(self) -> None:
        s1 = exponential_survival(0.1, 5.0)
        s2 = exponential_survival(0.1, 10.0)
        assert s2 < s1

    def test_array_input(self) -> None:
        t = np.array([0, 1, 5, 10])
        s = exponential_survival(0.5, t)
        assert s.shape == (4,)
        assert s[0] == pytest.approx(1.0)

    def test_negative_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_param"):
            exponential_survival(-0.1, 1.0)


# ---------------------------------------------------------------------------
# Weibull survival
# ---------------------------------------------------------------------------

class TestWeibullSurvival:
    def test_k_one_matches_exponential(self) -> None:
        lam = 5.0
        t = 3.0
        w = weibull_survival(lam, 1.0, t)
        e = exponential_survival(1.0 / lam, t)
        assert w == pytest.approx(e, rel=1e-6)

    def test_at_zero(self) -> None:
        assert weibull_survival(2.0, 1.5, 0.0) == pytest.approx(1.0)

    def test_decreasing(self) -> None:
        s1 = weibull_survival(5.0, 2.0, 2.0)
        s2 = weibull_survival(5.0, 2.0, 6.0)
        assert s2 < s1

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            weibull_survival(-1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            weibull_survival(1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Log-rank test
# ---------------------------------------------------------------------------

class TestLogRankTest:
    def test_same_distribution(self) -> None:
        rng = np.random.default_rng(42)
        d1 = rng.exponential(10, size=100)
        e1 = np.ones(100, dtype=int)
        d2 = rng.exponential(10, size=100)
        e2 = np.ones(100, dtype=int)
        result = log_rank_test(d1, e1, d2, e2)
        # Should not reject H0
        assert result["p_value"] > 0.05

    def test_different_distribution(self) -> None:
        rng = np.random.default_rng(42)
        d1 = rng.exponential(5, size=200)
        e1 = np.ones(200, dtype=int)
        d2 = rng.exponential(20, size=200)
        e2 = np.ones(200, dtype=int)
        result = log_rank_test(d1, e1, d2, e2)
        # Should reject H0
        assert result["p_value"] < 0.05

    def test_returns_expected_keys(self) -> None:
        d1 = np.array([1, 2, 3], dtype=float)
        e1 = np.ones(3, dtype=int)
        d2 = np.array([2, 4, 6], dtype=float)
        e2 = np.ones(3, dtype=int)
        result = log_rank_test(d1, e1, d2, e2)
        assert set(result.keys()) == {
            "test_statistic", "p_value", "observed1", "expected1",
        }


# ---------------------------------------------------------------------------
# Median survival time
# ---------------------------------------------------------------------------

class TestMedianSurvivalTime:
    def test_finite(self) -> None:
        d, e = _make_survival_data(n=200)
        med = median_survival_time(d, e)
        assert np.isfinite(med) or med == np.inf

    def test_all_censored(self) -> None:
        d = np.array([1, 2, 3], dtype=float)
        e = np.zeros(3, dtype=int)
        med = median_survival_time(d, e)
        assert med == np.inf

    def test_known_exponential(self) -> None:
        """For a large exponential sample, median ≈ ln(2) / lambda."""
        rng = np.random.default_rng(42)
        lam = 0.1
        d = rng.exponential(1.0 / lam, size=5000)
        e = np.ones(len(d), dtype=int)
        med = median_survival_time(d, e)
        expected = np.log(2) / lam
        assert med == pytest.approx(expected, rel=0.1)
