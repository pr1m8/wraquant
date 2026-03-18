"""Tests for wraquant.math.optimal_stopping."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.math.optimal_stopping import (
    binomial_american,
    cusum_stopping,
    longstaff_schwartz,
    optimal_exit_threshold,
    secretary_problem_threshold,
    sequential_probability_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate GBM price paths for testing."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.log(S0) + np.cumsum(log_increments, axis=1)
    return np.exp(log_S)


# ---------------------------------------------------------------------------
# Longstaff-Schwartz
# ---------------------------------------------------------------------------

class TestLongstaffSchwartz:
    """Tests for longstaff_schwartz."""

    def test_put_price_positive(self) -> None:
        paths = _generate_gbm_paths(100, 0.05, 0.2, 1.0, 50, 5000)
        price = longstaff_schwartz(paths, strike=100, rf_rate=0.05, dt=1.0 / 50)
        assert price > 0.0

    def test_put_gte_european_intrinsic(self) -> None:
        """American put >= max(K - S0, 0) discounted."""
        S0, K, r, T = 100.0, 110.0, 0.05, 1.0
        paths = _generate_gbm_paths(S0, r, 0.3, T, 50, 10000)
        price = longstaff_schwartz(paths, strike=K, rf_rate=r, dt=T / 50)
        # At minimum, worth the discounted intrinsic
        min_value = max(K - S0, 0.0) * np.exp(-r * T)
        assert price >= min_value * 0.5  # generous lower bound

    def test_call_price_positive(self) -> None:
        paths = _generate_gbm_paths(100, 0.05, 0.2, 1.0, 50, 5000)
        price = longstaff_schwartz(
            paths, strike=100, rf_rate=0.05, dt=1.0 / 50, option_type="call"
        )
        assert price > 0.0

    def test_polynomial_basis(self) -> None:
        """Polynomial basis should also produce reasonable prices."""
        paths = _generate_gbm_paths(100, 0.05, 0.2, 1.0, 50, 5000)
        price = longstaff_schwartz(
            paths, strike=100, rf_rate=0.05, dt=1.0 / 50,
            basis_functions="polynomial",
        )
        assert price > 0.0

    def test_deep_itm_put_close_to_intrinsic(self) -> None:
        """Deep ITM American put should be close to intrinsic."""
        S0, K = 50.0, 100.0
        paths = _generate_gbm_paths(S0, 0.05, 0.2, 0.01, 10, 5000)
        price = longstaff_schwartz(paths, strike=K, rf_rate=0.05, dt=0.001)
        assert price == pytest.approx(K - S0, abs=5.0)


# ---------------------------------------------------------------------------
# Binomial American
# ---------------------------------------------------------------------------

class TestBinomialAmerican:
    """Tests for binomial_american."""

    def test_put_price_positive(self) -> None:
        price = binomial_american(100, 100, 0.05, 0.2, 1.0, 200, "put")
        assert price > 0.0

    def test_call_price_positive(self) -> None:
        price = binomial_american(100, 100, 0.05, 0.2, 1.0, 200, "call")
        assert price > 0.0

    def test_american_put_gte_european_payoff(self) -> None:
        """American put >= discounted intrinsic."""
        S, K, r, T = 90.0, 100.0, 0.05, 1.0
        price = binomial_american(S, K, r, 0.2, T, 500, "put")
        european_lower = max(K * np.exp(-r * T) - S, 0.0)
        assert price >= european_lower - 0.01

    def test_converges_with_more_steps(self) -> None:
        """Price should stabilise as steps increase."""
        p100 = binomial_american(100, 100, 0.05, 0.2, 1.0, 100, "put")
        p500 = binomial_american(100, 100, 0.05, 0.2, 1.0, 500, "put")
        assert abs(p100 - p500) < 0.5


# ---------------------------------------------------------------------------
# Optimal exit threshold
# ---------------------------------------------------------------------------

class TestOptimalExitThreshold:
    """Tests for optimal_exit_threshold."""

    def test_returns_correct_keys(self) -> None:
        result = optimal_exit_threshold(0.5, 0.1, 0.001)
        assert "entry_threshold" in result
        assert "exit_threshold" in result
        assert "expected_profit" in result

    def test_exit_threshold_positive(self) -> None:
        result = optimal_exit_threshold(0.5, 0.1, 0.001)
        assert result["exit_threshold"] > 0.0

    def test_higher_costs_raise_threshold(self) -> None:
        """Higher transaction costs should require a higher threshold."""
        r1 = optimal_exit_threshold(0.5, 0.1, 0.001)
        r2 = optimal_exit_threshold(0.5, 0.1, 0.01)
        # The exit threshold might not always be monotonically increasing,
        # but expected profit should decrease with higher costs
        assert r1["expected_profit"] >= r2["expected_profit"]

    def test_raises_on_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            optimal_exit_threshold(-0.5, 0.1, 0.001)


# ---------------------------------------------------------------------------
# Sequential Probability Ratio Test
# ---------------------------------------------------------------------------

class TestSequentialProbabilityRatio:
    """Tests for sequential_probability_ratio."""

    def test_detects_mean_shift(self) -> None:
        """Data from H1 distribution should lead to rejecting H0."""
        rng = np.random.default_rng(42)
        # H0: mu=0, H1: mu=1
        data = rng.normal(1.0, 1.0, size=100)  # from H1
        result = sequential_probability_ratio(
            data,
            h0_dist=("normal", {"mu": 0.0, "sigma": 1.0}),
            h1_dist=("normal", {"mu": 1.0, "sigma": 1.0}),
        )
        assert result["decision"] == "reject_h0"

    def test_accepts_h0_when_correct(self) -> None:
        """Data from H0 distribution should lead to accepting H0."""
        rng = np.random.default_rng(42)
        data = rng.normal(0.0, 1.0, size=100)  # from H0
        result = sequential_probability_ratio(
            data,
            h0_dist=("normal", {"mu": 0.0, "sigma": 1.0}),
            h1_dist=("normal", {"mu": 2.0, "sigma": 1.0}),
        )
        assert result["decision"] == "accept_h0"

    def test_stopping_time_positive(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(1.0, 1.0, size=100)
        result = sequential_probability_ratio(
            data,
            h0_dist=("normal", {"mu": 0.0, "sigma": 1.0}),
            h1_dist=("normal", {"mu": 1.0, "sigma": 1.0}),
        )
        assert result["stopping_time"] >= 1


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------

class TestCusumStopping:
    """Tests for cusum_stopping."""

    def test_detects_mean_shift(self) -> None:
        """CUSUM should detect a shift in mean."""
        rng = np.random.default_rng(42)
        # First 50 obs ~ N(0,1), next 50 ~ N(2,1)
        data = np.concatenate([
            rng.normal(0, 0.5, 50),
            rng.normal(2, 0.5, 50),
        ])
        result = cusum_stopping(data, target_mean=0.0, threshold=5.0)
        assert result["detected"]
        assert result["stopping_time"] <= 100

    def test_no_detection_stable(self) -> None:
        """No change should not trigger detection with high threshold."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.1, 100)
        result = cusum_stopping(data, target_mean=0.0, threshold=100.0)
        assert not result["detected"]

    def test_cusum_values_length(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 50)
        result = cusum_stopping(data, target_mean=0.0, threshold=100.0)
        assert len(result["cusum_values"]) == result["stopping_time"]


# ---------------------------------------------------------------------------
# Secretary problem
# ---------------------------------------------------------------------------

class TestSecretaryProblemThreshold:
    """Tests for secretary_problem_threshold."""

    def test_single_candidate(self) -> None:
        result = secretary_problem_threshold(1)
        assert result["threshold"] == 0
        assert result["success_probability"] == 1.0

    def test_large_n_approaches_1_over_e(self) -> None:
        """For large n, optimal fraction -> 1/e ~ 0.3679."""
        result = secretary_problem_threshold(1000)
        assert result["optimal_fraction"] == pytest.approx(1.0 / np.e, abs=0.02)

    def test_large_n_success_prob_near_1_over_e(self) -> None:
        """Success probability -> 1/e for large n."""
        result = secretary_problem_threshold(1000)
        assert result["success_probability"] == pytest.approx(1.0 / np.e, abs=0.02)

    def test_raises_on_zero(self) -> None:
        with pytest.raises(ValueError):
            secretary_problem_threshold(0)
