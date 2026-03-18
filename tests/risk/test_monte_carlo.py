"""Tests for advanced Monte Carlo methods."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.risk.monte_carlo import (
    antithetic_variates,
    block_bootstrap,
    filtered_historical_simulation,
    importance_sampling_var,
    stationary_bootstrap,
    stratified_sampling,
)


def _make_returns(n: int = 500, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.02, size=n)


# ---------------------------------------------------------------------------
# Importance sampling VaR
# ---------------------------------------------------------------------------

class TestImportanceSamplingVar:
    def test_returns_positive_var(self) -> None:
        ret = _make_returns()
        result = importance_sampling_var(ret, n_sims=5000, seed=42)
        assert result["var"] > 0

    def test_effective_sample_size_positive(self) -> None:
        ret = _make_returns()
        result = importance_sampling_var(ret, n_sims=5000, seed=42)
        assert result["effective_sample_size"] > 0

    def test_var_reasonable(self) -> None:
        ret = _make_returns(n=1000)
        result = importance_sampling_var(ret, n_sims=10000, target_quantile=0.05, seed=42)
        # Should be roughly consistent with empirical quantile
        empirical_var = -np.quantile(ret, 0.05)
        assert abs(result["var"] - empirical_var) / empirical_var < 1.0


# ---------------------------------------------------------------------------
# Antithetic variates
# ---------------------------------------------------------------------------

class TestAntitheticVariates:
    def test_shape(self) -> None:
        result = antithetic_variates(0.0, 1.0, n_sims=100, n_assets=3, seed=42)
        assert result.shape == (200, 3)

    def test_mean_close_to_mu(self) -> None:
        mu = 0.05
        result = antithetic_variates(mu, 0.1, n_sims=10000, n_assets=1, seed=42)
        assert np.mean(result) == pytest.approx(mu, abs=0.01)

    def test_variance_reduction(self) -> None:
        """Antithetic variates should reduce variance of mean estimate."""
        rng = np.random.default_rng(42)
        mu, sigma = 0.0, 1.0
        n = 1000

        # Regular samples
        regular = rng.normal(mu, sigma, size=(2 * n, 1))
        var_regular = np.var(np.mean(regular.reshape(-1, 2), axis=1))

        # Antithetic samples
        av = antithetic_variates(mu, sigma, n_sims=n, n_assets=1, seed=42)
        pairs = av.reshape(-1, 2)
        var_antithetic = np.var(np.mean(pairs, axis=1))

        # Antithetic variance should be smaller
        assert var_antithetic < var_regular


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

class TestStratifiedSampling:
    def test_output_length(self) -> None:
        ret = _make_returns()
        result = stratified_sampling(ret, n_strata=10, n_sims=1000, seed=42)
        assert len(result) == 1000

    def test_mean_close_to_data(self) -> None:
        ret = _make_returns(n=1000)
        result = stratified_sampling(ret, n_strata=20, n_sims=5000, seed=42)
        assert np.mean(result) == pytest.approx(np.mean(ret), abs=0.005)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

class TestBlockBootstrap:
    def test_shape(self) -> None:
        ret = _make_returns(n=100)
        result = block_bootstrap(ret, block_size=10, n_sims=50, seed=42)
        assert result.shape == (50, 100)

    def test_values_from_original(self) -> None:
        ret = _make_returns(n=100)
        result = block_bootstrap(ret, block_size=5, n_sims=10, seed=42)
        # All bootstrap values should come from the original series
        for val in result.ravel():
            assert val in ret

    def test_invalid_block_size(self) -> None:
        ret = _make_returns(n=50)
        with pytest.raises(ValueError, match="block_size"):
            block_bootstrap(ret, block_size=0, n_sims=10)


# ---------------------------------------------------------------------------
# Stationary bootstrap
# ---------------------------------------------------------------------------

class TestStationaryBootstrap:
    def test_shape(self) -> None:
        ret = _make_returns(n=100)
        result = stationary_bootstrap(ret, avg_block_size=10, n_sims=50, seed=42)
        assert result.shape == (50, 100)

    def test_values_from_original(self) -> None:
        ret = _make_returns(n=100)
        result = stationary_bootstrap(ret, avg_block_size=5, n_sims=10, seed=42)
        original_set = set(ret)
        for val in result.ravel():
            assert val in original_set

    def test_invalid_avg_block_size(self) -> None:
        ret = _make_returns(n=50)
        with pytest.raises(ValueError, match="avg_block_size"):
            stationary_bootstrap(ret, avg_block_size=0.5, n_sims=10)


# ---------------------------------------------------------------------------
# Filtered historical simulation
# ---------------------------------------------------------------------------

class TestFilteredHistoricalSimulation:
    def test_ewma_shape(self) -> None:
        ret = _make_returns(n=200)
        result = filtered_historical_simulation(ret, vol_model="ewma", n_sims=500, seed=42)
        assert result["simulated_returns"].shape == (500,)

    def test_garch_shape(self) -> None:
        ret = _make_returns(n=200)
        result = filtered_historical_simulation(ret, vol_model="garch", n_sims=500, seed=42)
        assert result["simulated_returns"].shape == (500,)

    def test_current_vol_positive(self) -> None:
        ret = _make_returns(n=200)
        result = filtered_historical_simulation(ret, seed=42)
        assert result["current_vol"] > 0

    def test_residuals_length(self) -> None:
        ret = _make_returns(n=200)
        result = filtered_historical_simulation(ret, seed=42)
        assert len(result["standardised_residuals"]) == 200

    def test_unknown_model_raises(self) -> None:
        ret = _make_returns()
        with pytest.raises(ValueError, match="vol_model"):
            filtered_historical_simulation(ret, vol_model="unknown")
