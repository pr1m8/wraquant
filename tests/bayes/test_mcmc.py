"""Tests for MCMC utilities (pure numpy/scipy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.bayes.mcmc import (
    convergence_diagnostics,
    gelman_rubin,
    gibbs_sampler,
    hamiltonian_monte_carlo,
    metropolis_hastings,
    nuts_diagnostic,
    slice_sampler,
    trace_summary,
)


# ---------------------------------------------------------------------------
# Metropolis-Hastings tests
# ---------------------------------------------------------------------------


class TestMetropolisHastings:
    def _normal_log_posterior(self, theta: np.ndarray) -> float:
        """Log posterior for N(3, 1)."""
        return float(-0.5 * (theta[0] - 3.0) ** 2)

    def test_recovers_normal_mean(self) -> None:
        result = metropolis_hastings(
            self._normal_log_posterior,
            initial=np.array([0.0]),
            n_samples=20_000,
            proposal_std=1.0,
            burn_in=5_000,
        )
        samples = result["samples"]
        assert abs(np.mean(samples) - 3.0) < 0.2

    def test_recovers_normal_std(self) -> None:
        result = metropolis_hastings(
            self._normal_log_posterior,
            initial=np.array([0.0]),
            n_samples=20_000,
            proposal_std=1.0,
            burn_in=5_000,
        )
        samples = result["samples"]
        assert abs(np.std(samples) - 1.0) < 0.2

    def test_acceptance_rate_reasonable(self) -> None:
        result = metropolis_hastings(
            self._normal_log_posterior,
            initial=np.array([0.0]),
            n_samples=10_000,
            proposal_std=1.0,
            burn_in=1_000,
        )
        # With proposal_std=1.0 targeting N(3,1), acceptance should be reasonable
        assert 0.1 < result["acceptance_rate"] < 0.9

    def test_output_structure(self) -> None:
        result = metropolis_hastings(
            self._normal_log_posterior,
            initial=np.array([0.0]),
            n_samples=1_000,
            burn_in=100,
        )
        assert "samples" in result
        assert "acceptance_rate" in result
        assert "log_posteriors" in result
        assert result["samples"].ndim == 2

    def test_multivariate(self) -> None:
        def log_post(theta: np.ndarray) -> float:
            return float(-0.5 * np.sum((theta - np.array([1.0, 2.0])) ** 2))

        result = metropolis_hastings(
            log_post,
            initial=np.array([0.0, 0.0]),
            n_samples=20_000,
            proposal_std=np.array([1.0, 1.0]),
            burn_in=5_000,
        )
        means = np.mean(result["samples"], axis=0)
        np.testing.assert_allclose(means, [1.0, 2.0], atol=0.3)

    def test_thinning(self) -> None:
        result = metropolis_hastings(
            self._normal_log_posterior,
            initial=np.array([0.0]),
            n_samples=10_000,
            burn_in=1_000,
            thin=5,
        )
        assert result["samples"].shape[0] == 10_000 // 5


# ---------------------------------------------------------------------------
# Gibbs sampler tests
# ---------------------------------------------------------------------------


class TestGibbsSampler:
    def test_normal_mean_with_known_variance(self) -> None:
        """Sample from N(3, 1) using Gibbs with a single conditional."""

        def conditional_mu(params: np.ndarray, rng: np.random.Generator) -> float:
            return float(rng.normal(3.0, 1.0))

        samples = gibbs_sampler(
            conditionals=[conditional_mu],
            initial=np.array([0.0]),
            n_samples=10_000,
            burn_in=1_000,
        )
        assert abs(np.mean(samples) - 3.0) < 0.2

    def test_bivariate_normal(self) -> None:
        """Sample from bivariate normal via Gibbs."""
        rho = 0.5
        mu1, mu2 = 1.0, 2.0

        def cond_x1(params: np.ndarray, rng: np.random.Generator) -> float:
            return float(rng.normal(mu1 + rho * (params[1] - mu2), np.sqrt(1 - rho**2)))

        def cond_x2(params: np.ndarray, rng: np.random.Generator) -> float:
            return float(rng.normal(mu2 + rho * (params[0] - mu1), np.sqrt(1 - rho**2)))

        samples = gibbs_sampler(
            conditionals=[cond_x1, cond_x2],
            initial=np.array([0.0, 0.0]),
            n_samples=20_000,
            burn_in=2_000,
        )
        means = np.mean(samples, axis=0)
        np.testing.assert_allclose(means, [mu1, mu2], atol=0.2)

    def test_wrong_number_of_conditionals_raises(self) -> None:
        def cond(params: np.ndarray, rng: np.random.Generator) -> float:
            return float(rng.normal(0, 1))

        with pytest.raises(ValueError, match="Number of conditionals"):
            gibbs_sampler(
                conditionals=[cond],
                initial=np.array([0.0, 0.0]),
                n_samples=100,
            )


# ---------------------------------------------------------------------------
# Gelman-Rubin tests
# ---------------------------------------------------------------------------


class TestGelmanRubin:
    def test_converged_chains(self) -> None:
        rng = np.random.default_rng(42)
        chains = np.array([
            rng.normal(0, 1, (1000, 2)),
            rng.normal(0, 1, (1000, 2)),
            rng.normal(0, 1, (1000, 2)),
        ])
        r_hat = gelman_rubin(chains)
        np.testing.assert_allclose(r_hat, 1.0, atol=0.1)

    def test_non_converged_chains(self) -> None:
        rng = np.random.default_rng(42)
        chains = np.array([
            rng.normal(0, 1, (1000, 1)),
            rng.normal(5, 1, (1000, 1)),  # different location
        ])
        r_hat = gelman_rubin(chains)
        assert r_hat[0] > 1.5  # should indicate non-convergence

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(42)
        chains = np.array([
            rng.normal(0, 1, (500, 3)),
            rng.normal(0, 1, (500, 3)),
        ])
        r_hat = gelman_rubin(chains)
        assert r_hat.shape == (3,)


# ---------------------------------------------------------------------------
# NUTS diagnostic tests
# ---------------------------------------------------------------------------


class TestNUTSDiagnostic:
    def test_single_chain(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, (1000, 2))
        diag = nuts_diagnostic(samples)
        assert diag["ess"].shape == (2,)
        assert np.all(diag["ess"] > 0)
        assert np.all(np.isnan(diag["r_hat"]))  # single chain
        assert diag["mean"].shape == (2,)
        assert diag["std"].shape == (2,)

    def test_multi_chain(self) -> None:
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, (3, 1000, 2))
        diag = nuts_diagnostic(chains)
        assert diag["ess"].shape == (2,)
        assert diag["r_hat"].shape == (2,)
        np.testing.assert_allclose(diag["r_hat"], 1.0, atol=0.1)


# ---------------------------------------------------------------------------
# Trace summary tests
# ---------------------------------------------------------------------------


class TestTraceSummary:
    def test_output_is_dataframe(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, (1000, 3))
        summary = trace_summary(samples, param_names=["alpha", "beta", "sigma"])
        assert isinstance(summary, pd.DataFrame)
        assert list(summary.index) == ["alpha", "beta", "sigma"]

    def test_columns_present(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, (1000, 2))
        summary = trace_summary(samples)
        assert "mean" in summary.columns
        assert "std" in summary.columns
        assert "ess" in summary.columns
        assert "r_hat" in summary.columns

    def test_default_param_names(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, (1000, 2))
        summary = trace_summary(samples)
        assert list(summary.index) == ["param_0", "param_1"]

    def test_with_multi_chain(self) -> None:
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, (3, 500, 2))
        summary = trace_summary(chains)
        assert "r_hat" in summary.columns
        # R-hat should be close to 1 for converged chains
        np.testing.assert_allclose(summary["r_hat"].values, 1.0, atol=0.15)


# ---------------------------------------------------------------------------
# Hamiltonian Monte Carlo tests
# ---------------------------------------------------------------------------


class TestHamiltonianMonteCarlo:
    def test_samples_converge_to_known_normal(self) -> None:
        """HMC should recover the mean and std of N(3, 1)."""

        def log_p(q: np.ndarray) -> float:
            return float(-0.5 * (q[0] - 3.0) ** 2)

        def grad_log_p(q: np.ndarray) -> np.ndarray:
            return np.array([-(q[0] - 3.0)])

        result = hamiltonian_monte_carlo(
            log_p,
            grad_log_p,
            initial=np.array([0.0]),
            n_samples=5_000,
            step_size=0.1,
            n_leapfrog=20,
            burn_in=1_000,
        )
        samples = result["samples"]
        assert abs(np.mean(samples) - 3.0) < 0.2
        assert abs(np.std(samples) - 1.0) < 0.3

    def test_acceptance_rate_reasonable(self) -> None:
        def log_p(q: np.ndarray) -> float:
            return float(-0.5 * np.sum(q ** 2))

        def grad_log_p(q: np.ndarray) -> np.ndarray:
            return -q

        result = hamiltonian_monte_carlo(
            log_p,
            grad_log_p,
            initial=np.zeros(2),
            n_samples=2_000,
            step_size=0.1,
            n_leapfrog=10,
            burn_in=500,
        )
        assert 0.3 < result["acceptance_rate"] < 1.0

    def test_multivariate_mean(self) -> None:
        """HMC should recover multivariate normal mean."""
        mu = np.array([1.0, -2.0])

        def log_p(q: np.ndarray) -> float:
            return float(-0.5 * np.sum((q - mu) ** 2))

        def grad_log_p(q: np.ndarray) -> np.ndarray:
            return -(q - mu)

        result = hamiltonian_monte_carlo(
            log_p,
            grad_log_p,
            initial=np.zeros(2),
            n_samples=5_000,
            step_size=0.1,
            n_leapfrog=15,
            burn_in=1_000,
        )
        means = np.mean(result["samples"], axis=0)
        np.testing.assert_allclose(means, mu, atol=0.3)

    def test_output_structure(self) -> None:
        def log_p(q: np.ndarray) -> float:
            return float(-0.5 * q[0] ** 2)

        def grad_log_p(q: np.ndarray) -> np.ndarray:
            return -q

        result = hamiltonian_monte_carlo(
            log_p,
            grad_log_p,
            initial=np.array([0.0]),
            n_samples=500,
            burn_in=100,
        )
        assert "samples" in result
        assert "acceptance_rate" in result
        assert "log_posteriors" in result
        assert result["samples"].shape == (500, 1)


# ---------------------------------------------------------------------------
# Slice sampler tests
# ---------------------------------------------------------------------------


class TestSliceSampler:
    def test_normal_mean(self) -> None:
        def log_p(x: float) -> float:
            return -0.5 * (x - 2.0) ** 2

        samples = slice_sampler(log_p, initial=0.0, n_samples=5_000, burn_in=500)
        assert abs(np.mean(samples) - 2.0) < 0.2

    def test_normal_std(self) -> None:
        def log_p(x: float) -> float:
            return -0.5 * (x - 0.0) ** 2

        samples = slice_sampler(log_p, initial=5.0, n_samples=5_000, burn_in=500)
        assert abs(np.std(samples) - 1.0) < 0.3

    def test_output_length(self) -> None:
        def log_p(x: float) -> float:
            return -0.5 * x ** 2

        samples = slice_sampler(log_p, n_samples=1_000, burn_in=100)
        assert len(samples) == 1_000


# ---------------------------------------------------------------------------
# Enhanced convergence diagnostics tests
# ---------------------------------------------------------------------------


class TestConvergenceDiagnostics:
    def test_converged_chains_r_hat_near_one(self) -> None:
        """For IID chains, R-hat and split-R-hat should be near 1."""
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(4, 2000, 3))
        diag = convergence_diagnostics(chains)
        np.testing.assert_allclose(diag["r_hat"].values, 1.0, atol=0.1)
        np.testing.assert_allclose(diag["split_r_hat"].values, 1.0, atol=0.1)

    def test_ess_positive(self) -> None:
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(3, 1000, 2))
        diag = convergence_diagnostics(chains)
        assert np.all(np.array(diag["ess"].values) > 0)

    def test_mcse_smaller_than_std(self) -> None:
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(4, 2000, 2))
        diag = convergence_diagnostics(chains)
        assert np.all(np.array(diag["mcse"].values) < np.array(diag["std"].values))

    def test_output_columns(self) -> None:
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(2, 500, 2))
        diag = convergence_diagnostics(chains, param_names=["alpha", "beta"])
        for col in ["mean", "std", "r_hat", "split_r_hat", "ess", "autocorrelation_time", "mcse"]:
            assert col in diag.columns
        assert list(diag.index) == ["alpha", "beta"]

    def test_single_chain_input(self) -> None:
        """A single 2D chain should be split and analysed."""
        rng = np.random.default_rng(42)
        chain = rng.normal(size=(1000, 2))
        diag = convergence_diagnostics(chain)
        assert diag.shape[0] == 2
        assert np.all(np.isfinite(diag["split_r_hat"].values))
