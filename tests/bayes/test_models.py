"""Tests for Bayesian models (pure numpy/scipy)."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.bayes.models import (
    bayes_factor,
    bayesian_portfolio,
    bayesian_regression,
    bayesian_sharpe,
    bayesian_var,
    credible_interval,
    posterior_predictive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regression_data(
    n: int = 200,
    k: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate regression data with known coefficients."""
    rng = np.random.default_rng(seed)
    true_beta = np.arange(1, k + 1, dtype=float)
    X = np.column_stack([np.ones(n)] + [rng.normal(0, 1, n) for _ in range(k - 1)])
    y = X @ true_beta + rng.normal(0, 0.5, n)
    return y, X, true_beta


# ---------------------------------------------------------------------------
# Bayesian regression tests
# ---------------------------------------------------------------------------


class TestBayesianRegression:
    def test_recovers_coefficients(self) -> None:
        y, X, true_beta = _make_regression_data()
        result = bayesian_regression(y, X)
        np.testing.assert_allclose(result.posterior_mean, true_beta, atol=0.3)

    def test_output_structure(self) -> None:
        y, X, _ = _make_regression_data()
        result = bayesian_regression(y, X)
        assert result.n_obs == 200
        assert result.n_features == 2
        assert result.posterior_mean.shape == (2,)
        assert result.posterior_cov.shape == (2, 2)
        assert result.sigma2 > 0

    def test_posterior_cov_symmetric(self) -> None:
        y, X, _ = _make_regression_data()
        result = bayesian_regression(y, X)
        np.testing.assert_allclose(result.posterior_cov, result.posterior_cov.T)

    def test_informative_prior_shrinks(self) -> None:
        """An informative prior should shrink estimates toward the prior."""
        y, X, _ = _make_regression_data()
        prior_mean = np.zeros(2)
        prior_cov = 0.01 * np.eye(2)  # very tight prior at zero
        result = bayesian_regression(y, X, prior_mean, prior_cov)
        # Posterior should be pulled toward zero
        assert np.linalg.norm(result.posterior_mean) < np.linalg.norm(
            np.array([1.0, 2.0])
        )

    def test_log_marginal_likelihood_finite(self) -> None:
        y, X, _ = _make_regression_data()
        result = bayesian_regression(y, X)
        assert np.isfinite(result.log_marginal_likelihood)


# ---------------------------------------------------------------------------
# Bayesian Sharpe ratio tests
# ---------------------------------------------------------------------------


class TestBayesianSharpe:
    def test_positive_sharpe_for_positive_returns(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.05, 0.1, 500)
        result = bayesian_sharpe(returns)
        assert result.posterior_mean > 0
        assert result.prob_positive > 0.5

    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.01, 0.1, 100)
        result = bayesian_sharpe(returns, n_samples=5000)
        assert hasattr(result, "posterior_mean")
        assert hasattr(result, "posterior_std")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "prob_positive")
        assert len(result.samples) == 5000

    def test_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.02, 0.1, 200)
        result = bayesian_sharpe(returns)
        assert result.ci_lower < result.posterior_mean < result.ci_upper

    def test_zero_returns_uncertain(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.1, 100)
        result = bayesian_sharpe(returns)
        # With zero mean returns, prob_positive should be near 0.5
        assert 0.2 < result.prob_positive < 0.8


# ---------------------------------------------------------------------------
# Bayesian portfolio tests
# ---------------------------------------------------------------------------


class TestBayesianPortfolio:
    def test_weights_shape(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.01, 0.05, (100, 3))
        result = bayesian_portfolio(returns, n_samples=500)
        assert result.weights_mean.shape == (3,)
        assert result.weights_std.shape == (3,)
        assert result.weight_samples.shape == (500, 3)

    def test_expected_return_reasonable(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.01, 0.05, (100, 3))
        result = bayesian_portfolio(returns, n_samples=500)
        assert np.isfinite(result.expected_return)
        assert np.isfinite(result.expected_risk)
        assert result.expected_risk >= 0


# ---------------------------------------------------------------------------
# Bayesian VaR tests
# ---------------------------------------------------------------------------


class TestBayesianVaR:
    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.01, 0.05, 200)
        result = bayesian_var(returns, confidence=0.95)
        assert hasattr(result, "var_mean")
        assert hasattr(result, "var_std")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert len(result.var_samples) == 10_000

    def test_higher_confidence_higher_var(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.01, 0.05, 500)
        var_95 = bayesian_var(returns, confidence=0.95)
        var_99 = bayesian_var(returns, confidence=0.99)
        assert var_99.var_mean > var_95.var_mean

    def test_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.05, 200)
        result = bayesian_var(returns)
        assert result.ci_lower < result.var_mean < result.ci_upper


# ---------------------------------------------------------------------------
# Credible interval tests
# ---------------------------------------------------------------------------


class TestCredibleInterval:
    def test_95_pct_interval(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 100_000)
        lower, upper = credible_interval(samples, alpha=0.05)
        # For standard normal, 95% HPD is approximately [-1.96, 1.96]
        assert abs(lower + 1.96) < 0.1
        assert abs(upper - 1.96) < 0.1

    def test_narrower_interval_for_higher_alpha(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 10_000)
        ci_95 = credible_interval(samples, alpha=0.05)
        ci_50 = credible_interval(samples, alpha=0.50)
        assert (ci_95[1] - ci_95[0]) > (ci_50[1] - ci_50[0])

    def test_skewed_distribution(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.exponential(1.0, 10_000)
        lower, upper = credible_interval(samples, alpha=0.05)
        assert lower >= 0
        assert upper > lower


# ---------------------------------------------------------------------------
# Bayes factor tests
# ---------------------------------------------------------------------------


class TestBayesFactor:
    def test_equal_models(self) -> None:
        bf = bayes_factor(0.0, 0.0)
        assert bf == pytest.approx(1.0)

    def test_model_1_better(self) -> None:
        bf = bayes_factor(-10.0, -20.0)
        assert bf > 1.0

    def test_model_2_better(self) -> None:
        bf = bayes_factor(-20.0, -10.0)
        assert bf < 1.0

    def test_handles_large_differences(self) -> None:
        bf = bayes_factor(100.0, -100.0)
        assert np.isfinite(bf)
        assert bf > 0


# ---------------------------------------------------------------------------
# Posterior predictive tests
# ---------------------------------------------------------------------------


class TestPosteriorPredictive:
    def test_output_shape(self) -> None:
        y, X, _ = _make_regression_data()
        samples = posterior_predictive(y, X, n_samples=100)
        assert samples.shape == (100, 200)

    def test_predictions_at_new_points(self) -> None:
        y, X, _ = _make_regression_data()
        X_new = np.column_stack([np.ones(5), np.zeros(5)])
        samples = posterior_predictive(y, X, X_new=X_new, n_samples=100)
        assert samples.shape == (100, 5)

    def test_predictions_centered_near_truth(self) -> None:
        y, X, true_beta = _make_regression_data(n=500)
        X_new = np.column_stack([np.ones(10), np.zeros(10)])
        samples = posterior_predictive(y, X, X_new=X_new, n_samples=5000)
        # Predicted mean at X_new should be close to intercept (true_beta[0] = 1)
        pred_mean = np.mean(samples, axis=0)
        np.testing.assert_allclose(pred_mean, 1.0, atol=0.3)
