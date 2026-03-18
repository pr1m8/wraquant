"""Tests for Bayesian models (pure numpy/scipy)."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.bayes.models import (
    bayes_factor,
    bayesian_changepoint,
    bayesian_cointegration,
    bayesian_factor_model,
    bayesian_linear_regression,
    bayesian_portfolio,
    bayesian_portfolio_bl,
    bayesian_regression,
    bayesian_sharpe,
    bayesian_var,
    bayesian_volatility,
    credible_interval,
    model_comparison,
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


# ---------------------------------------------------------------------------
# Enhanced Bayesian linear regression (Normal-InverseGamma)
# ---------------------------------------------------------------------------


class TestBayesianLinearRegression:
    def test_posterior_mean_close_to_ols(self) -> None:
        """The posterior mean should converge to OLS with weak priors."""
        y, X, true_beta = _make_regression_data(n=500)
        result = bayesian_linear_regression(y, X)
        # OLS estimate
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(result.posterior_mean, beta_ols, atol=0.1)

    def test_credible_intervals_contain_truth(self) -> None:
        y, X, true_beta = _make_regression_data(n=300)
        result = bayesian_linear_regression(y, X)
        for j in range(len(true_beta)):
            assert result.credible_intervals[j, 0] < true_beta[j] < result.credible_intervals[j, 1]

    def test_log_marginal_likelihood_finite(self) -> None:
        y, X, _ = _make_regression_data()
        result = bayesian_linear_regression(y, X)
        assert np.isfinite(result.log_marginal_likelihood)

    def test_sigma2_mean_positive(self) -> None:
        y, X, _ = _make_regression_data()
        result = bayesian_linear_regression(y, X)
        assert result.sigma2_mean > 0

    def test_output_shapes(self) -> None:
        y, X, _ = _make_regression_data()
        result = bayesian_linear_regression(y, X)
        assert result.posterior_mean.shape == (2,)
        assert result.posterior_cov_unscaled.shape == (2, 2)
        assert result.credible_intervals.shape == (2, 2)
        assert result.n_obs == 200
        assert result.n_features == 2


# ---------------------------------------------------------------------------
# Bayesian changepoint detection
# ---------------------------------------------------------------------------


class TestBayesianChangepoint:
    def test_detects_known_changepoint(self) -> None:
        """Should detect a changepoint when the mean shifts."""
        rng = np.random.default_rng(42)
        data = np.concatenate([
            rng.normal(0, 1, 100),
            rng.normal(5, 1, 100),
        ])
        result = bayesian_changepoint(data, hazard=1.0 / 50.0, threshold=0.05)
        # The changepoint posterior should spike near the true changepoint
        cp_post = result.changepoint_posterior
        # Find the index with maximum changepoint probability after index 80
        peak_idx = 80 + np.argmax(cp_post[80:120])
        assert abs(peak_idx - 100) < 15, f"Peak at {peak_idx}, expected near 100"

    def test_no_changepoint_in_stationary(self) -> None:
        """With stationary data and low hazard, few changepoints expected."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        result = bayesian_changepoint(data, hazard=1.0 / 200.0, threshold=0.5)
        # Should have very few (or no) high-probability changepoints
        # We use a high threshold to be strict
        assert len(result.most_likely_changepoints) < 20

    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 50)
        result = bayesian_changepoint(data)
        assert result.changepoint_posterior.shape == (50,)
        assert result.run_length_probs.shape[0] == 50


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


class TestModelComparison:
    def test_correct_ranking(self) -> None:
        """The model containing the true predictor should rank first."""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)  # noise predictor
        y = 1.0 + 2.0 * x1 + rng.normal(0, 0.5, n)

        X_good = np.column_stack([np.ones(n), x1])
        X_bad = np.column_stack([np.ones(n), x2])

        df = model_comparison(y, [X_good, X_bad], ["good", "bad"])
        assert df.index[0] == "good"
        assert df.loc["good", "rank"] == 1
        assert df.loc["bad", "rank"] == 2

    def test_bayes_factor_best_model_is_one(self) -> None:
        """The best model should have Bayes factor = 1."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.normal(size=n)
        y = x + rng.normal(0, 0.5, n)
        X1 = np.column_stack([np.ones(n), x])
        X2 = np.ones((n, 1))
        df = model_comparison(y, [X1, X2])
        best_name = df.index[0]
        assert df.loc[best_name, "bayes_factor"] == pytest.approx(1.0)

    def test_output_columns(self) -> None:
        rng = np.random.default_rng(42)
        n = 50
        y = rng.normal(size=n)
        X = np.ones((n, 1))
        df = model_comparison(y, [X])
        for col in ["log_marginal_likelihood", "waic", "loo_cv", "bayes_factor", "rank"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Bayesian factor model
# ---------------------------------------------------------------------------


class TestBayesianFactorModel:
    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        result = bayesian_factor_model(X, n_factors=2, n_samples=200)
        assert result.loadings_mean.shape == (5, 2)
        assert result.loadings_std.shape == (5, 2)
        assert result.scores_mean.shape == (100, 2)
        assert result.explained_variance.shape == (2,)
        assert result.explained_variance_ci.shape == (2, 2)
        assert result.noise_variance.shape == (5,)
        assert result.n_factors == 2

    def test_explained_variance_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        F = rng.normal(size=(150, 2))
        L = rng.normal(size=(4, 2))
        X = F @ L.T + rng.normal(0, 0.3, (150, 4))
        result = bayesian_factor_model(X, n_factors=2, n_samples=200)
        assert np.all(result.explained_variance >= 0)


# ---------------------------------------------------------------------------
# Bayesian portfolio (Black-Litterman)
# ---------------------------------------------------------------------------


class TestBayesianPortfolioBL:
    def test_weights_shape(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, (200, 3))
        result = bayesian_portfolio_bl(returns, n_samples=500)
        assert result.weights_mean.shape == (3,)
        assert result.weights_std.shape == (3,)
        assert result.weights_ci.shape == (3, 2)
        assert result.weight_samples.shape == (500, 3)

    def test_with_views(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, (200, 3))
        P = np.array([[1, -1, 0]])
        q = np.array([0.001])
        result = bayesian_portfolio_bl(returns, views=q, P=P, n_samples=500)
        assert result.posterior_mean.shape == (3,)
        assert result.posterior_cov.shape == (3, 3)


# ---------------------------------------------------------------------------
# Bayesian volatility
# ---------------------------------------------------------------------------


class TestBayesianVolatility:
    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 100)
        result = bayesian_volatility(returns, n_samples=200, burn_in=100)
        assert result.vol_mean.shape == (100,)
        assert result.vol_ci_lower.shape == (100,)
        assert result.vol_ci_upper.shape == (100,)
        assert len(result.mu_posterior) == 200
        assert len(result.phi_posterior) == 200
        assert len(result.sigma_eta_posterior) == 200

    def test_vol_positive(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 100)
        result = bayesian_volatility(returns, n_samples=200, burn_in=100)
        assert np.all(result.vol_mean > 0)
        assert np.all(result.vol_ci_lower > 0)

    def test_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 100)
        result = bayesian_volatility(returns, n_samples=200, burn_in=100)
        assert np.all(result.vol_ci_lower <= result.vol_mean)
        assert np.all(result.vol_ci_upper >= result.vol_mean)


# ---------------------------------------------------------------------------
# Bayesian cointegration
# ---------------------------------------------------------------------------


class TestBayesianCointegration:
    def test_cointegrated_pair(self) -> None:
        """Should find high probability of cointegration for a known pair."""
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.normal(size=300))
        y = 0.8 * x + rng.normal(0, 0.5, 300)
        result = bayesian_cointegration(y, x)
        assert result.prob_cointegrated > 0.7

    def test_cointegrated_higher_than_independent(self) -> None:
        """A truly cointegrated pair should have a more negative rho
        (stronger mean reversion) than independent random walks."""
        rng = np.random.default_rng(42)
        # Cointegrated pair
        x_coint = np.cumsum(rng.normal(size=300))
        y_coint = 0.8 * x_coint + rng.normal(0, 0.5, 300)
        result_coint = bayesian_cointegration(y_coint, x_coint)

        # Independent random walks
        x_ind = np.cumsum(rng.normal(size=300))
        y_ind = np.cumsum(rng.normal(size=300))
        result_ind = bayesian_cointegration(y_ind, x_ind)

        # The cointegrated pair should have a more negative mean rho
        mean_rho_coint = np.mean(result_coint.residual_adf_samples)
        mean_rho_ind = np.mean(result_ind.residual_adf_samples)
        assert mean_rho_coint < mean_rho_ind

    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.normal(size=100))
        y = 0.5 * x + rng.normal(0, 1, 100)
        result = bayesian_cointegration(y, x, n_samples=1000)
        assert hasattr(result, "prob_cointegrated")
        assert hasattr(result, "cointegrating_vector_mean")
        assert hasattr(result, "spread_mean")
        assert len(result.residual_adf_samples) == 1000
        assert len(result.spread_mean) == 100
