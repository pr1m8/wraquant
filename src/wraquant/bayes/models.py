"""Pure numpy/scipy Bayesian methods for quantitative finance.

Includes conjugate Bayesian linear regression, Bayesian Sharpe ratio
estimation, Bayesian portfolio allocation, Bayesian VaR, credible
intervals, Bayes factors, and posterior predictive sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = [
    "bayesian_regression",
    "bayesian_sharpe",
    "bayesian_portfolio",
    "bayesian_var",
    "credible_interval",
    "bayes_factor",
    "posterior_predictive",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BayesianRegressionResult:
    """Result container for Bayesian linear regression.

    Parameters
    ----------
    posterior_mean : np.ndarray
        Posterior mean of the coefficient vector.
    posterior_cov : np.ndarray
        Posterior covariance matrix of the coefficients.
    sigma2 : float
        Estimated noise variance.
    log_marginal_likelihood : float
        Log marginal likelihood of the data under the model.
    n_obs : int
        Number of observations.
    n_features : int
        Number of features (including intercept if present).
    """

    posterior_mean: np.ndarray
    posterior_cov: np.ndarray
    sigma2: float
    log_marginal_likelihood: float
    n_obs: int
    n_features: int


@dataclass
class BayesianSharpeResult:
    """Result container for Bayesian Sharpe ratio estimation.

    Parameters
    ----------
    posterior_mean : float
        Posterior mean of the Sharpe ratio.
    posterior_std : float
        Posterior standard deviation of the Sharpe ratio.
    ci_lower : float
        Lower bound of the 95% credible interval.
    ci_upper : float
        Upper bound of the 95% credible interval.
    prob_positive : float
        Posterior probability that the Sharpe ratio is positive.
    samples : np.ndarray
        Posterior samples of the Sharpe ratio.
    """

    posterior_mean: float
    posterior_std: float
    ci_lower: float
    ci_upper: float
    prob_positive: float
    samples: np.ndarray


@dataclass
class BayesianPortfolioResult:
    """Result container for Bayesian portfolio allocation.

    Parameters
    ----------
    weights_mean : np.ndarray
        Mean of posterior portfolio weights.
    weights_std : np.ndarray
        Standard deviation of posterior portfolio weights.
    expected_return : float
        Expected portfolio return under posterior mean weights.
    expected_risk : float
        Expected portfolio risk under posterior mean weights.
    weight_samples : np.ndarray
        Posterior samples of portfolio weights (n_samples, n_assets).
    """

    weights_mean: np.ndarray
    weights_std: np.ndarray
    expected_return: float
    expected_risk: float
    weight_samples: np.ndarray


@dataclass
class BayesianVaRResult:
    """Result container for Bayesian Value-at-Risk.

    Parameters
    ----------
    var_mean : float
        Mean of the posterior VaR distribution.
    var_std : float
        Standard deviation of the posterior VaR distribution.
    ci_lower : float
        Lower bound of the 95% credible interval for VaR.
    ci_upper : float
        Upper bound of the 95% credible interval for VaR.
    var_samples : np.ndarray
        Posterior samples of VaR.
    """

    var_mean: float
    var_std: float
    ci_lower: float
    ci_upper: float
    var_samples: np.ndarray


# ---------------------------------------------------------------------------
# Bayesian linear regression (conjugate prior)
# ---------------------------------------------------------------------------


def bayesian_regression(
    y: np.ndarray,
    X: np.ndarray,
    prior_mean: np.ndarray | None = None,
    prior_cov: np.ndarray | None = None,
) -> BayesianRegressionResult:
    """Conjugate Bayesian linear regression with known noise variance estimate.

    Assumes the model y = X @ beta + eps, eps ~ N(0, sigma^2 I).
    Uses a normal prior on beta and estimates sigma^2 from OLS residuals.

    Parameters
    ----------
    y : np.ndarray
        Response vector (n_obs,).
    X : np.ndarray
        Design matrix (n_obs, n_features). Include an intercept column
        if desired.
    prior_mean : np.ndarray or None
        Prior mean for beta (n_features,). Defaults to zeros.
    prior_cov : np.ndarray or None
        Prior covariance for beta (n_features, n_features). Defaults to
        10 * I (weakly informative).

    Returns
    -------
    BayesianRegressionResult
        Posterior mean, covariance, sigma^2, and log marginal likelihood.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, k = X.shape

    if prior_mean is None:
        prior_mean = np.zeros(k)
    else:
        prior_mean = np.asarray(prior_mean, dtype=float).ravel()

    if prior_cov is None:
        prior_cov = 10.0 * np.eye(k)
    else:
        prior_cov = np.asarray(prior_cov, dtype=float)

    # Estimate sigma^2 from OLS
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta_ols
    sigma2 = float(np.sum(resid**2) / max(n - k, 1))

    # Posterior
    prior_prec = np.linalg.inv(prior_cov)
    data_prec = X.T @ X / sigma2
    posterior_prec = prior_prec + data_prec
    posterior_cov = np.linalg.inv(posterior_prec)
    posterior_mean = posterior_cov @ (prior_prec @ prior_mean + X.T @ y / sigma2)

    # Log marginal likelihood (approximate)
    sign_prior, logdet_prior = np.linalg.slogdet(prior_cov)
    sign_post, logdet_post = np.linalg.slogdet(posterior_cov)
    log_ml = (
        -0.5 * n * np.log(2 * np.pi * sigma2)
        - 0.5 * np.sum(resid**2) / sigma2
        + 0.5 * logdet_post
        - 0.5 * logdet_prior
    )

    return BayesianRegressionResult(
        posterior_mean=posterior_mean,
        posterior_cov=posterior_cov,
        sigma2=sigma2,
        log_marginal_likelihood=float(log_ml),
        n_obs=n,
        n_features=k,
    )


# ---------------------------------------------------------------------------
# Bayesian Sharpe ratio
# ---------------------------------------------------------------------------


def bayesian_sharpe(
    returns: np.ndarray,
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
    n_samples: int = 10_000,
    rng_seed: int = 42,
) -> BayesianSharpeResult:
    """Estimate the Bayesian Sharpe ratio with posterior sampling.

    Uses a normal-inverse-gamma conjugate prior for the mean and variance
    of returns, then computes the posterior distribution of the Sharpe
    ratio (mu / sigma).

    Parameters
    ----------
    returns : np.ndarray
        Array of observed returns.
    prior_mu : float
        Prior mean for the return mean. Default is 0.
    prior_sigma : float
        Prior standard deviation for the return mean. Default is 1.
    n_samples : int
        Number of posterior samples to draw.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    BayesianSharpeResult
        Posterior summary of the Sharpe ratio.
    """
    returns = np.asarray(returns, dtype=float).ravel()
    n = len(returns)
    rng = np.random.default_rng(rng_seed)

    # Sufficient statistics
    y_bar = np.mean(returns)
    s2 = np.var(returns, ddof=1)

    # Normal-inverse-gamma posterior parameters
    prior_var = prior_sigma**2
    kappa_0 = 1.0 / prior_var
    kappa_n = kappa_0 + n
    mu_n = (kappa_0 * prior_mu + n * y_bar) / kappa_n

    # Posterior for sigma^2: Inverse-Gamma
    alpha_0 = 1.0  # weakly informative
    beta_0 = 1.0
    alpha_n = alpha_0 + n / 2.0
    beta_n = beta_0 + 0.5 * (n - 1) * s2 + 0.5 * kappa_0 * n * (y_bar - prior_mu) ** 2 / kappa_n

    # Sample sigma^2 from Inverse-Gamma (via Gamma for 1/sigma^2)
    sigma2_samples = 1.0 / rng.gamma(alpha_n, 1.0 / beta_n, size=n_samples)
    sigma2_samples = np.maximum(sigma2_samples, 1e-15)

    # Sample mu | sigma^2 from Normal
    mu_samples = rng.normal(mu_n, np.sqrt(sigma2_samples / kappa_n))

    # Sharpe ratio samples
    sharpe_samples = mu_samples / np.sqrt(sigma2_samples)

    ci = credible_interval(sharpe_samples, alpha=0.05)

    return BayesianSharpeResult(
        posterior_mean=float(np.mean(sharpe_samples)),
        posterior_std=float(np.std(sharpe_samples)),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        prob_positive=float(np.mean(sharpe_samples > 0)),
        samples=sharpe_samples,
    )


# ---------------------------------------------------------------------------
# Bayesian portfolio allocation
# ---------------------------------------------------------------------------


def bayesian_portfolio(
    returns: np.ndarray,
    prior_cov_scale: float = 1.0,
    n_samples: int = 5_000,
    rng_seed: int = 42,
) -> BayesianPortfolioResult:
    """Bayesian portfolio allocation via posterior sampling.

    Samples from the posterior of (mu, Sigma) using a conjugate
    normal-inverse-Wishart prior, then computes the mean-variance
    optimal portfolio for each posterior draw.

    Parameters
    ----------
    returns : np.ndarray
        Return matrix (n_periods, n_assets).
    prior_cov_scale : float
        Scale factor for the prior covariance. Larger values give a
        more diffuse prior. Default is 1.0.
    n_samples : int
        Number of posterior portfolio weight samples.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    BayesianPortfolioResult
        Posterior summary of portfolio weights.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    n, p = returns.shape
    rng = np.random.default_rng(rng_seed)

    # Sufficient statistics
    y_bar = np.mean(returns, axis=0)
    S = np.cov(returns, rowvar=False, ddof=1)
    if S.ndim == 0:
        S = S.reshape(1, 1)

    # Conjugate prior parameters (weakly informative)
    mu_0 = np.zeros(p)
    kappa_0 = 0.01
    nu_0 = p + 2.0
    Psi_0 = prior_cov_scale * np.eye(p)

    # Posterior parameters
    kappa_n = kappa_0 + n
    mu_n = (kappa_0 * mu_0 + n * y_bar) / kappa_n
    nu_n = nu_0 + n
    Psi_n = Psi_0 + (n - 1) * S + (kappa_0 * n / kappa_n) * np.outer(y_bar - mu_0, y_bar - mu_0)

    weight_samples = np.zeros((n_samples, p))

    for i in range(n_samples):
        # Sample Sigma from Inverse-Wishart
        # Inverse-Wishart(nu_n, Psi_n) = inv(Wishart(nu_n, inv(Psi_n)))
        Psi_n_inv = np.linalg.inv(Psi_n)
        # Ensure symmetric positive definite
        Psi_n_inv = 0.5 * (Psi_n_inv + Psi_n_inv.T) + 1e-10 * np.eye(p)
        W = stats.wishart.rvs(df=int(nu_n), scale=Psi_n_inv, random_state=rng)
        if np.ndim(W) == 0:
            W = np.array([[W]])
        Sigma_sample = np.linalg.inv(W)
        Sigma_sample = 0.5 * (Sigma_sample + Sigma_sample.T)

        # Sample mu | Sigma from Normal
        mu_cov = Sigma_sample / kappa_n
        mu_cov = 0.5 * (mu_cov + mu_cov.T) + 1e-12 * np.eye(p)
        mu_sample = rng.multivariate_normal(mu_n, mu_cov)

        # Compute mean-variance optimal weights (unconstrained)
        try:
            Sigma_inv = np.linalg.inv(Sigma_sample + 1e-10 * np.eye(p))
            w = Sigma_inv @ mu_sample
            w_sum = np.sum(w)
            if abs(w_sum) > 1e-12:
                w = w / w_sum  # normalize to sum to 1
            else:
                w = np.ones(p) / p
        except np.linalg.LinAlgError:
            w = np.ones(p) / p

        weight_samples[i] = w

    weights_mean = np.mean(weight_samples, axis=0)
    weights_std = np.std(weight_samples, axis=0)

    # Portfolio metrics under posterior mean weights
    expected_return = float(weights_mean @ y_bar)
    expected_risk = float(np.sqrt(weights_mean @ S @ weights_mean))

    return BayesianPortfolioResult(
        weights_mean=weights_mean,
        weights_std=weights_std,
        expected_return=expected_return,
        expected_risk=expected_risk,
        weight_samples=weight_samples,
    )


# ---------------------------------------------------------------------------
# Bayesian VaR
# ---------------------------------------------------------------------------


def bayesian_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    n_posterior: int = 10_000,
    rng_seed: int = 42,
) -> BayesianVaRResult:
    """Bayesian Value-at-Risk with parameter uncertainty.

    Samples from the posterior of (mu, sigma^2) using conjugate priors,
    then computes VaR for each posterior draw to account for parameter
    uncertainty.

    Parameters
    ----------
    returns : np.ndarray
        Array of observed returns.
    confidence : float
        Confidence level for VaR (e.g., 0.95 for 95% VaR).
    n_posterior : int
        Number of posterior samples.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    BayesianVaRResult
        Posterior summary of VaR.
    """
    returns = np.asarray(returns, dtype=float).ravel()
    n = len(returns)
    rng = np.random.default_rng(rng_seed)

    y_bar = np.mean(returns)
    s2 = np.var(returns, ddof=1)

    # Weakly informative conjugate prior
    kappa_0 = 0.01
    mu_0 = 0.0
    alpha_0 = 1.0
    beta_0 = 1.0

    kappa_n = kappa_0 + n
    mu_n = (kappa_0 * mu_0 + n * y_bar) / kappa_n
    alpha_n = alpha_0 + n / 2.0
    beta_n = beta_0 + 0.5 * (n - 1) * s2 + 0.5 * kappa_0 * n * (y_bar - mu_0) ** 2 / kappa_n

    # Sample sigma^2 from Inverse-Gamma
    sigma2_samples = 1.0 / rng.gamma(alpha_n, 1.0 / beta_n, size=n_posterior)
    sigma2_samples = np.maximum(sigma2_samples, 1e-15)

    # Sample mu | sigma^2
    mu_samples = rng.normal(mu_n, np.sqrt(sigma2_samples / kappa_n))

    # VaR for each posterior sample
    z = stats.norm.ppf(1 - confidence)
    var_samples = -(mu_samples + z * np.sqrt(sigma2_samples))

    ci = credible_interval(var_samples, alpha=0.05)

    return BayesianVaRResult(
        var_mean=float(np.mean(var_samples)),
        var_std=float(np.std(var_samples)),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        var_samples=var_samples,
    )


# ---------------------------------------------------------------------------
# Credible interval
# ---------------------------------------------------------------------------


def credible_interval(
    samples: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute the Highest Posterior Density (HPD) credible interval.

    Uses the shortest interval containing (1 - alpha) of the posterior
    mass.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples (1D array).
    alpha : float
        Significance level. Default is 0.05 (95% credible interval).

    Returns
    -------
    tuple[float, float]
        (lower, upper) bounds of the HPD interval.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    n_included = int(np.ceil((1 - alpha) * n))

    # Find the shortest interval
    widths = sorted_samples[n_included:] - sorted_samples[: n - n_included]
    if len(widths) == 0:
        return (float(sorted_samples[0]), float(sorted_samples[-1]))

    idx = int(np.argmin(widths))
    return (float(sorted_samples[idx]), float(sorted_samples[idx + n_included]))


# ---------------------------------------------------------------------------
# Bayes factor
# ---------------------------------------------------------------------------


def bayes_factor(
    log_likelihood_1: float,
    log_likelihood_2: float,
) -> float:
    """Compute the Bayes factor comparing model 1 to model 2.

    BF = exp(log_likelihood_1 - log_likelihood_2).

    Parameters
    ----------
    log_likelihood_1 : float
        Log marginal likelihood of model 1.
    log_likelihood_2 : float
        Log marginal likelihood of model 2.

    Returns
    -------
    float
        Bayes factor (BF > 1 favors model 1, BF < 1 favors model 2).
    """
    diff = log_likelihood_1 - log_likelihood_2
    # Clip to avoid overflow
    diff = np.clip(diff, -500, 500)
    return float(np.exp(diff))


# ---------------------------------------------------------------------------
# Posterior predictive
# ---------------------------------------------------------------------------


def posterior_predictive(
    y: np.ndarray,
    X: np.ndarray,
    prior_mean: np.ndarray | None = None,
    prior_cov: np.ndarray | None = None,
    n_samples: int = 1_000,
    rng_seed: int = 42,
    X_new: np.ndarray | None = None,
) -> np.ndarray:
    """Generate posterior predictive samples for Bayesian linear regression.

    Parameters
    ----------
    y : np.ndarray
        Response vector (n_obs,).
    X : np.ndarray
        Design matrix (n_obs, n_features).
    prior_mean : np.ndarray or None
        Prior mean for beta. Defaults to zeros.
    prior_cov : np.ndarray or None
        Prior covariance for beta. Defaults to 10 * I.
    n_samples : int
        Number of posterior predictive samples.
    rng_seed : int
        Random seed for reproducibility.
    X_new : np.ndarray or None
        New design matrix for predictions. If None, uses X.

    Returns
    -------
    np.ndarray
        Posterior predictive samples, shape (n_samples, n_pred).
    """
    rng = np.random.default_rng(rng_seed)

    result = bayesian_regression(y, X, prior_mean, prior_cov)

    if X_new is None:
        X_new = np.asarray(X, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
    else:
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

    n_pred = X_new.shape[0]
    predictions = np.zeros((n_samples, n_pred))

    # Ensure posterior_cov is positive definite
    post_cov = result.posterior_cov
    post_cov = 0.5 * (post_cov + post_cov.T) + 1e-10 * np.eye(post_cov.shape[0])

    for i in range(n_samples):
        # Sample beta from posterior
        beta_sample = rng.multivariate_normal(result.posterior_mean, post_cov)
        # Sample noise
        y_hat = X_new @ beta_sample
        noise = rng.normal(0, np.sqrt(result.sigma2), size=n_pred)
        predictions[i] = y_hat + noise

    return predictions
