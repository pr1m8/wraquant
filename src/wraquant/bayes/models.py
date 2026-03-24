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
    "bayesian_linear_regression",
    "bayesian_factor_model",
    "bayesian_changepoint",
    "bayesian_portfolio_bl",
    "bayesian_volatility",
    "bayesian_cointegration",
    "bayesian_regime_inference",
    "model_comparison",
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

    # Estimate sigma^2 from OLS (use canonical wraquant regression)
    from wraquant.stats.regression import ols as _ols

    _ols_result = _ols(y, X, add_constant=False)
    beta_ols = _ols_result["coefficients"]
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


# ---------------------------------------------------------------------------
# Enhanced Bayesian linear regression (Normal-InverseGamma conjugate)
# ---------------------------------------------------------------------------


@dataclass
class BayesianLinearRegressionResult:
    """Result container for the enhanced Bayesian linear regression.

    This uses the full Normal-Inverse-Gamma conjugate prior, which means
    **both** the regression coefficients and the noise variance are treated
    as unknown and given a joint prior.  The posterior is available in
    closed form, so no MCMC is needed.

    Attributes:
        posterior_mean: Posterior mean of the coefficient vector beta.
        posterior_cov_unscaled: Posterior precision-scaled covariance
            (V_n).  The actual posterior covariance of beta given sigma^2
            is ``sigma^2 * posterior_cov_unscaled``.
        a_n: Shape parameter of the posterior Inverse-Gamma for sigma^2.
        b_n: Scale parameter of the posterior Inverse-Gamma for sigma^2.
        sigma2_mean: Posterior mean of sigma^2 = b_n / (a_n - 1).
        log_marginal_likelihood: Log marginal likelihood p(y | model),
            used for Bayes factor model comparison.
        credible_intervals: (n_features, 2) array of 95 % credible
            intervals for each coefficient, marginalised over sigma^2
            (Student-t).
        n_obs: Number of observations.
        n_features: Number of features.

    Notes:
        **Bayesian vs frequentist**: A 95 % credible interval means
        "there is a 95 % posterior probability that the parameter lies in
        this interval", which is the statement most practitioners actually
        want.  A frequentist 95 % confidence interval instead says "if we
        repeated the experiment many times, 95 % of the resulting intervals
        would contain the true value" -- a subtly different claim.
    """

    posterior_mean: np.ndarray
    posterior_cov_unscaled: np.ndarray
    a_n: float
    b_n: float
    sigma2_mean: float
    log_marginal_likelihood: float
    credible_intervals: np.ndarray
    n_obs: int
    n_features: int


def bayesian_linear_regression(
    y: np.ndarray,
    X: np.ndarray,
    prior_mean: np.ndarray | None = None,
    prior_cov: np.ndarray | None = None,
    a_0: float = 1.0,
    b_0: float = 1.0,
    alpha: float = 0.05,
) -> BayesianLinearRegressionResult:
    """Enhanced Bayesian linear regression with Normal-Inverse-Gamma prior.

    This is the "textbook" conjugate Bayesian regression where both the
    coefficients **and** the noise variance are unknown:

        y | X, beta, sigma^2 ~ N(X beta, sigma^2 I)
        beta | sigma^2       ~ N(m_0, sigma^2 V_0)
        sigma^2              ~ InvGamma(a_0, b_0)

    The posterior is available in closed form as:

        beta | sigma^2, y ~ N(m_n, sigma^2 V_n)
        sigma^2 | y       ~ InvGamma(a_n, b_n)
        beta | y          ~ Student-t(2 a_n, m_n, (b_n / a_n) V_n)

    **When to use this instead of OLS**: Use this when you want full
    uncertainty quantification (credible intervals on coefficients *and*
    on sigma^2), when your data set is small and prior information
    matters, or when you need the marginal likelihood for model
    comparison (Bayes factors).

    Args:
        y: Response vector of shape ``(n,)``.
        X: Design matrix of shape ``(n, k)``.  Include an intercept
            column yourself if you want one.
        prior_mean: Prior mean for beta, shape ``(k,)``.  Defaults to
            zeros (agnostic prior).
        prior_cov: Prior covariance scale for beta, ``V_0`` of shape
            ``(k, k)``.  The actual prior covariance is
            ``sigma^2 * prior_cov``.  Defaults to ``100 * I`` (weakly
            informative).
        a_0: Shape parameter for the Inverse-Gamma prior on sigma^2.
            Default ``1.0``.
        b_0: Scale parameter for the Inverse-Gamma prior on sigma^2.
            Default ``1.0``.
        alpha: Significance level for credible intervals. Default ``0.05``
            gives 95 % intervals.

    Returns:
        BayesianLinearRegressionResult with closed-form posterior
        analytics, credible intervals, and log marginal likelihood.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = np.column_stack([np.ones(100), rng.normal(size=100)])
        >>> y = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, 100)
        >>> result = bayesian_linear_regression(y, X)
        >>> print(result.posterior_mean)  # close to [1, 2]
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
        prior_cov = 100.0 * np.eye(k)
    else:
        prior_cov = np.asarray(prior_cov, dtype=float)

    # Prior precision
    V0_inv = np.linalg.inv(prior_cov)

    # Posterior for beta
    XtX = X.T @ X
    Vn_inv = V0_inv + XtX
    V_n = np.linalg.inv(Vn_inv)
    m_n = V_n @ (V0_inv @ prior_mean + X.T @ y)

    # Posterior for sigma^2
    a_n = a_0 + n / 2.0
    residual_term = float(
        y @ y
        + prior_mean @ V0_inv @ prior_mean
        - m_n @ Vn_inv @ m_n
    )
    b_n = b_0 + 0.5 * residual_term

    sigma2_mean = b_n / max(a_n - 1.0, 1e-12)

    # Log marginal likelihood  p(y | model)
    sign0, logdet_V0 = np.linalg.slogdet(prior_cov)
    sign_n, logdet_Vn = np.linalg.slogdet(V_n)
    from scipy.special import gammaln

    log_ml = (
        -0.5 * n * np.log(2.0 * np.pi)
        + 0.5 * logdet_Vn
        - 0.5 * logdet_V0
        + a_0 * np.log(b_0)
        - a_n * np.log(b_n)
        + float(gammaln(a_n) - gammaln(a_0))
    )

    # Credible intervals via marginal Student-t for each coefficient
    df_t = 2.0 * a_n
    scale = np.sqrt(np.diag(V_n) * b_n / a_n)
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df_t)
    ci = np.column_stack([m_n - t_crit * scale, m_n + t_crit * scale])

    return BayesianLinearRegressionResult(
        posterior_mean=m_n,
        posterior_cov_unscaled=V_n,
        a_n=float(a_n),
        b_n=float(b_n),
        sigma2_mean=float(sigma2_mean),
        log_marginal_likelihood=float(log_ml),
        credible_intervals=ci,
        n_obs=n,
        n_features=k,
    )


# ---------------------------------------------------------------------------
# Bayesian factor model (Bayesian PCA)
# ---------------------------------------------------------------------------


@dataclass
class BayesianFactorModelResult:
    """Result container for the Bayesian factor model.

    Attributes:
        loadings_mean: Posterior mean of the factor loadings matrix,
            shape ``(n_variables, n_factors)``.
        loadings_std: Posterior std of the factor loadings, same shape.
        scores_mean: Posterior mean of the factor scores,
            shape ``(n_obs, n_factors)``.
        explained_variance: Fraction of total variance explained by each
            factor, shape ``(n_factors,)``.
        explained_variance_ci: 95 % credible intervals for explained
            variance, shape ``(n_factors, 2)``.
        noise_variance: Estimated idiosyncratic noise variance per
            variable, shape ``(n_variables,)``.
        n_factors: Number of latent factors used.
    """

    loadings_mean: np.ndarray
    loadings_std: np.ndarray
    scores_mean: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ci: np.ndarray
    noise_variance: np.ndarray
    n_factors: int


def bayesian_factor_model(
    X: np.ndarray,
    n_factors: int = 2,
    n_samples: int = 1_000,
    rng_seed: int = 42,
) -> BayesianFactorModelResult:
    """Bayesian factor model via Gibbs sampling (Bayesian PCA).

    Estimates a latent factor model of the form:

        X = F @ Lambda^T + epsilon

    where F is the ``(n_obs, n_factors)`` matrix of latent factor scores
    and Lambda is the ``(n_variables, n_factors)`` loading matrix.  Both
    are given conjugate Gaussian priors and sampled via Gibbs.

    **When to use this**: Use a Bayesian factor model instead of standard
    PCA when you need uncertainty estimates on the loadings and explained
    variance, especially with short financial time series where the
    number of observations is not much larger than the number of assets.

    Args:
        X: Data matrix of shape ``(n_obs, n_variables)``.  Columns are
            typically asset returns.
        n_factors: Number of latent factors to estimate.
        n_samples: Number of Gibbs samples to draw (after a built-in
            burn-in of ``n_samples // 2``).
        rng_seed: Random seed for reproducibility.

    Returns:
        BayesianFactorModelResult with posterior summaries for loadings,
        scores, and explained variance.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> F = rng.normal(size=(200, 2))
        >>> L = rng.normal(size=(5, 2))
        >>> X = F @ L.T + rng.normal(0, 0.3, (200, 5))
        >>> result = bayesian_factor_model(X, n_factors=2)
        >>> print(result.loadings_mean.shape)  # (5, 2)
    """
    rng = np.random.default_rng(rng_seed)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    # Centre the data
    X_mean = X.mean(axis=0)
    X_c = X - X_mean

    burn_in = n_samples // 2
    total = n_samples + burn_in

    # Initialise from SVD
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    F = U[:, :n_factors] * S[:n_factors]
    Lambda = Vt[:n_factors, :].T
    psi = np.var(X_c - F @ Lambda.T, axis=0) + 1e-6

    # Storage
    Lambda_samples = np.zeros((n_samples, p, n_factors))
    F_samples = np.zeros((n_samples, n, n_factors))
    psi_samples = np.zeros((n_samples, p))

    for it in range(total):
        # --- Sample Lambda | F, psi ---
        for j in range(p):
            prec_L = np.eye(n_factors) + (F.T @ F) / psi[j]
            cov_L = np.linalg.inv(prec_L)
            mean_L = cov_L @ (F.T @ X_c[:, j]) / psi[j]
            Lambda[j, :] = rng.multivariate_normal(mean_L, cov_L)

        # --- Sample F | Lambda, psi ---
        Psi_inv = np.diag(1.0 / psi)
        prec_F = np.eye(n_factors) + Lambda.T @ Psi_inv @ Lambda
        cov_F = np.linalg.inv(prec_F)
        for i in range(n):
            mean_F = cov_F @ (Lambda.T @ Psi_inv @ X_c[i, :])
            F[i, :] = rng.multivariate_normal(mean_F, cov_F)

        # --- Sample psi | F, Lambda ---
        residuals = X_c - F @ Lambda.T
        for j in range(p):
            a_post = 1.0 + n / 2.0
            b_post = 1.0 + 0.5 * np.sum(residuals[:, j] ** 2)
            psi[j] = 1.0 / rng.gamma(a_post, 1.0 / b_post)

        if it >= burn_in:
            idx = it - burn_in
            Lambda_samples[idx] = Lambda.copy()
            F_samples[idx] = F.copy()
            psi_samples[idx] = psi.copy()

    # Posterior summaries
    loadings_mean = Lambda_samples.mean(axis=0)
    loadings_std = Lambda_samples.std(axis=0)
    scores_mean = F_samples.mean(axis=0)
    noise_variance = psi_samples.mean(axis=0)

    # Explained variance with credible intervals
    total_var = np.var(X_c, axis=0).sum()
    ev_samples = np.zeros((n_samples, n_factors))
    for s in range(n_samples):
        for f in range(n_factors):
            col_var = np.var(F_samples[s, :, f]) * np.sum(Lambda_samples[s, :, f] ** 2)
            ev_samples[s, f] = col_var / total_var if total_var > 0 else 0.0

    explained_variance = ev_samples.mean(axis=0)
    ev_ci = np.column_stack([
        np.percentile(ev_samples, 2.5, axis=0),
        np.percentile(ev_samples, 97.5, axis=0),
    ])

    return BayesianFactorModelResult(
        loadings_mean=loadings_mean,
        loadings_std=loadings_std,
        scores_mean=scores_mean,
        explained_variance=explained_variance,
        explained_variance_ci=ev_ci,
        noise_variance=noise_variance,
        n_factors=n_factors,
    )


# ---------------------------------------------------------------------------
# Bayesian online changepoint detection (Adams-MacKay)
# ---------------------------------------------------------------------------


@dataclass
class BayesianChangepointResult:
    """Result container for Bayesian online changepoint detection.

    Attributes:
        run_length_probs: Matrix of run-length probabilities, shape
            ``(T, T + 1)`` where entry ``[t, r]`` is the probability
            that at time ``t`` the current run length is ``r``.
        changepoint_posterior: Posterior probability of a changepoint at
            each time step, shape ``(T,)``.
        most_likely_changepoints: Indices where the changepoint
            posterior exceeds ``threshold``.
        hazard_rate: The constant hazard rate (1 / expected run length)
            used.
    """

    run_length_probs: np.ndarray
    changepoint_posterior: np.ndarray
    most_likely_changepoints: np.ndarray
    hazard_rate: float


def bayesian_changepoint(
    data: np.ndarray,
    hazard: float = 1.0 / 100.0,
    mu_0: float = 0.0,
    kappa_0: float = 1.0,
    alpha_0: float = 1.0,
    beta_0: float = 1.0,
    threshold: float = 0.3,
) -> BayesianChangepointResult:
    """Bayesian online changepoint detection (Adams & MacKay, 2007).

    This algorithm processes observations one at a time and maintains a
    probability distribution over the *run length* -- how long since the
    last changepoint.  At each new observation the run-length
    distribution is updated in O(t) time, making the algorithm
    efficient for streaming data.

    The underlying model assumes each segment has data drawn from a
    Normal distribution with unknown mean and variance, using a
    Normal-Inverse-Gamma conjugate prior that is re-initialised at each
    changepoint.

    **When to use this**: Use this for detecting structural breaks in
    financial time series (regime changes, volatility shifts, mean
    reversions).  Unlike classical CUSUM or Bai-Perron tests, this gives
    a full posterior probability of a changepoint at each time step
    rather than a binary yes/no answer.

    Args:
        data: Univariate time series of shape ``(T,)``.
        hazard: Constant hazard rate = 1 / (expected run length).
            ``hazard = 0.01`` means changepoints happen roughly every
            100 time steps.
        mu_0: Prior mean for the Normal component.
        kappa_0: Prior precision scaling (how confident in ``mu_0``).
        alpha_0: Shape parameter for the Inverse-Gamma noise prior.
        beta_0: Scale parameter for the Inverse-Gamma noise prior.
        threshold: Minimum posterior probability to flag a changepoint.

    Returns:
        BayesianChangepointResult with run-length probabilities and
        changepoint posterior.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> data = np.concatenate([rng.normal(0, 1, 100),
        ...                        rng.normal(5, 1, 100)])
        >>> result = bayesian_changepoint(data, hazard=1/50)
        >>> cps = result.most_likely_changepoints
        >>> # Should find a changepoint near index 100
    """
    data = np.asarray(data, dtype=float).ravel()
    T = len(data)

    # Run-length probabilities: R[t, r] = P(r_t = r | x_{1:t})
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    # Sufficient statistics for each run length
    mu = np.full(T + 1, mu_0)
    kappa = np.full(T + 1, kappa_0)
    alpha = np.full(T + 1, alpha_0)
    beta = np.full(T + 1, beta_0)

    for t in range(T):
        x = data[t]

        # Predictive probability under each run length (Student-t)
        df = 2.0 * alpha[: t + 1]
        loc = mu[: t + 1]
        scale = np.sqrt(beta[: t + 1] * (kappa[: t + 1] + 1.0) / (kappa[: t + 1] * alpha[: t + 1]))
        pred = stats.t.pdf(x, df=df, loc=loc, scale=scale)

        # Growth probability (run length increases by 1)
        growth = R[t, : t + 1] * pred * (1.0 - hazard)

        # Changepoint probability (run length resets to 0)
        cp = np.sum(R[t, : t + 1] * pred * hazard)

        # Update run-length distribution
        R[t + 1, 1 : t + 2] = growth
        R[t + 1, 0] = cp

        # Normalise
        evidence = R[t + 1, : t + 2].sum()
        if evidence > 0:
            R[t + 1, : t + 2] /= evidence

        # Update sufficient statistics for the new run lengths
        new_mu = np.empty(t + 2)
        new_kappa = np.empty(t + 2)
        new_alpha = np.empty(t + 2)
        new_beta = np.empty(t + 2)

        # Run length 0: reset to prior
        new_mu[0] = mu_0
        new_kappa[0] = kappa_0
        new_alpha[0] = alpha_0
        new_beta[0] = beta_0

        # Run lengths 1 .. t+1: Bayesian update
        old_mu = mu[: t + 1]
        old_kappa = kappa[: t + 1]
        old_alpha = alpha[: t + 1]
        old_beta = beta[: t + 1]

        new_kappa[1 : t + 2] = old_kappa + 1.0
        new_mu[1 : t + 2] = (old_kappa * old_mu + x) / new_kappa[1 : t + 2]
        new_alpha[1 : t + 2] = old_alpha + 0.5
        new_beta[1 : t + 2] = (
            old_beta
            + 0.5 * old_kappa * (x - old_mu) ** 2 / new_kappa[1 : t + 2]
        )

        mu = new_mu
        kappa = new_kappa
        alpha = new_alpha
        beta = new_beta

    # Changepoint detection via MAP run-length drops.
    #
    # The raw P(r_t = 0) is always approximately equal to the hazard rate
    # when a single run length dominates (a well-known property of the
    # Adams-MacKay algorithm).  Instead, we detect changepoints by
    # looking at the MAP (most probable) run length: when it drops
    # sharply from a large value back toward zero, a changepoint has
    # occurred.
    map_run_length = np.zeros(T, dtype=int)
    for t in range(T):
        map_run_length[t] = int(np.argmax(R[t + 1, : t + 2]))

    # Build a changepoint posterior from run-length drops:
    # P(cp at t) = P(r_t <= short_threshold) where short_threshold
    # captures the probability mass at short run lengths.
    cp_posterior = np.zeros(T)
    for t in range(T):
        short_cutoff = max(3, int(0.05 * (t + 1)))
        cp_posterior[t] = float(np.sum(R[t + 1, :short_cutoff]))

    # Also flag changepoints where the MAP run length drops
    # significantly (drops by more than 50% in one step)
    cp_from_map = np.zeros(T, dtype=bool)
    for t in range(1, T):
        if map_run_length[t] < map_run_length[t - 1] * 0.5 and map_run_length[t - 1] > 5:
            cp_from_map[t] = True

    most_likely = np.where(cp_from_map | (cp_posterior > threshold))[0]

    return BayesianChangepointResult(
        run_length_probs=R[1:, :],
        changepoint_posterior=cp_posterior,
        most_likely_changepoints=most_likely,
        hazard_rate=hazard,
    )


# ---------------------------------------------------------------------------
# Enhanced Bayesian portfolio (Black-Litterman with full posterior)
# ---------------------------------------------------------------------------


@dataclass
class BayesianPortfolioBLResult:
    """Result container for the enhanced Black-Litterman portfolio.

    Attributes:
        posterior_mean: Posterior mean of expected returns, shape
            ``(n_assets,)``.
        posterior_cov: Posterior covariance of returns, shape
            ``(n_assets, n_assets)``.
        weights_mean: Mean of posterior optimal portfolio weights.
        weights_std: Standard deviation of posterior optimal weights.
        weights_ci: 95 % credible intervals for each weight, shape
            ``(n_assets, 2)``.
        weight_samples: Raw posterior weight samples, shape
            ``(n_samples, n_assets)``.
    """

    posterior_mean: np.ndarray
    posterior_cov: np.ndarray
    weights_mean: np.ndarray
    weights_std: np.ndarray
    weights_ci: np.ndarray
    weight_samples: np.ndarray


def bayesian_portfolio_bl(
    returns: np.ndarray,
    views: np.ndarray | None = None,
    view_confidences: np.ndarray | None = None,
    P: np.ndarray | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    n_samples: int = 5_000,
    rng_seed: int = 42,
) -> BayesianPortfolioBLResult:
    """Black-Litterman model with full Bayesian posterior sampling.

    The Black-Litterman model combines a market equilibrium prior
    (implied by CAPM or equal-weighted) with investor views to produce
    a posterior distribution of expected returns.  This implementation
    goes beyond the standard BL point estimate: it samples from the
    full posterior to give uncertainty-aware optimal weights.

    **When to use this**: Use BL when you have subjective views about
    expected returns or relative performance and want to combine them
    with market-implied priors.  The Bayesian extension is especially
    useful for constructing robust portfolios with weight confidence
    intervals.

    Args:
        returns: Historical return matrix, shape ``(n_periods, n_assets)``.
        views: View vector ``q``, shape ``(n_views,)``.  If None, the
            posterior equals the prior (equilibrium returns).
        view_confidences: Diagonal of the view uncertainty matrix
            ``Omega``, shape ``(n_views,)``.  Smaller values mean more
            confident views.  If None, defaults to
            ``tau * diag(P @ Sigma @ P^T)``.
        P: Pick matrix mapping assets to views, shape
            ``(n_views, n_assets)``.  Required if ``views`` is given.
        tau: Scaling factor for the uncertainty in the prior mean.
            Typical range 0.01 -- 0.10.
        risk_aversion: Risk-aversion parameter delta for the quadratic
            utility.  Default ``2.5``.
        n_samples: Number of posterior samples for weight uncertainty.
        rng_seed: Random seed for reproducibility.

    Returns:
        BayesianPortfolioBLResult with posterior mean/covariance of
        returns and weight uncertainty.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> returns = rng.normal(0.001, 0.02, (252, 3))
        >>> # View: asset 0 will outperform asset 1 by 2 % annualised
        >>> P = np.array([[1, -1, 0]])
        >>> q = np.array([0.02 / 252])
        >>> result = bayesian_portfolio_bl(returns, views=q, P=P)
    """
    rng = np.random.default_rng(rng_seed)
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    n, p = returns.shape

    Sigma = np.cov(returns, rowvar=False, ddof=1)
    if Sigma.ndim == 0:
        Sigma = Sigma.reshape(1, 1)

    # Equilibrium returns (reverse optimisation with equal-weighted mkt portfolio)
    w_mkt = np.ones(p) / p
    pi = risk_aversion * Sigma @ w_mkt  # equilibrium excess returns

    # Prior for mu: N(pi, tau * Sigma)
    tau_Sigma = tau * Sigma

    if views is not None:
        views = np.asarray(views, dtype=float).ravel()
        if P is None:
            raise ValueError("Pick matrix P is required when views are provided.")
        P = np.asarray(P, dtype=float)
        if P.ndim == 1:
            P = P.reshape(1, -1)
        n_views = len(views)

        if view_confidences is None:
            Omega = np.diag(np.diag(tau * P @ Sigma @ P.T))
        else:
            Omega = np.diag(np.asarray(view_confidences, dtype=float).ravel())

        # Posterior mean and covariance of mu (Black-Litterman formula)
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        Omega_inv = np.linalg.inv(Omega)
        post_prec = tau_Sigma_inv + P.T @ Omega_inv @ P
        post_cov_mu = np.linalg.inv(post_prec)
        post_mean_mu = post_cov_mu @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ views)
    else:
        post_mean_mu = pi
        post_cov_mu = tau_Sigma

    # Posterior covariance of returns (for portfolio optimisation)
    post_cov_returns = Sigma + post_cov_mu

    # Sample from posterior of mu, then compute optimal weights
    post_cov_mu_sym = 0.5 * (post_cov_mu + post_cov_mu.T) + 1e-10 * np.eye(p)

    weight_samples = np.zeros((n_samples, p))
    Sigma_reg = Sigma + 1e-10 * np.eye(p)
    Sigma_inv = np.linalg.inv(Sigma_reg)

    for i in range(n_samples):
        mu_sample = rng.multivariate_normal(post_mean_mu, post_cov_mu_sym)
        # Optimal weights: w* = (1/delta) Sigma^{-1} mu
        w = Sigma_inv @ mu_sample / risk_aversion
        w_sum = np.sum(w)
        if abs(w_sum) > 1e-12:
            w = w / w_sum
        else:
            w = np.ones(p) / p
        weight_samples[i] = w

    weights_mean = weight_samples.mean(axis=0)
    weights_std = weight_samples.std(axis=0)
    weights_ci = np.column_stack([
        np.percentile(weight_samples, 2.5, axis=0),
        np.percentile(weight_samples, 97.5, axis=0),
    ])

    return BayesianPortfolioBLResult(
        posterior_mean=post_mean_mu,
        posterior_cov=post_cov_returns,
        weights_mean=weights_mean,
        weights_std=weights_std,
        weights_ci=weights_ci,
        weight_samples=weight_samples,
    )


# ---------------------------------------------------------------------------
# Bayesian stochastic volatility
# ---------------------------------------------------------------------------


@dataclass
class BayesianVolatilityResult:
    """Result container for the Bayesian stochastic volatility model.

    Attributes:
        vol_mean: Posterior mean of the time-varying volatility path,
            shape ``(T,)``.
        vol_ci_lower: Lower 2.5 % quantile of the volatility path.
        vol_ci_upper: Upper 97.5 % quantile of the volatility path.
        mu_posterior: Posterior samples of the log-vol mean level.
        phi_posterior: Posterior samples of the AR(1) persistence
            parameter.
        sigma_eta_posterior: Posterior samples of the log-vol innovation
            std.
    """

    vol_mean: np.ndarray
    vol_ci_lower: np.ndarray
    vol_ci_upper: np.ndarray
    mu_posterior: np.ndarray
    phi_posterior: np.ndarray
    sigma_eta_posterior: np.ndarray


def bayesian_volatility(
    returns: np.ndarray,
    n_samples: int = 2_000,
    burn_in: int = 1_000,
    rng_seed: int = 42,
) -> BayesianVolatilityResult:
    """Bayesian stochastic volatility model via MCMC.

    Estimates a time-varying volatility path using the standard
    stochastic volatility (SV) model:

        y_t = exp(h_t / 2) * epsilon_t,   epsilon_t ~ N(0, 1)
        h_t = mu + phi * (h_{t-1} - mu) + eta_t,   eta_t ~ N(0, sigma_eta^2)

    where ``h_t`` is the log-variance at time ``t``, ``mu`` is the
    long-run mean of log-variance, ``phi`` is the persistence
    (typically close to 1 for financial data), and ``sigma_eta`` is
    the volatility-of-volatility.

    The model is estimated using a Metropolis-within-Gibbs sampler:
    the log-volatility path ``h`` is sampled block-wise, and the
    parameters ``(mu, phi, sigma_eta)`` are sampled from their
    conditionals.

    **When to use this**: Use this instead of GARCH when you believe
    volatility evolves as a latent (unobserved) state rather than
    being a deterministic function of past returns.  The Bayesian
    approach gives full uncertainty bands on the volatility path.

    Args:
        returns: Return series of shape ``(T,)``.  Should be
            de-meaned (or close to zero-mean).
        n_samples: Number of MCMC samples to keep after burn-in.
        burn_in: Number of initial samples to discard.
        rng_seed: Random seed.

    Returns:
        BayesianVolatilityResult with posterior volatility path and
        parameter posteriors.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> vol = np.exp(0.5 * np.cumsum(rng.normal(0, 0.1, 200)))
        >>> returns = vol * rng.normal(size=200)
        >>> result = bayesian_volatility(returns, n_samples=500, burn_in=200)
        >>> print(result.vol_mean.shape)  # (200,)
    """
    rng = np.random.default_rng(rng_seed)
    returns = np.asarray(returns, dtype=float).ravel()
    T = len(returns)

    # Replace exact zeros to avoid log(0)
    y = returns.copy()
    y[y == 0] = 1e-8

    # Initialise log-volatility from log(y^2)
    log_y2 = np.log(y ** 2 + 1e-8)
    h = log_y2.copy()  # initial log-vol

    # Initialise parameters
    mu = np.mean(h)
    phi = 0.95
    sigma_eta = 0.2

    # Storage
    total = n_samples + burn_in
    h_samples = np.zeros((n_samples, T))
    mu_samples = np.zeros(n_samples)
    phi_samples = np.zeros(n_samples)
    sigma_eta_samples = np.zeros(n_samples)

    for it in range(total):
        # --- Sample h (single-site Metropolis) ---
        for t in range(T):
            # Propose from random walk
            h_prop = h[t] + rng.normal(0, 0.5)

            # Log-likelihood contribution at t
            ll_curr = -0.5 * h[t] - 0.5 * y[t] ** 2 * np.exp(-h[t])
            ll_prop = -0.5 * h_prop - 0.5 * y[t] ** 2 * np.exp(-h_prop)

            # Log-prior contribution from AR(1) transitions
            lp_curr = 0.0
            lp_prop = 0.0

            if t == 0:
                # Stationary distribution prior for h_0
                var_stat = sigma_eta ** 2 / max(1.0 - phi ** 2, 1e-8)
                lp_curr += -0.5 * (h[t] - mu) ** 2 / var_stat
                lp_prop += -0.5 * (h_prop - mu) ** 2 / var_stat
            else:
                lp_curr += -0.5 * (h[t] - mu - phi * (h[t - 1] - mu)) ** 2 / (sigma_eta ** 2)
                lp_prop += -0.5 * (h_prop - mu - phi * (h[t - 1] - mu)) ** 2 / (sigma_eta ** 2)

            if t < T - 1:
                lp_curr += -0.5 * (h[t + 1] - mu - phi * (h[t] - mu)) ** 2 / (sigma_eta ** 2)
                lp_prop += -0.5 * (h[t + 1] - mu - phi * (h_prop - mu)) ** 2 / (sigma_eta ** 2)

            log_alpha = (ll_prop + lp_prop) - (ll_curr + lp_curr)
            if np.log(rng.uniform()) < log_alpha:
                h[t] = h_prop

        # --- Sample mu | h, phi, sigma_eta ---
        h_diff = h[1:] - phi * h[:-1]
        # mu contribution: h_t = mu + phi*(h_{t-1} - mu) + eta
        # => h_t = mu*(1-phi) + phi*h_{t-1} + eta
        # So h_diff = h_t - phi*h_{t-1} ~ N(mu*(1-phi), sigma_eta^2)
        n_trans = T - 1
        prior_prec_mu = 1.0 / 10.0  # prior N(0, 10)
        data_prec_mu = n_trans * (1 - phi) ** 2 / (sigma_eta ** 2)
        post_prec_mu = prior_prec_mu + data_prec_mu
        post_mean_mu = (prior_prec_mu * 0.0 + (1 - phi) * np.sum(h_diff) / (sigma_eta ** 2)) / post_prec_mu
        mu = rng.normal(post_mean_mu, 1.0 / np.sqrt(post_prec_mu))

        # --- Sample phi | h, mu, sigma_eta (truncated to (-1, 1)) ---
        x_t = h[:-1] - mu
        y_t = h[1:] - mu
        x_sum = np.sum(x_t ** 2)
        if x_sum > 1e-12:
            post_var_phi = sigma_eta ** 2 / x_sum
            post_mean_phi = np.sum(x_t * y_t) / x_sum
            phi_prop = rng.normal(post_mean_phi, np.sqrt(post_var_phi))
            if abs(phi_prop) < 1.0:
                phi = phi_prop

        # --- Sample sigma_eta^2 | h, mu, phi ---
        residuals = y_t - phi * x_t
        a_post = 1.0 + n_trans / 2.0
        b_post = 1.0 + 0.5 * np.sum(residuals ** 2)
        sigma_eta_sq = 1.0 / rng.gamma(a_post, 1.0 / b_post)
        sigma_eta = np.sqrt(max(sigma_eta_sq, 1e-10))

        if it >= burn_in:
            idx = it - burn_in
            h_samples[idx] = h.copy()
            mu_samples[idx] = mu
            phi_samples[idx] = phi
            sigma_eta_samples[idx] = sigma_eta

    # Posterior volatility path
    vol_samples = np.exp(h_samples / 2.0)
    vol_mean = vol_samples.mean(axis=0)
    vol_ci_lower = np.percentile(vol_samples, 2.5, axis=0)
    vol_ci_upper = np.percentile(vol_samples, 97.5, axis=0)

    return BayesianVolatilityResult(
        vol_mean=vol_mean,
        vol_ci_lower=vol_ci_lower,
        vol_ci_upper=vol_ci_upper,
        mu_posterior=mu_samples,
        phi_posterior=phi_samples,
        sigma_eta_posterior=sigma_eta_samples,
    )


# ---------------------------------------------------------------------------
# Bayesian cointegration test
# ---------------------------------------------------------------------------


@dataclass
class BayesianCointegrationResult:
    """Result container for the Bayesian cointegration test.

    Attributes:
        prob_cointegrated: Posterior probability that the two series are
            cointegrated (based on the residual unit-root test).
        cointegrating_vector_mean: Posterior mean of the cointegrating
            vector ``[1, -beta]`` (i.e. beta is the slope coefficient).
        cointegrating_vector_std: Posterior std of beta.
        residual_adf_samples: Posterior samples of the ADF-like
            autoregressive coefficient on the residuals.
        spread_mean: Posterior mean of the spread ``y - beta * x``.
    """

    prob_cointegrated: float
    cointegrating_vector_mean: float
    cointegrating_vector_std: float
    residual_adf_samples: np.ndarray
    spread_mean: np.ndarray


def bayesian_cointegration(
    y: np.ndarray,
    x: np.ndarray,
    n_samples: int = 5_000,
    rng_seed: int = 42,
) -> BayesianCointegrationResult:
    """Bayesian cointegration test between two time series.

    Tests whether ``y`` and ``x`` are cointegrated by:

    1. Estimating the cointegrating regression ``y = alpha + beta * x + e``
       using Bayesian linear regression (to get a posterior for beta).
    2. Testing the residuals ``e_t`` for stationarity using a Bayesian
       version of the ADF test: ``Delta e_t = rho * e_{t-1} + u_t``.
       If ``rho < 0``, the residuals are mean-reverting (cointegrated).

    The posterior probability of cointegration is estimated as
    ``P(rho < 0 | data)``.

    **When to use this**: Use this for pairs trading (testing if two
    assets share a long-run equilibrium) or for constructing
    cointegrated portfolios.  The Bayesian approach gives a probability
    rather than a p-value, which is more natural for decision-making.

    Args:
        y: First time series, shape ``(T,)``.
        x: Second time series, shape ``(T,)``.
        n_samples: Number of posterior samples for the ADF coefficient.
        rng_seed: Random seed.

    Returns:
        BayesianCointegrationResult with the posterior probability
        of cointegration and the cointegrating vector.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = np.cumsum(rng.normal(size=300))
        >>> y = 0.8 * x + rng.normal(0, 0.5, 300)  # cointegrated
        >>> result = bayesian_cointegration(y, x)
        >>> print(f"P(cointegrated) = {result.prob_cointegrated:.2f}")
    """
    rng = np.random.default_rng(rng_seed)
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    T = len(y)

    # Step 1: Bayesian regression  y = alpha + beta * x + e
    X_reg = np.column_stack([np.ones(T), x])
    reg_result = bayesian_linear_regression(y, X_reg)
    beta_mean = reg_result.posterior_mean[1]
    beta_std = np.sqrt(reg_result.posterior_cov_unscaled[1, 1] * reg_result.sigma2_mean)

    # Sample beta from posterior (marginal Student-t)
    df_beta = 2.0 * reg_result.a_n
    scale_beta = np.sqrt(reg_result.posterior_cov_unscaled[1, 1] * reg_result.b_n / reg_result.a_n)
    beta_samples = stats.t.rvs(
        df=df_beta, loc=beta_mean, scale=scale_beta, size=n_samples, random_state=rng
    )

    # Step 2: For each beta sample, compute residuals and test for unit root
    rho_samples = np.zeros(n_samples)
    for i in range(n_samples):
        resid = y - reg_result.posterior_mean[0] - beta_samples[i] * x
        # ADF-like regression: Delta_e_t = rho * e_{t-1} + u_t
        de = np.diff(resid)
        e_lag = resid[:-1]
        if np.var(e_lag) < 1e-15:
            rho_samples[i] = 0.0
            continue
        # Simple OLS for rho
        rho_ols = np.sum(de * e_lag) / np.sum(e_lag ** 2)
        resid_u = de - rho_ols * e_lag
        sigma2_u = np.var(resid_u, ddof=1) if len(resid_u) > 1 else 1e-6
        var_rho = sigma2_u / max(np.sum(e_lag ** 2), 1e-12)
        rho_samples[i] = rng.normal(rho_ols, np.sqrt(max(var_rho, 1e-12)))

    prob_coint = float(np.mean(rho_samples < 0))

    # Spread under posterior mean beta
    spread_mean = y - reg_result.posterior_mean[0] - beta_mean * x

    return BayesianCointegrationResult(
        prob_cointegrated=prob_coint,
        cointegrating_vector_mean=float(beta_mean),
        cointegrating_vector_std=float(beta_std),
        residual_adf_samples=rho_samples,
        spread_mean=spread_mean,
    )


# ---------------------------------------------------------------------------
# Bayesian model comparison
# ---------------------------------------------------------------------------


def model_comparison(
    y: np.ndarray,
    X_list: list[np.ndarray],
    model_names: list[str] | None = None,
) -> "pd.DataFrame":
    """Bayesian model comparison via marginal likelihood and information criteria.

    Fits each candidate model using ``bayesian_linear_regression`` and
    computes:

    - **Log marginal likelihood** (for Bayes factor computation).
    - **WAIC** (Widely Applicable Information Criterion) -- a Bayesian
      analogue of AIC that accounts for the effective number of
      parameters.
    - **LOO-CV** (Leave-One-Out Cross-Validation) approximation via
      importance sampling.
    - **Bayes factor** relative to the best model.

    Models are ranked by log marginal likelihood (higher is better).

    **When to use this**: Use this when choosing between competing
    regression specifications (e.g., which factors belong in a return
    model).  The marginal likelihood automatically penalises model
    complexity (Occam's razor), unlike AIC/BIC which use fixed
    penalties.

    Args:
        y: Response vector of shape ``(n,)``.
        X_list: List of design matrices, one per model.  Each has shape
            ``(n, k_m)`` where ``k_m`` can differ.
        model_names: Optional names for each model.  Defaults to
            ``['model_0', 'model_1', ...]``.

    Returns:
        pd.DataFrame with columns ``log_marginal_likelihood``,
        ``waic``, ``loo_cv``, ``bayes_factor``, ``rank``, sorted by
        rank (best first).

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 200
        >>> x1 = rng.normal(size=n)
        >>> x2 = rng.normal(size=n)
        >>> noise = rng.normal(size=n)
        >>> y = 1.0 + 2.0 * x1 + rng.normal(0, 0.5, n)
        >>> X_good = np.column_stack([np.ones(n), x1])
        >>> X_bad = np.column_stack([np.ones(n), x2])
        >>> df = model_comparison(y, [X_good, X_bad], ["good", "bad"])
        >>> assert df.index[0] == "good"  # good model ranked first
    """
    import pandas as pd

    y = np.asarray(y, dtype=float).ravel()
    n_models = len(X_list)

    if model_names is None:
        model_names = [f"model_{i}" for i in range(n_models)]

    results = []
    for i, X in enumerate(X_list):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = len(y)

        res = bayesian_linear_regression(y, X)

        # WAIC approximation
        # Compute pointwise log-likelihood at posterior mean
        y_hat = X @ res.posterior_mean
        sigma2_est = res.sigma2_mean
        log_lik = -0.5 * np.log(2 * np.pi * sigma2_est) - 0.5 * (y - y_hat) ** 2 / sigma2_est

        # p_waic: effective number of parameters (variance of log-lik)
        # Approximate using posterior predictive variance
        p_waic = float(res.n_features)  # simple approximation
        lppd = float(np.sum(log_lik))
        waic = -2.0 * (lppd - p_waic)

        # LOO-CV approximation (Pareto-smoothed importance sampling idea,
        # simplified to analytical leave-one-out for linear regression)
        H = X @ np.linalg.inv(X.T @ X + 1e-10 * np.eye(X.shape[1])) @ X.T
        h_diag = np.diag(H)
        resid = y - y_hat
        loo_resid = resid / np.maximum(1.0 - h_diag, 1e-8)
        loo_cv = float(np.mean(loo_resid ** 2))

        results.append({
            "log_marginal_likelihood": res.log_marginal_likelihood,
            "waic": waic,
            "loo_cv": loo_cv,
        })

    # Bayes factors relative to best model
    log_mls = np.array([r["log_marginal_likelihood"] for r in results])
    best_log_ml = np.max(log_mls)
    for i, r in enumerate(results):
        diff = log_mls[i] - best_log_ml
        r["bayes_factor"] = float(np.exp(np.clip(diff, -500, 500)))

    # Rank by log marginal likelihood (higher is better)
    ranking = np.argsort(-log_mls)
    for i, r in enumerate(results):
        r["rank"] = int(np.where(ranking == i)[0][0]) + 1

    df = pd.DataFrame(results, index=model_names)
    df = df.sort_values("rank")

    return df


# ---------------------------------------------------------------------------
# Bayesian regime inference — bayes/ → regimes/ bridge
# ---------------------------------------------------------------------------


def bayesian_regime_inference(
    returns: np.ndarray,
    n_regimes: int = 2,
    n_samples: int = 2000,
    seed: int | None = None,
) -> "RegimeResult":
    """Bayesian regime detection using conjugate priors.

    Alternative to frequentist HMM (``wraquant.regimes.base.detect_regimes``)
    that provides full posterior uncertainty on regime assignments and
    transition probabilities.  Uses a Gibbs sampler with conjugate
    Normal-Inverse-Gamma priors for each regime's mean and variance, and
    a Dirichlet prior for transition probabilities.

    This bridges ``bayes`` and ``regimes`` by returning a ``RegimeResult``
    (the standard container from ``wraquant.regimes.base``), so the output
    can be fed directly into downstream regime-aware portfolio or risk
    functions.

    The generative model is:

        s_t | s_{t-1} ~ Categorical(transition_matrix[s_{t-1}])
        r_t | s_t     ~ N(mu_{s_t}, sigma^2_{s_t})

    with priors:

        mu_k          ~ N(0, prior_var)
        sigma^2_k     ~ InvGamma(alpha_0, beta_0)
        transition[k] ~ Dirichlet(1, ..., 1)

    Parameters:
        returns: Return series, shape ``(T,)``.
        n_regimes: Number of regimes to detect (default 2).
        n_samples: Number of Gibbs samples to draw (after a burn-in
            of ``n_samples // 2``).  More samples give smoother
            posterior estimates.
        seed: Random seed for reproducibility.

    Returns:
        RegimeResult with:

        - ``states`` -- MAP (most probable) regime at each time step.
        - ``probabilities`` -- Posterior regime probabilities (T, K).
        - ``transition_matrix`` -- Posterior mean transition matrix.
        - ``means`` -- Posterior mean returns per regime.
        - ``covariances`` -- Posterior variance per regime.
        - ``statistics`` -- Per-regime summary statistics.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> # Two-regime returns: low vol and high vol
        >>> regime = np.concatenate([np.zeros(200), np.ones(200)]).astype(int)
        >>> mu = np.array([0.001, -0.002])
        >>> sigma = np.array([0.01, 0.03])
        >>> returns = rng.normal(mu[regime], sigma[regime])
        >>> result = bayesian_regime_inference(returns, n_regimes=2, seed=42)
        >>> result.n_regimes
        2
        >>> result.states.shape
        (400,)

    Notes:
        The Gibbs sampler alternates between:
        1. Sampling regime states via forward-filtering backward-sampling.
        2. Sampling regime parameters (mu, sigma^2) from conjugate posteriors.
        3. Sampling transition probabilities from Dirichlet posteriors.

    References:
        - Chib (1996), "Calculating posterior distributions and modal
          estimates in Markov mixture models"
        - Hamilton (1989), "A new approach to the economic analysis of
          nonstationary time series"

    See Also:
        wraquant.regimes.base.detect_regimes: Frequentist regime detection.
        wraquant.regimes.base.RegimeResult: Standard result container.
        bayesian_changepoint: Changepoint detection (different model).
    """
    import pandas as pd

    from wraquant.regimes.base import RegimeResult

    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float).ravel()
    T = len(returns)
    K = n_regimes

    burn_in = n_samples // 2
    total_iter = n_samples + burn_in

    # --- Initialise parameters ---
    # K-means-like initialisation: sort returns and split into K groups
    sorted_r = np.sort(returns)
    chunk = T // K
    mu = np.array([np.mean(sorted_r[i * chunk : (i + 1) * chunk]) for i in range(K)])
    sigma2 = np.array([np.var(sorted_r[i * chunk : (i + 1) * chunk], ddof=1) + 1e-8 for i in range(K)])

    # Transition matrix: high self-persistence
    trans = np.full((K, K), 0.05 / max(K - 1, 1))
    np.fill_diagonal(trans, 0.95)
    trans = trans / trans.sum(axis=1, keepdims=True)

    # States: initialise from closest mean
    states = np.zeros(T, dtype=int)
    for t in range(T):
        states[t] = int(np.argmin([abs(returns[t] - mu[k]) for k in range(K)]))

    # Prior hyperparameters
    alpha_0 = 2.0  # InvGamma shape
    beta_0 = 0.001  # InvGamma scale
    mu_prior = 0.0
    mu_prior_var = 1.0
    dirichlet_alpha = np.ones(K)  # symmetric Dirichlet

    # Storage for posterior samples
    state_samples = np.zeros((n_samples, T), dtype=int)
    trans_samples = np.zeros((n_samples, K, K))
    mu_samples = np.zeros((n_samples, K))
    sigma2_samples = np.zeros((n_samples, K))

    for it in range(total_iter):
        # --- Step 1: Sample states (forward-filtering backward-sampling) ---
        # Forward pass: compute filtered probabilities
        log_alpha = np.zeros((T, K))
        for k in range(K):
            log_alpha[0, k] = -0.5 * np.log(2 * np.pi * sigma2[k]) - 0.5 * (returns[0] - mu[k]) ** 2 / sigma2[k]
        # Normalise
        log_alpha[0] -= np.max(log_alpha[0])
        alpha_filt = np.exp(log_alpha[0])
        alpha_filt /= alpha_filt.sum()
        log_alpha[0] = np.log(alpha_filt + 1e-300)

        for t in range(1, T):
            for k in range(K):
                log_emit = -0.5 * np.log(2 * np.pi * sigma2[k]) - 0.5 * (returns[t] - mu[k]) ** 2 / sigma2[k]
                # Sum over previous states
                log_trans = np.log(trans[:, k] + 1e-300)
                log_alpha[t, k] = log_emit + np.logaddexp.reduce(log_alpha[t - 1] + log_trans)
            # Normalise to prevent underflow
            max_la = np.max(log_alpha[t])
            log_alpha[t] -= max_la
            exp_la = np.exp(log_alpha[t])
            exp_la /= exp_la.sum()
            log_alpha[t] = np.log(exp_la + 1e-300)

        # Backward sampling
        # Sample last state
        prob_T = np.exp(log_alpha[T - 1])
        prob_T /= prob_T.sum()
        states[T - 1] = rng.choice(K, p=prob_T)

        for t in range(T - 2, -1, -1):
            log_prob = log_alpha[t] + np.log(trans[:, states[t + 1]] + 1e-300)
            log_prob -= np.max(log_prob)
            prob = np.exp(log_prob)
            prob /= prob.sum()
            states[t] = rng.choice(K, p=prob)

        # --- Step 2: Sample mu, sigma^2 for each regime ---
        for k in range(K):
            mask = states == k
            n_k = int(mask.sum())
            if n_k < 2:
                # Not enough data; keep current values
                continue
            r_k = returns[mask]
            y_bar = np.mean(r_k)

            # Posterior for mu | sigma^2 (conjugate normal)
            prior_prec = 1.0 / mu_prior_var
            data_prec = n_k / sigma2[k]
            post_prec = prior_prec + data_prec
            post_mean = (prior_prec * mu_prior + data_prec * y_bar) / post_prec
            mu[k] = rng.normal(post_mean, 1.0 / np.sqrt(post_prec))

            # Posterior for sigma^2 (conjugate InvGamma)
            a_post = alpha_0 + n_k / 2.0
            b_post = beta_0 + 0.5 * np.sum((r_k - mu[k]) ** 2)
            sigma2[k] = 1.0 / rng.gamma(a_post, 1.0 / max(b_post, 1e-12))
            sigma2[k] = max(sigma2[k], 1e-10)

        # --- Step 3: Sample transition matrix (Dirichlet posterior) ---
        counts = np.zeros((K, K))
        for t in range(T - 1):
            counts[states[t], states[t + 1]] += 1
        for k in range(K):
            post_alpha = dirichlet_alpha + counts[k]
            trans[k] = rng.dirichlet(post_alpha)

        # Store after burn-in
        if it >= burn_in:
            idx = it - burn_in
            state_samples[idx] = states.copy()
            trans_samples[idx] = trans.copy()
            mu_samples[idx] = mu.copy()
            sigma2_samples[idx] = sigma2.copy()

    # --- Compute posterior summaries ---
    # Posterior regime probabilities: fraction of samples in each regime
    probabilities = np.zeros((T, K))
    for k in range(K):
        probabilities[:, k] = np.mean(state_samples == k, axis=0)

    # MAP states
    map_states = np.argmax(probabilities, axis=1)

    # Re-order by ascending volatility
    post_sigma2_mean = sigma2_samples.mean(axis=0)
    order = np.argsort(post_sigma2_mean)
    state_map = {int(old): new for new, old in enumerate(order)}

    map_states_ordered = np.array([state_map[int(s)] for s in map_states])
    probs_ordered = probabilities[:, order]
    mu_ordered = mu_samples.mean(axis=0)[order]
    sigma2_ordered = post_sigma2_mean[order]

    # Reorder transition matrix
    trans_mean = trans_samples.mean(axis=0)
    trans_ordered = trans_mean[np.ix_(order, order)]

    # Compute per-regime statistics
    records = []
    for k in range(K):
        mask = map_states_ordered == k
        n_k = int(mask.sum())
        r_k = returns[mask]
        records.append({
            "regime": k,
            "mean": float(np.mean(r_k)) if n_k > 0 else 0.0,
            "std": float(np.std(r_k, ddof=1)) if n_k > 1 else 0.0,
            "pct_time": float(n_k / T),
            "n_obs": n_k,
        })
    statistics = pd.DataFrame(records).set_index("regime")

    return RegimeResult(
        states=map_states_ordered,
        probabilities=probs_ordered,
        transition_matrix=trans_ordered,
        n_regimes=K,
        means=mu_ordered,
        covariances=sigma2_ordered,
        statistics=statistics,
        method="bayesian_gibbs",
        model=None,
        metadata={
            "n_samples": n_samples,
            "burn_in": burn_in,
            "mu_samples": mu_samples,
            "sigma2_samples": sigma2_samples,
            "trans_samples": trans_samples,
        },
    )
