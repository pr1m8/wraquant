"""Lévy processes for fat-tailed asset models.

Provides simulation, density evaluation, and calibration for:
- Variance Gamma (VG)
- Normal Inverse Gaussian (NIG)
- CGMY
- Stable Lévy processes

All implementations are pure numpy/scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import kv as bessel_kv  # modified Bessel K_v
from scipy.stats import norm

__all__ = [
    "variance_gamma_pdf",
    "variance_gamma_simulate",
    "nig_pdf",
    "nig_simulate",
    "cgmy_simulate",
    "fit_variance_gamma",
    "fit_nig",
    "levy_stable_simulate",
    "characteristic_function_vg",
]


# ---------------------------------------------------------------------------
# Variance Gamma
# ---------------------------------------------------------------------------

def variance_gamma_pdf(
    x: ArrayLike,
    sigma: float,
    nu: float,
    theta: float,
) -> np.ndarray:
    r"""Evaluate the Variance Gamma probability density function.

    .. math::

        f(x) = \frac{2\,e^{\theta\,x / \sigma^2}}
        {\nu^{1/\nu}\,\sqrt{2\pi}\,\sigma\,\Gamma(1/\nu)}
        \left(\frac{x^2}{2\sigma^2/\nu + \theta^2}\right)^{1/(2\nu) - 1/4}
        K_{1/\nu - 1/2}\!\left(\frac{1}{\sigma^2}
        \sqrt{x^2(2\sigma^2/\nu + \theta^2)}\right)

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF.
    sigma : float
        Volatility of the Brownian motion component (> 0).
    nu : float
        Variance rate of the Gamma subordinator (> 0).
    theta : float
        Drift of the Brownian motion component (controls skewness).

    Returns
    -------
    np.ndarray
        PDF values.
    """
    from scipy.special import gamma as gamma_fn

    x = np.asarray(x, dtype=float)
    sigma2 = sigma ** 2

    # Parameters for the Bessel representation
    lam = 1.0 / nu
    alpha_param = theta / sigma2
    beta_param = np.sqrt(theta ** 2 + 2.0 * sigma2 / nu) / sigma2

    order = lam - 0.5

    abs_x = np.abs(x)
    # Avoid division by zero at x=0
    safe_x = np.where(abs_x > 0, abs_x, 1e-300)

    log_pdf = (
        np.log(2.0)
        + alpha_param * x
        + np.log(safe_x) * (lam - 0.5)
        - (lam * np.log(nu) + 0.5 * np.log(2.0 * np.pi) + np.log(sigma)
           + np.log(gamma_fn(lam)))
        + np.log(np.maximum(bessel_kv(order, beta_param * safe_x), 1e-300))
        - (lam - 0.5) * np.log(sigma2 * beta_param)
    )

    pdf = np.exp(log_pdf)
    # Handle x = 0 gracefully
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    return pdf


def variance_gamma_simulate(
    sigma: float,
    nu: float,
    theta: float,
    n_steps: int,
    dt: float = 1.0 / 252,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a Variance Gamma process via time-changed Brownian motion.

    X(t) = theta * G(t) + sigma * W(G(t))

    where G is a Gamma subordinator with shape = dt/nu and scale = nu.

    Parameters
    ----------
    sigma : float
        Volatility parameter (> 0).
    nu : float
        Variance rate of the Gamma time change (> 0).
    theta : float
        Drift parameter (skewness).
    n_steps : int
        Number of time steps.
    dt : float, optional
        Time increment (default 1/252 for daily).
    seed : int or None, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Cumulative VG process values of length *n_steps + 1* (starts at 0).
    """
    rng = np.random.default_rng(seed)
    shape = dt / nu
    scale = nu
    # Gamma increments
    dg = rng.gamma(shape, scale, size=n_steps)
    # Brownian increments subordinated by Gamma
    dw = rng.standard_normal(n_steps)
    increments = theta * dg + sigma * np.sqrt(dg) * dw
    return np.concatenate([[0.0], np.cumsum(increments)])


def characteristic_function_vg(
    u: ArrayLike,
    sigma: float,
    nu: float,
    theta: float,
) -> np.ndarray:
    r"""Characteristic function of the Variance Gamma distribution at time 1.

    .. math::

        \phi(u) = \left(1 - i\,\theta\,\nu\,u
        + \tfrac{1}{2}\sigma^2\nu\,u^2\right)^{-1/\nu}

    Parameters
    ----------
    u : array_like
        Fourier variable(s).
    sigma : float
        Volatility parameter.
    nu : float
        Variance rate parameter.
    theta : float
        Drift parameter.

    Returns
    -------
    np.ndarray
        Complex-valued characteristic function values.
    """
    u = np.asarray(u, dtype=complex)
    base = 1.0 - 1j * theta * nu * u + 0.5 * sigma ** 2 * nu * u ** 2
    return base ** (-1.0 / nu)


# ---------------------------------------------------------------------------
# Normal Inverse Gaussian
# ---------------------------------------------------------------------------

def nig_pdf(
    x: ArrayLike,
    alpha: float,
    beta: float,
    mu: float,
    delta: float,
) -> np.ndarray:
    r"""Evaluate the Normal Inverse Gaussian probability density function.

    .. math::

        f(x) = \frac{\alpha\,\delta}{\pi}
        \exp\!\left(\delta\sqrt{\alpha^2 - \beta^2}
        + \beta(x - \mu)\right)
        \frac{K_1\!\left(\alpha\sqrt{\delta^2 + (x-\mu)^2}\right)}
        {\sqrt{\delta^2 + (x - \mu)^2}}

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF.
    alpha : float
        Tail heaviness parameter (> 0, alpha > |beta|).
    beta : float
        Skewness parameter (-alpha < beta < alpha).
    mu : float
        Location parameter.
    delta : float
        Scale parameter (> 0).

    Returns
    -------
    np.ndarray
        PDF values.
    """
    x = np.asarray(x, dtype=float)
    gamma_param = np.sqrt(alpha ** 2 - beta ** 2)
    q = np.sqrt(delta ** 2 + (x - mu) ** 2)

    log_pdf = (
        np.log(alpha * delta / np.pi)
        + delta * gamma_param
        + beta * (x - mu)
        + np.log(np.maximum(bessel_kv(1.0, alpha * q), 1e-300))
        - np.log(q)
    )

    pdf = np.exp(log_pdf)
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    return pdf


def nig_simulate(
    alpha: float,
    beta: float,
    mu: float,
    delta: float,
    n_steps: int,
    dt: float = 1.0 / 252,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a Normal Inverse Gaussian process.

    NIG is obtained as a normal variance-mean mixture with an Inverse
    Gaussian subordinator.

    Parameters
    ----------
    alpha : float
        Tail heaviness (> 0, alpha > |beta|).
    beta : float
        Skewness.
    mu : float
        Location (drift).
    delta : float
        Scale (> 0).
    n_steps : int
        Number of time steps.
    dt : float, optional
        Time increment (default 1/252).
    seed : int or None, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Cumulative NIG process of length *n_steps + 1* (starts at 0).
    """
    rng = np.random.default_rng(seed)
    gamma_param = np.sqrt(alpha ** 2 - beta ** 2)

    # Inverse Gaussian increments  IG(delta*dt, delta*dt*gamma)
    ig_mu_param = delta * dt / gamma_param if gamma_param > 0 else delta * dt
    ig_lambda = (delta * dt) ** 2

    # Simulate IG via the standard method
    ig = _inverse_gaussian_sample(rng, ig_mu_param, ig_lambda, n_steps)

    # Normal variance-mean mixture
    z = rng.standard_normal(n_steps)
    increments = mu * dt + beta * ig + np.sqrt(ig) * z

    return np.concatenate([[0.0], np.cumsum(increments)])


def _inverse_gaussian_sample(
    rng: np.random.Generator,
    mu_ig: float,
    lam: float,
    n: int,
) -> np.ndarray:
    """Sample from Inverse Gaussian(mu, lambda) using the standard algorithm."""
    v = rng.standard_normal(n)
    y = v ** 2
    x = mu_ig + (mu_ig ** 2 * y) / (2.0 * lam) - (mu_ig / (2.0 * lam)) * np.sqrt(
        4.0 * mu_ig * lam * y + mu_ig ** 2 * y ** 2
    )
    u = rng.uniform(size=n)
    result = np.where(u <= mu_ig / (mu_ig + x), x, mu_ig ** 2 / x)
    return result


# ---------------------------------------------------------------------------
# CGMY
# ---------------------------------------------------------------------------

def cgmy_simulate(
    C: float,
    G: float,
    M: float,
    Y: float,
    n_steps: int,
    dt: float = 1.0 / 252,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a CGMY process via series representation (shot-noise).

    For Y < 0 the process has finite activity and can be approximated
    by compound Poisson.  For 0 < Y < 2 the process has infinite
    activity and is approximated by truncating the small-jump component
    at epsilon and adding a Brownian correction.

    Parameters
    ----------
    C : float
        Overall activity level (> 0).
    G : float
        Rate of exponential decay for negative jumps (> 0).
    M : float
        Rate of exponential decay for positive jumps (> 0).
    Y : float
        Fine structure parameter (Y < 2).
    n_steps : int
        Number of time steps.
    dt : float, optional
        Time increment (default 1/252).
    seed : int or None, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Cumulative CGMY process of length *n_steps + 1* (starts at 0).
    """
    rng = np.random.default_rng(seed)

    if Y < 0:
        # Finite activity: compound Poisson approximation
        return _cgmy_finite_activity(C, G, M, Y, n_steps, dt, rng)
    else:
        # Infinite activity: truncated small jumps + Gaussian correction
        return _cgmy_infinite_activity(C, G, M, Y, n_steps, dt, rng)


def _cgmy_finite_activity(
    C: float,
    G: float,
    M: float,
    Y: float,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """CGMY with finite activity via compound Poisson."""
    from scipy.special import gamma as gamma_fn

    # Total intensity (for finite activity Y < 0)
    intensity = C * (gamma_fn(-Y) * (G ** Y + M ** Y))
    increments = np.zeros(n_steps)

    for i in range(n_steps):
        n_jumps = rng.poisson(intensity * dt)
        if n_jumps > 0:
            # Each jump: choose positive with prob M^Y/(G^Y+M^Y)
            p_pos = M ** Y / (G ** Y + M ** Y)
            signs = rng.binomial(1, p_pos, size=n_jumps)
            sizes = np.empty(n_jumps)
            for k in range(n_jumps):
                if signs[k]:
                    sizes[k] = rng.exponential(1.0 / M)
                else:
                    sizes[k] = -rng.exponential(1.0 / G)
            increments[i] = sizes.sum()

    return np.concatenate([[0.0], np.cumsum(increments)])


def _cgmy_infinite_activity(
    C: float,
    G: float,
    M: float,
    Y: float,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """CGMY with infinite activity via truncation + Gaussian correction."""
    epsilon = 1e-4  # truncation level

    # Approximate: large jumps as compound Poisson, small jumps as Gaussian
    # Tail mass of CGMY Lévy measure: integral_{epsilon}^{inf} C*x^{-1-Y}*e^{-Mx} dx
    # For Y > 0, use power-law tail approximation: C * epsilon^{-Y} * exp(-M*eps) / Y
    if Y > 0:
        intensity_pos = C * epsilon ** (-Y) * np.exp(-M * epsilon) / Y
        intensity_neg = C * epsilon ** (-Y) * np.exp(-G * epsilon) / Y
    else:
        # Y = 0: finite activity, integral = C * E_1(M*epsilon) ~ -C*log(M*eps)
        intensity_pos = C * max(-np.log(M * epsilon), 1.0) if M * epsilon > 0 else C
        intensity_neg = C * max(-np.log(G * epsilon), 1.0) if G * epsilon > 0 else C

    total_intensity = intensity_pos + intensity_neg
    if not np.isfinite(total_intensity) or total_intensity <= 0:
        total_intensity = max(intensity_pos, intensity_neg, 1.0)
    # Ensure finite intensities
    intensity_pos = min(intensity_pos, total_intensity) if np.isfinite(intensity_pos) else total_intensity / 2
    intensity_neg = min(intensity_neg, total_intensity) if np.isfinite(intensity_neg) else total_intensity / 2

    # Variance of small jumps (< epsilon): sigma^2 ~ 2*C*epsilon^(2-Y)/(2-Y)
    small_var = 2.0 * C * epsilon ** (2.0 - Y) / max(2.0 - Y, 0.01) * dt

    increments = np.zeros(n_steps)
    for i in range(n_steps):
        # Large jumps
        n_jumps = rng.poisson(total_intensity * dt)
        if n_jumps > 0:
            p_pos = intensity_pos / total_intensity if total_intensity > 0 else 0.5
            signs = rng.binomial(1, p_pos, size=n_jumps)
            sizes = np.empty(n_jumps)
            for k in range(n_jumps):
                if signs[k]:
                    sizes[k] = epsilon + rng.exponential(1.0 / M)
                else:
                    sizes[k] = -(epsilon + rng.exponential(1.0 / G))
            increments[i] = sizes.sum()

        # Gaussian correction for small jumps
        if small_var > 0:
            increments[i] += rng.normal(0, np.sqrt(small_var))

    return np.concatenate([[0.0], np.cumsum(increments)])


# ---------------------------------------------------------------------------
# Stable Lévy
# ---------------------------------------------------------------------------

def levy_stable_simulate(
    alpha: float,
    beta: float,
    n_steps: int,
    seed: int | None = None,
) -> np.ndarray:
    r"""Simulate a stable Lévy process using the Chambers-Mallows-Stuck method.

    Parameters
    ----------
    alpha : float
        Stability index, 0 < alpha <= 2.
    beta : float
        Skewness parameter, -1 <= beta <= 1.
    n_steps : int
        Number of increments.
    seed : int or None, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Cumulative process values of length *n_steps + 1* (starts at 0).
    """
    rng = np.random.default_rng(seed)
    increments = _stable_random(rng, alpha, beta, n_steps)
    return np.concatenate([[0.0], np.cumsum(increments)])


def _stable_random(
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    n: int,
) -> np.ndarray:
    """Chambers-Mallows-Stuck algorithm for stable random variates."""
    V = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    W = rng.exponential(1.0, size=n)

    if alpha == 1.0:
        X = (
            (np.pi / 2 + beta * V) * np.tan(V)
            - beta * np.log(np.pi / 2 * W * np.cos(V) / (np.pi / 2 + beta * V))
        )
    else:
        zeta = -beta * np.tan(np.pi * alpha / 2)
        xi = np.arctan(-zeta) / alpha
        s = (1.0 + zeta ** 2) ** (1.0 / (2.0 * alpha))
        X = s * (
            np.sin(alpha * (V + xi))
            / np.cos(V) ** (1.0 / alpha)
            * (np.cos(V - alpha * (V + xi)) / W) ** ((1.0 - alpha) / alpha)
        )
    return X


# ---------------------------------------------------------------------------
# Fitting / calibration
# ---------------------------------------------------------------------------

def fit_variance_gamma(
    returns: ArrayLike,
) -> dict[str, float]:
    """Fit a Variance Gamma distribution to return data via MLE.

    Parameters
    ----------
    returns : array_like
        Observed returns.

    Returns
    -------
    dict
        ``sigma`` – fitted volatility.
        ``nu``    – fitted variance rate.
        ``theta`` – fitted drift.
        ``log_likelihood`` – maximised log-likelihood.
    """
    returns = np.asarray(returns, dtype=float)

    # Initial guesses from moments
    mu_hat = np.mean(returns)
    sigma_hat = np.std(returns)
    skew_hat = float(np.mean(((returns - mu_hat) / sigma_hat) ** 3))
    kurt_hat = float(np.mean(((returns - mu_hat) / sigma_hat) ** 4))

    sigma0 = max(sigma_hat, 1e-6)
    nu0 = max((kurt_hat / 3.0 - 1.0), 0.01)
    theta0 = skew_hat * sigma_hat / max(3.0 * nu0, 0.01)

    def neg_ll(params: np.ndarray) -> float:
        s, v, th = params
        if s <= 1e-10 or v <= 1e-10:
            return 1e12
        try:
            pdf_vals = variance_gamma_pdf(returns, s, v, th)
            pdf_vals = np.maximum(pdf_vals, 1e-300)
            ll = np.sum(np.log(pdf_vals))
            if not np.isfinite(ll):
                return 1e12
            return -ll
        except Exception:
            return 1e12

    result = minimize(
        neg_ll,
        x0=np.array([sigma0, nu0, theta0]),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    sigma_fit, nu_fit, theta_fit = result.x
    return {
        "sigma": float(abs(sigma_fit)),
        "nu": float(abs(nu_fit)),
        "theta": float(theta_fit),
        "log_likelihood": float(-result.fun),
    }


def fit_nig(
    returns: ArrayLike,
) -> dict[str, float]:
    """Fit a Normal Inverse Gaussian distribution to return data via MLE.

    Parameters
    ----------
    returns : array_like
        Observed returns.

    Returns
    -------
    dict
        ``alpha`` – fitted tail parameter.
        ``beta``  – fitted skewness parameter.
        ``mu``    – fitted location.
        ``delta`` – fitted scale.
        ``log_likelihood`` – maximised log-likelihood.
    """
    returns = np.asarray(returns, dtype=float)

    mu_hat = np.mean(returns)
    sigma_hat = np.std(returns)

    # Reasonable initial guesses
    alpha0 = 1.0 / max(sigma_hat, 1e-6)
    beta0 = 0.0
    mu0 = mu_hat
    delta0 = max(sigma_hat, 1e-6)

    def neg_ll(params: np.ndarray) -> float:
        a, b, m, d = params
        if a <= abs(b) + 1e-6 or d <= 1e-10 or a <= 0:
            return 1e12
        try:
            pdf_vals = nig_pdf(returns, a, b, m, d)
            pdf_vals = np.maximum(pdf_vals, 1e-300)
            ll = np.sum(np.log(pdf_vals))
            if not np.isfinite(ll):
                return 1e12
            return -ll
        except Exception:
            return 1e12

    result = minimize(
        neg_ll,
        x0=np.array([alpha0, beta0, mu0, delta0]),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    a_fit, b_fit, m_fit, d_fit = result.x
    return {
        "alpha": float(abs(a_fit)),
        "beta": float(b_fit),
        "mu": float(m_fit),
        "delta": float(abs(d_fit)),
        "log_likelihood": float(-result.fun),
    }
