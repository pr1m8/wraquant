"""Option pricing under Lévy process models.

Implements FFT-based (Carr-Madan) and COS-method pricing for European
options under Variance Gamma, NIG, and arbitrary characteristic function
models.  All implementations are pure numpy/scipy.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "vg_european_fft",
    "nig_european_fft",
    "fft_option_price",
    "cos_method",
]


# ---------------------------------------------------------------------------
# Characteristic functions (risk-neutral)
# ---------------------------------------------------------------------------


def _char_fn_vg(
    u: np.ndarray,
    sigma: float,
    nu: float,
    theta: float,
    rf_rate: float,
    T: float,
) -> np.ndarray:
    r"""Risk-neutral VG characteristic function.

    The log characteristic function of the VG process at time *T* under
    the risk-neutral measure is:

    .. math::

        \psi(u) = i\,u\,(r + \omega)\,T
        - \frac{T}{\nu}\,
        \ln\!\left(1 - i\,\theta\,\nu\,u
        + \tfrac{1}{2}\sigma^2\nu\,u^2\right)

    where  :math:`\omega = \frac{1}{\nu}\ln(1 - \theta\nu - \sigma^2\nu/2)`
    is the convexity correction.
    """
    omega = (1.0 / nu) * np.log(1.0 - theta * nu - 0.5 * sigma**2 * nu)
    base = 1.0 - 1j * theta * nu * u + 0.5 * sigma**2 * nu * u**2
    return np.exp(1j * u * (rf_rate + omega) * T - (T / nu) * np.log(base))


def _char_fn_nig(
    u: np.ndarray,
    alpha: float,
    beta: float,
    mu: float,
    delta: float,
    rf_rate: float,
    T: float,
) -> np.ndarray:
    r"""Risk-neutral NIG characteristic function.

    .. math::

        \psi(u) = i\,u\,(r + \omega)\,T
        + \delta\,T\left(\sqrt{\alpha^2 - \beta^2}
        - \sqrt{\alpha^2 - (\beta + i\,u)^2}\right)

    where :math:`\omega = \delta(\sqrt{\alpha^2 - (\beta+1)^2}
    - \sqrt{\alpha^2 - \beta^2})`.
    """
    gamma0 = np.sqrt(alpha**2 - beta**2)
    gamma1 = np.sqrt(alpha**2 - (beta + 1.0) ** 2)
    omega = delta * (gamma1 - gamma0) - mu
    gamma_u = np.sqrt(alpha**2 - (beta + 1j * u) ** 2)
    return np.exp(1j * u * (rf_rate + omega + mu) * T + delta * T * (gamma0 - gamma_u))


# ---------------------------------------------------------------------------
# Generic FFT pricing (Carr-Madan)
# ---------------------------------------------------------------------------


def fft_option_price(
    char_fn: Callable[[np.ndarray], np.ndarray],
    spot: float,
    strike: float | np.ndarray,
    rf_rate: float,
    T: float,
    n_fft: int = 4096,
    alpha_damp: float = 1.5,
) -> float | np.ndarray:
    r"""Price European call option(s) via the Carr-Madan FFT method.

    The Carr-Madan (1999) approach uses the Fast Fourier Transform to
    efficiently evaluate option prices across a grid of log-strikes in
    a single FFT pass.  The method works with **any model** for which
    the characteristic function of :math:`\ln(S_T)` is known, making
    it the universal workhorse for Fourier-based option pricing.

    The core idea: multiply the modified characteristic function by
    Simpson's rule weights, apply the FFT, and interpolate to the
    desired strike(s).

    Parameters:
        char_fn (callable): Characteristic function of
            :math:`\ln(S_T)` under the risk-neutral measure.
            Signature: ``char_fn(u) -> complex ndarray``.
        spot (float): Current spot price.
        strike (float | ndarray): Strike price(s).  Accepts a scalar
            or an array for vectorised pricing.
        rf_rate (float): Risk-free rate (annualised, continuously
            compounded).
        T (float): Time to maturity (years).
        n_fft (int): Number of FFT points (default 4096).  Must be a
            power of 2 for optimal FFT performance.
        alpha_damp (float): Carr-Madan dampening parameter (default
            1.5).  Controls the integrand's decay; values in [1, 2]
            work well for most models.

    Returns:
        float | ndarray: European call price(s).  Returns a float when
            *strike* is scalar, otherwise an ndarray.

    Example:
        >>> import numpy as np
        >>> sigma, r, T = 0.2, 0.05, 1.0
        >>> log_S = np.log(100.0)
        >>> drift = (r - 0.5 * sigma**2) * T
        >>> char_fn = lambda u: np.exp(1j*u*(log_S + drift) - 0.5*sigma**2*T*u**2)
        >>> price = fft_option_price(char_fn, 100.0, 100.0, r, T)
        >>> 9.0 < price < 12.0
        True

    See Also:
        cos_method: Alternative Fourier method (faster for single strikes).
        characteristic_function_price: Unified interface to FFT and COS.

    References:
        Carr, P. & Madan, D.B. (1999). *Option Valuation Using the Fast
        Fourier Transform.* Journal of Computational Finance 2(4).
    """
    scalar_strike = np.ndim(strike) == 0
    K = np.atleast_1d(np.asarray(strike, dtype=float))

    # FFT grid parameters
    eta = 0.25  # spacing in u-space
    lam = 2.0 * np.pi / (n_fft * eta)  # spacing in log-strike space
    b = n_fft * lam / 2.0  # upper boundary of log-strike

    # Integration grid
    j = np.arange(n_fft)
    u = eta * j

    # Modified characteristic function (Carr-Madan integrand)
    cf_vals = char_fn(u - (alpha_damp + 1.0) * 1j)
    denom = alpha_damp**2 + alpha_damp - u**2 + 1j * u * (2.0 * alpha_damp + 1.0)

    # Avoid division by zero
    denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)

    psi = np.exp(-rf_rate * T) * cf_vals / denom

    # Simpson's rule weights
    simpson = 3.0 + (-1.0) ** (j + 1)
    simpson[0] = 1.0
    simpson = simpson / 3.0

    x = np.exp(1j * b * u) * psi * eta * simpson

    # FFT
    fft_result = np.fft.fft(x)

    # Log-strike grid
    k_grid = -b + lam * np.arange(n_fft)

    # Call prices on the grid
    call_prices = np.exp(-alpha_damp * k_grid) / np.pi * np.real(fft_result)

    # Interpolate to desired strikes
    log_K = np.log(K)
    prices = np.interp(log_K, k_grid, call_prices)
    prices = np.maximum(prices, 0.0)

    if scalar_strike:
        return float(prices[0])
    return prices


# ---------------------------------------------------------------------------
# VG European FFT
# ---------------------------------------------------------------------------


def vg_european_fft(
    spot: float,
    strike: float,
    rf_rate: float,
    T: float,
    sigma: float,
    nu: float,
    theta: float,
    n_fft: int = 4096,
) -> float:
    """Price a European call under the Variance Gamma model via FFT.

    Combines the VG characteristic function with the Carr-Madan FFT
    method to price a European call option.  The VG model captures
    skewness and excess kurtosis in the return distribution through
    three parameters.

    Use this when the Black-Scholes assumption of Gaussian returns is
    too restrictive and you observe heavier tails or asymmetry in the
    market-implied distribution.

    Parameters:
        spot (float): Current spot price.
        strike (float): Strike price.
        rf_rate (float): Risk-free rate (annualised).
        T (float): Time to maturity in years.
        sigma (float): VG volatility parameter (controls overall
            dispersion).
        nu (float): Variance rate of the Gamma subordinator (controls
            kurtosis; as nu -> 0 the model converges to Black-Scholes).
        theta (float): Drift of the VG process (controls skewness;
            negative = left-skewed returns).
        n_fft (int): FFT grid size (default 4096).

    Returns:
        float: European call price.

    Example:
        >>> price = vg_european_fft(100, 100, 0.05, 1.0, 0.2, 0.5, -0.1)
        >>> price > 0
        True

    See Also:
        nig_european_fft: NIG model pricing via FFT.
        fft_option_price: Generic FFT pricer for any characteristic fn.
        vg_characteristic: Reusable VG characteristic function.
    """
    log_spot = np.log(spot)

    def char_fn(u: np.ndarray) -> np.ndarray:
        return np.exp(1j * u * log_spot) * _char_fn_vg(u, sigma, nu, theta, rf_rate, T)

    return fft_option_price(char_fn, spot, strike, rf_rate, T, n_fft)


# ---------------------------------------------------------------------------
# NIG European FFT
# ---------------------------------------------------------------------------


def nig_european_fft(
    spot: float,
    strike: float,
    rf_rate: float,
    T: float,
    alpha: float,
    beta: float,
    mu: float,
    delta: float,
    n_fft: int = 4096,
) -> float:
    """Price a European call under the Normal Inverse Gaussian model via FFT.

    Combines the NIG characteristic function with the Carr-Madan FFT
    method.  NIG features semi-heavy tails (heavier than Gaussian,
    lighter than Cauchy) and is widely used in credit, commodity, and
    FX markets.

    Parameters:
        spot (float): Current spot price.
        strike (float): Strike price.
        rf_rate (float): Risk-free rate (annualised).
        T (float): Time to maturity in years.
        alpha (float): Tail heaviness parameter (alpha > 0).  Larger
            values produce lighter tails.
        beta (float): Skewness parameter (-alpha < beta < alpha).
            Negative beta produces left-skewed returns.
        mu (float): Location parameter.
        delta (float): Scale parameter (delta > 0).
        n_fft (int): FFT grid size (default 4096).

    Returns:
        float: European call price.

    Example:
        >>> price = nig_european_fft(100, 100, 0.05, 1.0, 15.0, -3.0, 0.0, 0.5)
        >>> price > 0
        True

    See Also:
        vg_european_fft: Variance Gamma model pricing via FFT.
        fft_option_price: Generic FFT pricer for any characteristic fn.
        nig_characteristic: Reusable NIG characteristic function.
    """
    log_spot = np.log(spot)

    def char_fn(u: np.ndarray) -> np.ndarray:
        return np.exp(1j * u * log_spot) * _char_fn_nig(
            u, alpha, beta, mu, delta, rf_rate, T
        )

    return fft_option_price(char_fn, spot, strike, rf_rate, T, n_fft)


# ---------------------------------------------------------------------------
# COS method
# ---------------------------------------------------------------------------


def cos_method(
    char_fn: Callable[[np.ndarray], np.ndarray],
    spot: float,
    strike: float,
    rf_rate: float,
    T: float,
    n_terms: int = 256,
    L: float = 10.0,
) -> float:
    r"""Price a European call option using the COS (Fourier-cosine) method.

    The COS method of Fang & Oosterlee (2008) expands the risk-neutral
    density in a cosine series and integrates analytically against the
    payoff coefficients.  It is faster than the FFT method for pricing
    a single option because it avoids the full FFT pass and works
    directly with the characteristic function at specific frequencies.

    The price is computed as:

    .. math::

        C = e^{-rT} \sum_{k=0}^{N-1} ' \text{Re}\bigl[\phi(u_k)\,
        e^{-iu_k a}\bigr]\,V_k

    where :math:`u_k = k\pi/(b-a)` and :math:`V_k` are the cosine
    coefficients of the call payoff.

    Parameters:
        char_fn (callable): Characteristic function of
            :math:`\ln(S_T)`.  Signature: ``char_fn(u) -> complex
            ndarray``.
        spot (float): Current spot price.
        strike (float): Strike price.
        rf_rate (float): Risk-free rate (annualised).
        T (float): Time to maturity in years.
        n_terms (int): Number of cosine expansion terms (default 256).
            64--256 terms suffice for most models.
        L (float): Truncation range controlling the integration domain
            ``[log(spot) - L, log(spot) + L]`` (default 10).

    Returns:
        float: European call price.

    See Also:
        fft_option_price: FFT-based pricing (better for strike grids).
        characteristic_function_price: Unified interface.

    References:
        Fang, F. & Oosterlee, C.W. (2008). *A Novel Pricing Method for
        European Options Based on Fourier-Cosine Series Expansions.*
        SIAM Journal on Scientific Computing 31(2).
    """
    # The char_fn is assumed to be the characteristic function of log(S_T).
    # We work with x = log(S_T), and the call payoff is max(e^x - K, 0).
    # The integration domain [a, b] is centered on log(spot) + drift.
    log_spot = np.log(spot)

    # Integration domain [a, b] — symmetric around log(spot)
    a = log_spot - L
    b = log_spot + L

    log_K = np.log(strike)

    k = np.arange(n_terms)
    u_k = k * np.pi / (b - a)

    # Characteristic function values
    cf_vals = char_fn(u_k)

    # Cosine coefficients for call payoff: max(e^x - K, 0) on [a, b]
    # Payoff is nonzero for x >= log(K)
    chi_k = _chi_cos(k, a, b, log_K, b)  # integral of e^x cos(...)
    psi_k = _psi_cos(k, a, b, log_K, b)  # integral of cos(...)

    V_k = 2.0 / (b - a) * (chi_k - strike * psi_k)

    # COS formula: sum Re[phi(u_k) * exp(-i*u_k*a)] * V_k
    cf_shifted = cf_vals * np.exp(-1j * u_k * a)
    summand = np.real(cf_shifted) * V_k
    summand[0] *= 0.5  # first term gets factor 1/2

    price = np.exp(-rf_rate * T) * np.sum(summand)
    return float(max(price, 0.0))


def _chi_cos(
    k: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
) -> np.ndarray:
    r"""Compute :math:`\chi_k = \int_c^d e^x \cos(k\pi(x-a)/(b-a))\,dx`."""
    bma = b - a
    k_pi = k * np.pi / bma

    # For k=0 special case
    result = np.empty_like(k, dtype=float)

    # General formula
    denom = 1.0 + k_pi**2
    cos_d = np.cos(k_pi * (d - a))
    cos_c = np.cos(k_pi * (c - a))
    sin_d = np.sin(k_pi * (d - a))
    sin_c = np.sin(k_pi * (c - a))

    result = (
        1.0
        / denom
        * (np.exp(d) * (cos_d + k_pi * sin_d) - np.exp(c) * (cos_c + k_pi * sin_c))
    )
    return result


def _psi_cos(
    k: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
) -> np.ndarray:
    r"""Compute :math:`\psi_k = \int_c^d \cos(k\pi(x-a)/(b-a))\,dx`."""
    bma = b - a
    result = np.empty_like(k, dtype=float)

    # k = 0
    result[0] = d - c

    # k > 0
    k_nonzero = k[1:]
    k_pi = k_nonzero * np.pi / bma
    result[1:] = (np.sin(k_pi * (d - a)) - np.sin(k_pi * (c - a))) / k_pi

    return result
