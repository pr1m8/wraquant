"""Option Greeks via analytical Black-Scholes formulas.

Provides closed-form solutions for delta, gamma, theta, vega, and rho
for European options under the Black-Scholes model.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from wraquant.core.types import OptionType

__all__ = [
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "all_greeks",
]


def _parse_option_type(option_type: str | OptionType) -> OptionType:
    """Normalize option_type to OptionType enum."""
    if isinstance(option_type, OptionType):
        return option_type
    return OptionType(option_type.lower())


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> np.float64:
    """Compute d1 in the Black-Scholes formula."""
    return np.float64((np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> np.float64:
    """Compute d2 in the Black-Scholes formula."""
    return np.float64(_d1(S, K, T, r, sigma) - sigma * np.sqrt(T))


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str | OptionType = "call",
) -> np.float64:
    """Compute the Black-Scholes delta (sensitivity to underlying price).

    Parameters:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying (annualized).
        option_type: 'call' or 'put'.

    Returns:
        Delta value. Call delta in [0, 1], put delta in [-1, 0].

    Example:
        >>> delta(100, 100, 1.0, 0.05, 0.2, 'call')
        0.636...
    """
    otype = _parse_option_type(option_type)
    d1_val = _d1(S, K, T, r, sigma)

    if otype == OptionType.CALL:
        return np.float64(norm.cdf(d1_val))
    return np.float64(norm.cdf(d1_val) - 1.0)


def gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> np.float64:
    """Compute the Black-Scholes gamma (rate of change of delta).

    Gamma is the same for both calls and puts.

    Parameters:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying (annualized).

    Returns:
        Gamma value (always non-negative).

    Example:
        >>> gamma(100, 100, 1.0, 0.05, 0.2)
        0.018...
    """
    d1_val = _d1(S, K, T, r, sigma)
    return np.float64(norm.pdf(d1_val) / (S * sigma * np.sqrt(T)))


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str | OptionType = "call",
) -> np.float64:
    """Compute the Black-Scholes theta (time decay per year).

    Parameters:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying (annualized).
        option_type: 'call' or 'put'.

    Returns:
        Theta value (typically negative, representing value decay per year).

    Example:
        >>> theta(100, 100, 1.0, 0.05, 0.2, 'call')
        -6.41...
    """
    otype = _parse_option_type(option_type)
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = d1_val - sigma * np.sqrt(T)

    term1 = -(S * norm.pdf(d1_val) * sigma) / (2.0 * np.sqrt(T))

    if otype == OptionType.CALL:
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2_val)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)

    return np.float64(term1 + term2)


def vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> np.float64:
    """Compute the Black-Scholes vega (sensitivity to volatility).

    Vega is the same for both calls and puts. Returned per unit change
    in sigma (i.e., multiply by 0.01 for a 1% vol move).

    Parameters:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying (annualized).

    Returns:
        Vega value (always non-negative).

    Example:
        >>> vega(100, 100, 1.0, 0.05, 0.2)
        37.52...
    """
    d1_val = _d1(S, K, T, r, sigma)
    return np.float64(S * norm.pdf(d1_val) * np.sqrt(T))


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str | OptionType = "call",
) -> np.float64:
    """Compute the Black-Scholes rho (sensitivity to interest rate).

    Parameters:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying (annualized).
        option_type: 'call' or 'put'.

    Returns:
        Rho value per unit change in r.

    Example:
        >>> rho(100, 100, 1.0, 0.05, 0.2, 'call')
        53.23...
    """
    otype = _parse_option_type(option_type)
    d2_val = _d2(S, K, T, r, sigma)

    if otype == OptionType.CALL:
        return np.float64(K * T * np.exp(-r * T) * norm.cdf(d2_val))
    return np.float64(-K * T * np.exp(-r * T) * norm.cdf(-d2_val))


def all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str | OptionType = "call",
) -> dict[str, np.float64]:
    """Compute all Black-Scholes Greeks at once.

    Parameters:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying (annualized).
        option_type: 'call' or 'put'.

    Returns:
        Dictionary with keys 'delta', 'gamma', 'theta', 'vega', 'rho'.

    Example:
        >>> greeks = all_greeks(100, 100, 1.0, 0.05, 0.2, 'call')
        >>> sorted(greeks.keys())
        ['delta', 'gamma', 'rho', 'theta', 'vega']
    """
    otype = _parse_option_type(option_type)

    # Compute d1 and d2 once
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = d1_val - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1_val)
    sqrt_T = np.sqrt(T)
    exp_neg_rT = np.exp(-r * T)

    # Delta
    if otype == OptionType.CALL:
        delta_val = norm.cdf(d1_val)
    else:
        delta_val = norm.cdf(d1_val) - 1.0

    # Gamma (same for calls and puts)
    gamma_val = pdf_d1 / (S * sigma * sqrt_T)

    # Theta
    term1 = -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
    if otype == OptionType.CALL:
        theta_val = term1 - r * K * exp_neg_rT * norm.cdf(d2_val)
    else:
        theta_val = term1 + r * K * exp_neg_rT * norm.cdf(-d2_val)

    # Vega (same for calls and puts)
    vega_val = S * pdf_d1 * sqrt_T

    # Rho
    if otype == OptionType.CALL:
        rho_val = K * T * exp_neg_rT * norm.cdf(d2_val)
    else:
        rho_val = -K * T * exp_neg_rT * norm.cdf(-d2_val)

    return {
        "delta": np.float64(delta_val),
        "gamma": np.float64(gamma_val),
        "theta": np.float64(theta_val),
        "vega": np.float64(vega_val),
        "rho": np.float64(rho_val),
    }
