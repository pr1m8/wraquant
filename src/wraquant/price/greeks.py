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
    r"""Compute the Black-Scholes delta (sensitivity to underlying price).

    Delta measures how much the option price changes for a one-unit
    change in the underlying price.  It is the first derivative of
    the option price with respect to S and also represents the
    hedge ratio (number of shares to hold per option to be
    delta-neutral).

    .. math::

        \Delta_{\text{call}} = \Phi(d_1), \quad
        \Delta_{\text{put}} = \Phi(d_1) - 1

    Parameters:
        S (float): Current underlying price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying (annualized).
        option_type (str | OptionType): ``'call'`` or ``'put'``.

    Returns:
        np.float64: Delta value.  Call delta is in [0, 1]; put delta
            is in [-1, 0].  At-the-money forward options have delta
            near 0.5 (call) or -0.5 (put).

    Example:
        >>> delta(100, 100, 1.0, 0.05, 0.2, 'call')
        0.636...
        >>> delta(100, 100, 1.0, 0.05, 0.2, 'put')
        -0.363...

    See Also:
        gamma: Rate of change of delta.
        all_greeks: Compute all Greeks in a single call.
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
    r"""Compute the Black-Scholes gamma (rate of change of delta).

    Gamma is the second derivative of option price with respect to S.
    It measures the convexity of the option position and is identical
    for calls and puts.  High gamma means the hedge ratio changes
    rapidly, requiring frequent rebalancing.

    .. math::

        \Gamma = \frac{\phi(d_1)}{S\,\sigma\,\sqrt{T}}

    Gamma is highest for at-the-money options near expiry.

    Parameters:
        S (float): Current underlying price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying (annualized).

    Returns:
        np.float64: Gamma value (always non-negative).  Expressed in
            price units per unit squared move in the underlying.

    Example:
        >>> gamma(100, 100, 1.0, 0.05, 0.2)
        0.018...

    See Also:
        delta: First-order sensitivity to the underlying.
        all_greeks: Compute all Greeks in a single call.
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
    r"""Compute the Black-Scholes theta (time decay per year).

    Theta measures the rate at which the option loses value as time
    passes, holding all else constant.  It is typically negative for
    long option positions (time works against the holder).  Divide by
    252 to get the approximate daily time decay.

    Parameters:
        S (float): Current underlying price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying (annualized).
        option_type (str | OptionType): ``'call'`` or ``'put'``.

    Returns:
        np.float64: Theta value per year.  Typically negative for long
            options.  For daily theta, divide by 252 (or 365 for
            calendar days).

    Example:
        >>> theta(100, 100, 1.0, 0.05, 0.2, 'call')
        -6.41...

    See Also:
        vega: Sensitivity to volatility.
        all_greeks: Compute all Greeks in a single call.
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
    r"""Compute the Black-Scholes vega (sensitivity to volatility).

    Vega measures how much the option price changes for a one-unit
    (100 percentage point) change in implied volatility.  It is
    identical for calls and puts and is always non-negative.  To get
    the price change for a 1 percentage point vol move, multiply the
    result by 0.01.

    .. math::

        \mathcal{V} = S\,\phi(d_1)\,\sqrt{T}

    Parameters:
        S (float): Current underlying price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying (annualized).

    Returns:
        np.float64: Vega per unit change in sigma (always non-negative).
            Multiply by 0.01 for the price change per 1% vol move.

    Example:
        >>> vega(100, 100, 1.0, 0.05, 0.2)
        37.52...

    See Also:
        implied_volatility: Invert the BS formula using vega as the
            Newton-Raphson derivative.
        all_greeks: Compute all Greeks in a single call.
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
    r"""Compute the Black-Scholes rho (sensitivity to interest rate).

    Rho measures how much the option price changes for a one-unit
    (100 percentage point) change in the risk-free rate.  Multiply
    by 0.01 for the price change per 1% rate move.  Rho is typically
    the smallest of the Greeks for short-dated equity options but
    becomes significant for long-dated or fixed-income options.

    Parameters:
        S (float): Current underlying price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying (annualized).
        option_type (str | OptionType): ``'call'`` or ``'put'``.

    Returns:
        np.float64: Rho per unit change in r.  Positive for calls,
            negative for puts.

    Example:
        >>> rho(100, 100, 1.0, 0.05, 0.2, 'call')
        53.23...

    See Also:
        all_greeks: Compute all Greeks in a single call.
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
    """Compute all Black-Scholes Greeks in a single efficient call.

    Computes d1 and d2 once and derives all five Greeks, avoiding the
    redundant computation that would occur when calling each Greek
    function individually.  Use this when you need a full risk profile
    for a position.

    Parameters:
        S (float): Current underlying price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying (annualized).
        option_type (str | OptionType): ``'call'`` or ``'put'``.

    Returns:
        dict[str, np.float64]: Dictionary with keys ``'delta'``,
            ``'gamma'``, ``'theta'``, ``'vega'``, ``'rho'``.  See
            the individual Greek functions for interpretation of each
            value.

    Example:
        >>> greeks = all_greeks(100, 100, 1.0, 0.05, 0.2, 'call')
        >>> sorted(greeks.keys())
        ['delta', 'gamma', 'rho', 'theta', 'vega']

    See Also:
        delta, gamma, theta, vega, rho: Individual Greek computations.
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
