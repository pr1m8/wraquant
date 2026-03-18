"""Options pricing models: Black-Scholes, binomial tree, and Monte Carlo.

Provides pure numpy/scipy implementations of standard option pricing
models for European and American options.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from wraquant.core.types import OptionStyle, OptionType

__all__ = [
    "black_scholes",
    "binomial_tree",
    "monte_carlo_option",
]


def _parse_option_type(option_type: str | OptionType) -> OptionType:
    """Normalize option_type to OptionType enum."""
    if isinstance(option_type, OptionType):
        return option_type
    return OptionType(option_type.lower())


def _parse_option_style(style: str | OptionStyle) -> OptionStyle:
    """Normalize style to OptionStyle enum."""
    if isinstance(style, OptionStyle):
        return style
    return OptionStyle(style.lower())


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str | OptionType = "call",
) -> np.float64:
    """Price a European option using the Black-Scholes formula.

    Parameters:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized, continuous compounding).
        sigma: Volatility of the underlying (annualized).
        option_type: 'call' or 'put'.

    Returns:
        The Black-Scholes option price.

    Example:
        >>> black_scholes(100, 100, 1.0, 0.05, 0.2)
        10.450...
    """
    otype = _parse_option_type(option_type)

    if T <= 0.0:
        # At expiration, return intrinsic value
        if otype == OptionType.CALL:
            return np.float64(max(S - K, 0.0))
        return np.float64(max(K - S, 0.0))

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if otype == OptionType.CALL:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return np.float64(price)


def binomial_tree(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    option_type: str | OptionType = "call",
    style: str | OptionStyle = "european",
) -> np.float64:
    """Price an option using the Cox-Ross-Rubinstein (CRR) binomial tree.

    Supports both European and American exercise styles.

    Parameters:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized, continuous compounding).
        sigma: Volatility of the underlying (annualized).
        n_steps: Number of time steps in the tree.
        option_type: 'call' or 'put'.
        style: 'european' or 'american'.

    Returns:
        The binomial tree option price.

    Example:
        >>> binomial_tree(100, 100, 1.0, 0.05, 0.2, 200)
        10.4...
    """
    otype = _parse_option_type(option_type)
    ostyle = _parse_option_style(style)

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Build the price tree at maturity
    prices = (
        S
        * u ** np.arange(n_steps, -1, -1, dtype=np.float64)
        * d ** np.arange(0, n_steps + 1, dtype=np.float64)
    )

    # Option payoff at maturity
    if otype == OptionType.CALL:
        values = np.maximum(prices - K, 0.0)
    else:
        values = np.maximum(K - prices, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])

        if ostyle == OptionStyle.AMERICAN:
            prices_i = (
                S
                * u ** np.arange(i, -1, -1, dtype=np.float64)
                * d ** np.arange(0, i + 1, dtype=np.float64)
            )
            if otype == OptionType.CALL:
                intrinsic = np.maximum(prices_i - K, 0.0)
            else:
                intrinsic = np.maximum(K - prices_i, 0.0)
            values = np.maximum(values, intrinsic)

    return np.float64(values[0])


def monte_carlo_option(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sims: int,
    n_steps: int,
    option_type: str | OptionType = "call",
    seed: int | None = None,
) -> np.float64:
    """Price a European option using Monte Carlo simulation.

    Uses geometric Brownian motion to simulate price paths and
    discounts the average payoff.

    Parameters:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized, continuous compounding).
        sigma: Volatility of the underlying (annualized).
        n_sims: Number of simulation paths.
        n_steps: Number of time steps per path.
        option_type: 'call' or 'put'.
        seed: Random seed for reproducibility.

    Returns:
        The Monte Carlo option price estimate.

    Example:
        >>> monte_carlo_option(100, 100, 1.0, 0.05, 0.2, 100000, 252, seed=42)
        10.4...
    """
    otype = _parse_option_type(option_type)
    rng = np.random.default_rng(seed)

    dt = T / n_steps
    nudt = (r - 0.5 * sigma**2) * dt
    sigdt = sigma * np.sqrt(dt)

    # Simulate log-returns and compute terminal prices
    z = rng.standard_normal((n_sims, n_steps))
    log_returns = nudt + sigdt * z
    log_S_T = np.log(S) + np.sum(log_returns, axis=1)
    S_T = np.exp(log_S_T)

    # Compute payoffs
    if otype == OptionType.CALL:
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    # Discounted expected payoff
    price = np.exp(-r * T) * np.mean(payoffs)
    return np.float64(price)
