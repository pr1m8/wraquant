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
    r"""Price a European option using the Black-Scholes closed-form formula.

    The Black-Scholes (1973) formula gives the theoretical price of a
    European option under the assumptions of log-normal asset dynamics,
    constant volatility, and continuous trading.  Use this as the
    baseline pricing model or as a quick sanity check against more
    complex models (Heston, Levy, etc.).

    For a call:

    .. math::

        C = S\,\Phi(d_1) - K\,e^{-rT}\,\Phi(d_2)

    where :math:`d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}`
    and :math:`d_2 = d_1 - \sigma\sqrt{T}`.

    Parameters:
        S (float): Current underlying price.
        K (float): Strike price.
        T (float): Time to expiration in years.  At ``T <= 0`` the
            intrinsic value is returned.
        r (float): Risk-free interest rate (annualized, continuously
            compounded).
        sigma (float): Volatility of the underlying (annualized).
        option_type (str | OptionType): ``'call'`` or ``'put'``.

    Returns:
        np.float64: The Black-Scholes option price.  Always non-negative.
            For deep out-of-the-money options the price approaches zero;
            for deep in-the-money options it approaches the intrinsic
            value discounted at the risk-free rate.

    Example:
        >>> black_scholes(100, 100, 1.0, 0.05, 0.2)
        10.450...
        >>> black_scholes(100, 100, 1.0, 0.05, 0.2, 'put')
        5.573...

    Notes:
        The model assumes constant volatility and no dividends.  For
        dividend-paying stocks, reduce ``S`` by the present value of
        expected dividends or use the Merton (1973) continuous-dividend
        extension (replace ``S`` with ``S * exp(-q * T)``).

    See Also:
        binomial_tree: Lattice method supporting American exercise.
        monte_carlo_option: Simulation-based pricing.
        implied_volatility: Invert this formula to recover vol from price.

    References:
        Black, F. & Scholes, M. (1973). *The Pricing of Options and
        Corporate Liabilities.* Journal of Political Economy 81(3).
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
    r"""Price an option using the Cox-Ross-Rubinstein (CRR) binomial tree.

    The binomial tree discretises the asset price evolution into up/down
    moves and computes the option value by backward induction.  Unlike
    Black-Scholes, the tree supports **American-style** early exercise.
    Use this when you need American option prices or when the analytical
    formula is unavailable.

    At each step the asset moves up by ``u = exp(sigma * sqrt(dt))`` or
    down by ``d = 1/u`` with risk-neutral probability
    ``p = (exp(r*dt) - d) / (u - d)``.  The tree converges to the
    Black-Scholes price as ``n_steps`` increases.

    Parameters:
        S (float): Current underlying price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized, continuously
            compounded).
        sigma (float): Volatility of the underlying (annualized).
        n_steps (int): Number of time steps in the tree.  200--500 steps
            give good accuracy for most applications.
        option_type (str | OptionType): ``'call'`` or ``'put'``.
        style (str | OptionStyle): ``'european'`` or ``'american'``.

    Returns:
        np.float64: The binomial tree option price.  For European options
            this converges to the Black-Scholes price; for American
            options the price is at least as large as the European price
            due to the early exercise premium.

    Example:
        >>> binomial_tree(100, 100, 1.0, 0.05, 0.2, 200)
        10.4...
        >>> binomial_tree(100, 100, 1.0, 0.05, 0.2, 200, 'put', 'american')
        6.08...

    Notes:
        Convergence is O(1/n_steps).  For faster convergence consider
        Richardson extrapolation (average the n and 2n step prices).

    See Also:
        black_scholes: Analytical European option pricing.
        monte_carlo_option: Simulation-based pricing.

    References:
        Cox, J.C., Ross, S.A. & Rubinstein, M. (1979). *Option Pricing:
        A Simplified Approach.* Journal of Financial Economics 7(3).
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
    r"""Price a European option using Monte Carlo simulation under GBM.

    Simulates ``n_sims`` asset price paths under risk-neutral geometric
    Brownian motion, computes the terminal payoff on each path, and
    returns the discounted average.  Use Monte Carlo when the payoff is
    path-dependent, the underlying follows a non-standard process, or
    when you need a flexible framework that can be extended to exotic
    options.

    The standard error of the estimate decreases as
    :math:`1/\sqrt{n\_sims}`, so 100 000 paths typically give 2--3
    significant digits of accuracy.

    Parameters:
        S (float): Current underlying price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized, continuously
            compounded).
        sigma (float): Volatility of the underlying (annualized).
        n_sims (int): Number of simulation paths.  More paths reduce
            variance but increase computation time.
        n_steps (int): Number of time steps per path.  For European
            vanilla options a single step suffices, but path-dependent
            payoffs require finer discretisation.
        option_type (str | OptionType): ``'call'`` or ``'put'``.
        seed (int | None): Random seed for reproducibility.

    Returns:
        np.float64: The Monte Carlo estimate of the option price.  This
            is an unbiased estimator that converges to the Black-Scholes
            price for European vanilla options under GBM.

    Example:
        >>> monte_carlo_option(100, 100, 1.0, 0.05, 0.2, 100000, 252, seed=42)
        10.4...

    Notes:
        For variance reduction consider antithetic variates or control
        variates (using the Black-Scholes price as a control).

    See Also:
        black_scholes: Closed-form European pricing (faster, exact).
        binomial_tree: Lattice method for American options.
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
