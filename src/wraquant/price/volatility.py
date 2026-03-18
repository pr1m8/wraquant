"""Implied volatility and volatility surface construction.

Provides Newton-Raphson implied volatility solver and tools for
building volatility smiles and surfaces from market data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from wraquant.core.exceptions import PricingError
from wraquant.core.types import OptionType
from wraquant.price.greeks import vega as bs_vega
from wraquant.price.options import black_scholes

__all__ = [
    "implied_volatility",
    "vol_smile",
    "vol_surface",
]


def _parse_option_type(option_type: str | OptionType) -> OptionType:
    """Normalize option_type to OptionType enum."""
    if isinstance(option_type, OptionType):
        return option_type
    return OptionType(option_type.lower())


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str | OptionType = "call",
    tol: float = 1e-8,
    max_iter: int = 200,
    initial_guess: float = 0.3,
) -> np.float64:
    """Compute implied volatility using Newton-Raphson iteration.

    Solves for sigma such that BS(S, K, T, r, sigma) = market_price.

    Parameters:
        market_price: Observed market price of the option.
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        option_type: 'call' or 'put'.
        tol: Convergence tolerance.
        max_iter: Maximum number of iterations.
        initial_guess: Starting volatility estimate.

    Returns:
        Implied volatility (annualized).

    Raises:
        PricingError: If the solver does not converge.

    Example:
        >>> price = black_scholes(100, 100, 1.0, 0.05, 0.2)
        >>> implied_volatility(price, 100, 100, 1.0, 0.05)
        0.2000...
    """
    otype = _parse_option_type(option_type)
    sigma = initial_guess

    for _ in range(max_iter):
        bs_price = black_scholes(S, K, T, r, sigma, otype)
        v = bs_vega(S, K, T, r, sigma)

        if v < 1e-12:
            # Vega too small; switch to bisection fallback
            return _implied_vol_bisection(market_price, S, K, T, r, otype, tol)

        diff = bs_price - market_price
        if abs(diff) < tol:
            return np.float64(sigma)

        sigma = sigma - diff / v

        # Keep sigma in reasonable bounds
        if sigma <= 0.0:
            sigma = 1e-4
        elif sigma > 10.0:
            sigma = 10.0

    raise PricingError(
        f"Implied volatility solver did not converge after {max_iter} iterations. "
        f"Last sigma={sigma:.6f}, price diff={diff:.2e}"
    )


def _implied_vol_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    tol: float,
    lo: float = 1e-6,
    hi: float = 10.0,
    max_iter: int = 200,
) -> np.float64:
    """Bisection fallback for implied volatility when Newton fails."""
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        price = black_scholes(S, K, T, r, mid, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return np.float64(mid)
        if diff > 0:
            hi = mid
        else:
            lo = mid

    raise PricingError(
        f"Implied volatility bisection did not converge after {max_iter} iterations."
    )


def vol_smile(
    strikes: Sequence[float] | npt.NDArray[np.floating],
    market_prices: Sequence[float] | npt.NDArray[np.floating],
    S: float,
    T: float,
    r: float,
    option_type: str | OptionType = "call",
) -> dict[str, npt.NDArray[np.float64]]:
    """Compute a volatility smile from market prices at multiple strikes.

    Parameters:
        strikes: Array of strike prices.
        market_prices: Corresponding market option prices.
        S: Current underlying price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        option_type: 'call' or 'put'.

    Returns:
        Dictionary with keys 'strikes' and 'implied_vols', each an ndarray.

    Example:
        >>> strikes = [90, 95, 100, 105, 110]
        >>> prices = [12.5, 9.0, 6.0, 3.8, 2.1]
        >>> result = vol_smile(strikes, prices, 100, 0.5, 0.05)
        >>> len(result['implied_vols'])
        5
    """
    otype = _parse_option_type(option_type)
    strikes_arr = np.asarray(strikes, dtype=np.float64)
    prices_arr = np.asarray(market_prices, dtype=np.float64)

    if len(strikes_arr) != len(prices_arr):
        raise ValueError("strikes and market_prices must have the same length.")

    ivols = np.empty(len(strikes_arr), dtype=np.float64)
    for i, (k, p) in enumerate(zip(strikes_arr, prices_arr, strict=False)):
        ivols[i] = implied_volatility(float(p), S, float(k), T, r, otype)

    return {
        "strikes": strikes_arr,
        "implied_vols": ivols,
    }


def vol_surface(
    strikes: Sequence[float] | npt.NDArray[np.floating],
    expiries: Sequence[float] | npt.NDArray[np.floating],
    market_prices: Sequence[Sequence[float]] | npt.NDArray[np.floating],
    S: float,
    r: float,
    option_type: str | OptionType = "call",
) -> dict[str, npt.NDArray[np.float64]]:
    """Construct a volatility surface from market prices across strikes and expiries.

    Parameters:
        strikes: Array of strike prices (columns).
        expiries: Array of times to expiry in years (rows).
        market_prices: 2D array of option prices, shape (len(expiries), len(strikes)).
        S: Current underlying price.
        r: Risk-free interest rate (annualized).
        option_type: 'call' or 'put'.

    Returns:
        Dictionary with keys:
        - 'strikes': 1D array of strikes
        - 'expiries': 1D array of expiries
        - 'implied_vols': 2D array of implied vols, shape (len(expiries), len(strikes))

    Example:
        >>> strikes = [95, 100, 105]
        >>> expiries = [0.25, 0.5]
        >>> prices = [[6.0, 3.5, 1.5], [8.0, 5.5, 3.5]]
        >>> surface = vol_surface(strikes, expiries, prices, 100, 0.05)
        >>> surface['implied_vols'].shape
        (2, 3)
    """
    otype = _parse_option_type(option_type)
    strikes_arr = np.asarray(strikes, dtype=np.float64)
    expiries_arr = np.asarray(expiries, dtype=np.float64)
    prices_arr = np.asarray(market_prices, dtype=np.float64)

    if prices_arr.shape != (len(expiries_arr), len(strikes_arr)):
        raise ValueError(
            f"market_prices shape {prices_arr.shape} does not match "
            f"(len(expiries), len(strikes)) = ({len(expiries_arr)}, {len(strikes_arr)})"
        )

    ivols = np.empty_like(prices_arr, dtype=np.float64)
    for i, T in enumerate(expiries_arr):
        for j, k in enumerate(strikes_arr):
            ivols[i, j] = implied_volatility(
                float(prices_arr[i, j]), S, float(k), float(T), r, otype
            )

    return {
        "strikes": strikes_arr,
        "expiries": expiries_arr,
        "implied_vols": ivols,
    }
