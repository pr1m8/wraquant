"""Ergodicity economics (Ole Peters framework).

Tools for distinguishing ensemble averages from time averages,
computing Kelly-optimal fractions, and measuring ergodicity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from wraquant.core._coerce import coerce_array

__all__ = [
    "ensemble_average",
    "time_average",
    "ergodicity_gap",
    "kelly_fraction",
    "growth_optimal_leverage",
    "ergodicity_ratio",
]


def ensemble_average(returns: ArrayLike) -> float:
    """Arithmetic mean of *returns* (the ensemble average).

    Parameters
    ----------
    returns : array_like
        Simple (arithmetic) returns, e.g. ``[0.05, -0.02, 0.03]``.

    Returns
    -------
    float
        Arithmetic mean of the returns.
    """
    returns = coerce_array(returns, name="returns")
    return float(np.mean(returns))


def time_average(returns: ArrayLike) -> float:
    r"""Time-average growth rate (geometric mean return).

    Computes the annualised-per-period geometric growth rate:

    .. math::

        g = \\left(\\prod_{i=1}^{N}(1 + r_i)\\right)^{1/N} - 1

    Parameters
    ----------
    returns : array_like
        Simple (arithmetic) returns.

    Returns
    -------
    float
        Geometric mean return per period.
    """
    returns = coerce_array(returns, name="returns")
    # Use log-sum for numerical stability
    log_growth = np.mean(np.log1p(returns))
    return float(np.expm1(log_growth))


def ergodicity_gap(returns: ArrayLike) -> float:
    """Difference between the ensemble average and the time average.

    A positive gap means the ensemble (arithmetic) average overstates
    the realised long-run growth.

    Parameters
    ----------
    returns : array_like
        Simple returns.

    Returns
    -------
    float
        ``ensemble_average(returns) - time_average(returns)``.
    """
    return ensemble_average(returns) - time_average(returns)


def kelly_fraction(returns: ArrayLike) -> float:
    """Optimal Kelly criterion fraction for a simple binary-style bet.

    For a series of returns this computes the leverage that maximises
    the expected log-growth rate via a simple numerical optimisation
    over a grid.

    Parameters
    ----------
    returns : array_like
        Simple returns for each period.

    Returns
    -------
    float
        Optimal Kelly fraction (leverage).  A value of 1.0 means
        full investment; values > 1.0 indicate levered positions.
    """
    returns = coerce_array(returns, name="returns")

    # Grid search for the leverage that maximises E[log(1 + f*r)]
    fractions = np.linspace(0.0, 5.0, 5001)
    best_f = 0.0
    best_g = -np.inf

    for f in fractions:
        leveraged = 1.0 + f * returns
        if np.any(leveraged <= 0):
            continue
        g = np.mean(np.log(leveraged))
        if g > best_g:
            best_g = g
            best_f = f

    return float(best_f)


def growth_optimal_leverage(
    returns: ArrayLike,
    risk_free: float = 0.0,
) -> float:
    """Leverage that maximises the time-average growth rate.

    Maximises ``E[log(1 + risk_free + f * (r - risk_free))]`` over *f*.

    Parameters
    ----------
    returns : array_like
        Simple returns per period.
    risk_free : float, optional
        Risk-free rate per period (default 0.0).

    Returns
    -------
    float
        Growth-optimal leverage.
    """
    returns = coerce_array(returns, name="returns")
    excess = returns - risk_free

    fractions = np.linspace(0.0, 5.0, 5001)
    best_f = 0.0
    best_g = -np.inf

    for f in fractions:
        total = 1.0 + risk_free + f * excess
        if np.any(total <= 0):
            continue
        g = np.mean(np.log(total))
        if g > best_g:
            best_g = g
            best_f = f

    return float(best_f)


def ergodicity_ratio(returns: ArrayLike) -> float:
    """Ratio of the time-average to the ensemble-average growth rate.

    A ratio of 1.0 indicates an ergodic process; ratios below 1.0
    indicate that time averaging yields lower growth than the
    ensemble expectation.

    Parameters
    ----------
    returns : array_like
        Simple returns.

    Returns
    -------
    float
        ``time_average(returns) / ensemble_average(returns)``.
        Returns ``1.0`` if the ensemble average is zero.
    """
    ea = ensemble_average(returns)
    ta = time_average(returns)
    if ea == 0.0:
        return 1.0
    return float(ta / ea)
