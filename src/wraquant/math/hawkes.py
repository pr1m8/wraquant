"""Hawkes process for modelling event clustering.

Useful for order-flow analysis, jump modelling, and trade-arrival processes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

__all__ = [
    "hawkes_intensity",
    "simulate_hawkes",
    "fit_hawkes",
    "hawkes_branching_ratio",
]


def hawkes_intensity(
    times: ArrayLike,
    mu: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    r"""Compute the conditional intensity of a univariate Hawkes process.

    .. math::

        \\lambda(t_i) = \\mu + \\alpha \\sum_{t_j < t_i} \\beta\\,
        e^{-\\beta (t_i - t_j)}

    The intensity is evaluated at each event time in *times*.

    Parameters
    ----------
    times : array_like
        Sorted event times.
    mu : float
        Background (base) intensity (must be > 0).
    alpha : float
        Excitation magnitude per event.
    beta : float
        Exponential decay rate of excitation (must be > 0).

    Returns
    -------
    np.ndarray
        Intensity evaluated at each event time.
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    intensity = np.empty(n, dtype=float)

    if n == 0:
        return intensity

    intensity[0] = mu
    for i in range(1, n):
        # Recursive computation: A_i = exp(-beta*dt)*(A_{i-1} + alpha*beta)
        # but we store the full kernel sum for clarity.
        dt = times[i] - times[:i]
        intensity[i] = mu + alpha * np.sum(beta * np.exp(-beta * dt))

    return intensity


def simulate_hawkes(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate event times from a univariate Hawkes process.

    Uses the Ogata thinning algorithm.

    Parameters
    ----------
    mu : float
        Background intensity.
    alpha : float
        Excitation magnitude.
    beta : float
        Decay rate.
    T : float
        Observation window ``[0, T]``.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sorted array of event times in ``[0, T]``.

    Raises
    ------
    ValueError
        If the branching ratio ``alpha / beta >= 1`` (non-stationary).
    """
    if alpha / beta >= 1.0:
        raise ValueError(
            f"Branching ratio alpha/beta = {alpha / beta:.4f} >= 1; "
            "process is non-stationary."
        )

    rng = np.random.default_rng(seed)
    events: list[float] = []
    t = 0.0

    # Upper bound on intensity (will be updated)
    lambda_bar = mu

    while t < T:
        # Propose next event time via thinning
        u = rng.uniform()
        w = -np.log(u) / lambda_bar  # inter-arrival from Poisson(lambda_bar)
        t += w
        if t >= T:
            break

        # Compute actual intensity at proposed time
        if events:
            dt = t - np.asarray(events)
            intensity_t = mu + alpha * np.sum(beta * np.exp(-beta * dt))
        else:
            intensity_t = mu

        # Accept / reject
        d = rng.uniform()
        if d <= intensity_t / lambda_bar:
            events.append(t)

        # Update upper bound
        lambda_bar = intensity_t + alpha * beta

    return np.asarray(events, dtype=float)


def fit_hawkes(
    times: ArrayLike,
    T: float | None = None,
) -> dict[str, float]:
    """Fit a univariate Hawkes process via maximum likelihood.

    Parameters
    ----------
    times : array_like
        Observed event times (sorted).
    T : float or None, optional
        End of the observation window.  Defaults to ``max(times)``.

    Returns
    -------
    dict
        ``mu``    – fitted background intensity.
        ``alpha`` – fitted excitation magnitude.
        ``beta``  – fitted decay rate.
        ``log_likelihood`` – maximised log-likelihood value.
        ``branching_ratio`` – ``alpha / beta``.
    """
    times = np.asarray(times, dtype=float)
    if T is None:
        T = float(times[-1])
    n = len(times)

    def neg_log_lik(params: np.ndarray) -> float:
        mu_, alpha_, beta_ = params
        if mu_ <= 0 or alpha_ <= 0 or beta_ <= 0:
            return 1e12
        if alpha_ / beta_ >= 1.0:
            return 1e12

        # Recursive computation of A_i = sum_{j<i} exp(-beta*(t_i - t_j))
        A = 0.0
        ll = 0.0
        for i in range(n):
            intensity_i = mu_ + alpha_ * beta_ * A
            if intensity_i <= 0:
                return 1e12
            ll += np.log(intensity_i)
            if i < n - 1:
                A = (A + 1.0) * np.exp(-beta_ * (times[i + 1] - times[i]))

        # Compensator: integral of intensity over [0, T]
        compensator = mu_ * T
        for i in range(n):
            compensator += alpha_ * (1.0 - np.exp(-beta_ * (T - times[i])))

        return -(ll - compensator)

    # Initial guesses
    mu0 = n / T * 0.5
    alpha0 = 0.3
    beta0 = 1.0
    x0 = np.array([mu0, alpha0, beta0])

    result = minimize(
        neg_log_lik,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    mu_fit, alpha_fit, beta_fit = result.x
    return {
        "mu": float(mu_fit),
        "alpha": float(alpha_fit),
        "beta": float(beta_fit),
        "log_likelihood": float(-result.fun),
        "branching_ratio": float(alpha_fit / beta_fit),
    }


def hawkes_branching_ratio(alpha: float, beta: float) -> float:
    """Compute the branching ratio of a Hawkes process.

    The branching ratio ``alpha / beta`` determines stationarity: the process
    is stationary if and only if the ratio is strictly less than 1.

    Parameters
    ----------
    alpha : float
        Excitation magnitude.
    beta : float
        Decay rate.

    Returns
    -------
    float
        Branching ratio ``alpha / beta``.
    """
    return alpha / beta
