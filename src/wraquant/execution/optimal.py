"""Optimal execution models.

Implements the Almgren-Chriss (2001) framework for optimal trade
execution, balancing expected market-impact cost against execution risk.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def almgren_chriss(
    total_qty: float,
    sigma: float,
    eta: float,
    gamma: float,
    lambda_risk: float,
    n_periods: int,
) -> NDArray[np.floating]:
    """Almgren-Chriss optimal execution trajectory.

    Minimises a mean-variance objective that trades off temporary impact
    cost, permanent impact cost, and execution-risk penalty.

    Parameters:
        total_qty: Total shares to execute.
        sigma: Price volatility (per period).
        eta: Temporary impact coefficient.
        gamma: Permanent impact coefficient.
        lambda_risk: Risk-aversion parameter.  ``lambda_risk = 0``
            yields the risk-neutral (minimum-cost) trajectory.
        n_periods: Number of execution periods.

    Returns:
        Optimal holdings trajectory of length ``n_periods + 1``,
        starting at *total_qty* and ending at 0.
    """
    if n_periods <= 0:
        raise ValueError("n_periods must be positive")

    tau = 1.0  # normalised period length

    # Almgren-Chriss kappa
    kappa_sq = lambda_risk * sigma**2 / eta if eta > 0 else 0.0
    kappa = np.sqrt(kappa_sq) if kappa_sq > 0 else 0.0

    trajectory = np.zeros(n_periods + 1, dtype=np.float64)
    trajectory[0] = total_qty

    if kappa < 1e-12:
        # Risk-neutral: linear liquidation
        for j in range(1, n_periods + 1):
            trajectory[j] = total_qty * (1.0 - j / n_periods)
    else:
        denom = np.sinh(kappa * n_periods * tau)
        if abs(denom) < 1e-15:
            # Fallback to linear
            for j in range(1, n_periods + 1):
                trajectory[j] = total_qty * (1.0 - j / n_periods)
        else:
            for j in range(1, n_periods + 1):
                trajectory[j] = total_qty * np.sinh(kappa * (n_periods - j) * tau) / denom

    # Enforce terminal condition
    trajectory[-1] = 0.0
    return trajectory


def optimal_execution_cost(
    trajectory: NDArray[np.floating],
    sigma: float,
    eta: float,
    gamma: float,
) -> dict[str, float]:
    """Compute expected cost and variance of a given execution trajectory.

    Parameters:
        trajectory: Holdings path of length ``n_periods + 1``, starting
            at the initial position and ending at 0.
        sigma: Price volatility per period.
        eta: Temporary impact coefficient.
        gamma: Permanent impact coefficient.

    Returns:
        Dictionary with ``'expected_cost'``, ``'variance'``, and
        ``'std_dev'``.
    """
    n = len(trajectory) - 1
    trades = -np.diff(trajectory)  # quantities sold each period (positive)

    # Temporary impact cost: eta * sum(n_j^2)
    temp_cost = eta * float(np.sum(trades**2))

    # Permanent impact cost: gamma * sum(n_j * x_j) where x_j is remaining
    perm_cost = gamma * float(np.sum(trades * trajectory[:-1]))

    expected_cost = temp_cost + perm_cost

    # Variance of execution: sigma^2 * sum(x_j^2)
    variance = sigma**2 * float(np.sum(trajectory[1:]**2))
    std_dev = np.sqrt(variance)

    return {
        "expected_cost": expected_cost,
        "variance": variance,
        "std_dev": std_dev,
    }


def execution_frontier(
    total_qty: float,
    sigma: float,
    eta: float,
    gamma: float,
    n_points: int = 20,
    n_periods: int = 20,
) -> dict[str, NDArray[np.floating]]:
    """Efficient frontier of (expected cost, risk) pairs.

    Sweeps over a range of risk-aversion parameters to trace out the
    trade-off between expected execution cost and execution risk.

    Parameters:
        total_qty: Total shares to execute.
        sigma: Price volatility per period.
        eta: Temporary impact coefficient.
        gamma: Permanent impact coefficient.
        n_points: Number of points on the frontier.
        n_periods: Number of execution periods for each trajectory.

    Returns:
        Dictionary with ``'lambda_values'``, ``'expected_cost'``, and
        ``'std_dev'`` arrays, each of length *n_points*.
    """
    # Sweep lambda from near-zero (risk-neutral) to aggressive risk aversion
    lambda_values = np.logspace(-4, 2, n_points)
    costs = np.zeros(n_points, dtype=np.float64)
    stds = np.zeros(n_points, dtype=np.float64)

    for i, lam in enumerate(lambda_values):
        traj = almgren_chriss(total_qty, sigma, eta, gamma, lam, n_periods)
        metrics = optimal_execution_cost(traj, sigma, eta, gamma)
        costs[i] = metrics["expected_cost"]
        stds[i] = metrics["std_dev"]

    return {
        "lambda_values": lambda_values,
        "expected_cost": costs,
        "std_dev": stds,
    }
