"""Stochastic process simulation.

Pure numpy implementations of common stochastic processes used in
quantitative finance: GBM, Heston, Merton jump-diffusion,
Ornstein-Uhlenbeck, and Cox-Ingersoll-Ross.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "geometric_brownian_motion",
    "heston",
    "jump_diffusion",
    "ornstein_uhlenbeck",
    "cir_process",
]


def geometric_brownian_motion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Simulate paths of geometric Brownian motion.

    dS = mu * S * dt + sigma * S * dW

    Parameters:
        S0: Initial price.
        mu: Drift rate (annualized).
        sigma: Volatility (annualized).
        T: Time horizon in years.
        n_steps: Number of time steps.
        n_paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_paths, n_steps + 1) with simulated price paths.
        Column 0 is the initial price S0.

    Example:
        >>> paths = geometric_brownian_motion(100, 0.05, 0.2, 1.0, 252, 1000, seed=42)
        >>> paths.shape
        (1000, 253)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0

    z = rng.standard_normal((n_paths, n_steps))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z

    log_returns = drift + diffusion
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))

    return paths


def heston(
    S0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate the Heston stochastic volatility model.

    dS = mu * S * dt + sqrt(v) * S * dW_1
    dv = kappa * (theta - v) * dt + sigma_v * sqrt(v) * dW_2
    corr(dW_1, dW_2) = rho

    Uses the full truncation scheme to ensure non-negative variance.

    Parameters:
        S0: Initial price.
        v0: Initial variance.
        mu: Drift rate (annualized).
        kappa: Mean reversion speed of variance.
        theta: Long-run variance level.
        sigma_v: Volatility of variance (vol of vol).
        rho: Correlation between price and variance Brownian motions.
        T: Time horizon in years.
        n_steps: Number of time steps.
        n_paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (price_paths, vol_paths), each of shape (n_paths, n_steps + 1).

    Example:
        >>> prices, vols = heston(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7, 1.0, 252, 1000, seed=42)
        >>> prices.shape
        (1000, 253)
        >>> vols.shape
        (1000, 253)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    prices = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    variances = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    prices[:, 0] = S0
    variances[:, 0] = v0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        # Correlate the Brownian motions
        w1 = z1
        w2 = rho * z1 + np.sqrt(1.0 - rho**2) * z2

        v_pos = np.maximum(variances[:, t], 0.0)  # Full truncation
        sqrt_v = np.sqrt(v_pos)

        # Update variance
        variances[:, t + 1] = (
            variances[:, t]
            + kappa * (theta - v_pos) * dt
            + sigma_v * sqrt_v * np.sqrt(dt) * w2
        )

        # Update price (log scheme)
        prices[:, t + 1] = prices[:, t] * np.exp(
            (mu - 0.5 * v_pos) * dt + sqrt_v * np.sqrt(dt) * w1
        )

    return prices, variances


def jump_diffusion(
    S0: float,
    mu: float,
    sigma: float,
    lam: float,
    jump_mean: float,
    jump_std: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Simulate the Merton jump-diffusion model.

    dS/S = (mu - lam * k) * dt + sigma * dW + J * dN

    where N is a Poisson process with intensity lam, and
    log(1+J) ~ N(jump_mean, jump_std^2).

    Parameters:
        S0: Initial price.
        mu: Drift rate (annualized, before jump compensation).
        sigma: Diffusion volatility (annualized).
        lam: Jump intensity (expected number of jumps per year).
        jump_mean: Mean of log jump size.
        jump_std: Standard deviation of log jump size.
        T: Time horizon in years.
        n_steps: Number of time steps.
        n_paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_paths, n_steps + 1) with simulated price paths.

    Example:
        >>> paths = jump_diffusion(100, 0.05, 0.2, 1.0, -0.1, 0.15, 1.0, 252, 1000, seed=42)
        >>> paths.shape
        (1000, 253)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Jump compensation: E[e^J - 1] = exp(jump_mean + 0.5*jump_std^2) - 1
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0

    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        n_jumps = rng.poisson(lam * dt, n_paths)

        # Sum of jump sizes for this step
        jump_sum = np.zeros(n_paths, dtype=np.float64)
        for i in range(n_paths):
            if n_jumps[i] > 0:
                jump_sum[i] = np.sum(rng.normal(jump_mean, jump_std, n_jumps[i]))

        # Drift includes jump compensation
        drift = (mu - lam * k - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z

        paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion + jump_sum)

    return paths


def ornstein_uhlenbeck(
    x0: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Simulate the Ornstein-Uhlenbeck (OU) mean-reverting process.

    dx = theta * (mu - x) * dt + sigma * dW

    Parameters:
        x0: Initial value.
        theta: Mean reversion speed.
        mu: Long-run mean level.
        sigma: Volatility.
        T: Time horizon.
        n_steps: Number of time steps.
        n_paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_paths, n_steps + 1) with simulated paths.

    Example:
        >>> paths = ornstein_uhlenbeck(0.05, 5.0, 0.03, 0.01, 1.0, 252, 1000, seed=42)
        >>> paths.shape
        (1000, 253)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = x0

    # Exact discretization of OU process
    exp_theta_dt = np.exp(-theta * dt)
    mean_coeff = 1.0 - exp_theta_dt
    std_coeff = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta))

    z = rng.standard_normal((n_paths, n_steps))

    for t in range(n_steps):
        paths[:, t + 1] = (
            paths[:, t] * exp_theta_dt + mu * mean_coeff + std_coeff * z[:, t]
        )

    return paths


def cir_process(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Simulate the Cox-Ingersoll-Ross (CIR) process.

    dx = kappa * (theta - x) * dt + sigma * sqrt(x) * dW

    When the Feller condition (2*kappa*theta >= sigma^2) is satisfied,
    the process stays strictly positive.

    Parameters:
        x0: Initial value (must be positive).
        kappa: Mean reversion speed.
        theta: Long-run mean level.
        sigma: Volatility coefficient.
        T: Time horizon.
        n_steps: Number of time steps.
        n_paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_paths, n_steps + 1) with simulated paths.

    Example:
        >>> paths = cir_process(0.04, 2.0, 0.04, 0.1, 1.0, 252, 1000, seed=42)
        >>> paths.shape
        (1000, 253)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = x0

    z = rng.standard_normal((n_paths, n_steps))

    for t in range(n_steps):
        x_pos = np.maximum(paths[:, t], 0.0)
        sqrt_x = np.sqrt(x_pos)
        paths[:, t + 1] = (
            paths[:, t]
            + kappa * (theta - x_pos) * dt
            + sigma * sqrt_x * np.sqrt(dt) * z[:, t]
        )
        # Reflection to keep non-negative
        paths[:, t + 1] = np.abs(paths[:, t + 1])

    return paths
