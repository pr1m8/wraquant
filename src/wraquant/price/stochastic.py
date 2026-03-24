"""Stochastic process simulation.

Pure numpy implementations of common stochastic processes used in
quantitative finance: GBM, Heston, Merton jump-diffusion,
Ornstein-Uhlenbeck, Cox-Ingersoll-Ross, SABR, rough Bergomi,
3/2 model, and Vasicek.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from wraquant.core._coerce import coerce_array

__all__ = [
    "geometric_brownian_motion",
    "heston",
    "jump_diffusion",
    "ornstein_uhlenbeck",
    "cir_process",
    "simulate_sabr",
    "simulate_rough_bergomi",
    "simulate_3_2_model",
    "simulate_cir",
    "simulate_vasicek",
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
    r"""Simulate paths of geometric Brownian motion (GBM).

    GBM is the standard model for equity price dynamics and underlies
    the Black-Scholes framework.  The SDE is:

    .. math::

        dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t

    The solution is log-normal:
    :math:`S_T = S_0 \exp\bigl((\mu - \sigma^2/2)T + \sigma W_T\bigr)`.

    Use GBM for quick scenario generation, Monte Carlo option pricing,
    or as a baseline against more complex processes (Heston, jump
    diffusion).

    Parameters:
        S0 (float): Initial price.
        mu (float): Drift rate (annualized).  Under the risk-neutral
            measure, set ``mu = r`` (the risk-free rate).
        sigma (float): Volatility (annualized).
        T (float): Time horizon in years.
        n_steps (int): Number of time steps.  252 for daily resolution.
        n_paths (int): Number of simulation paths.
        seed (int | None): Random seed for reproducibility.

    Returns:
        ndarray: Array of shape ``(n_paths, n_steps + 1)`` with
            simulated price paths.  Column 0 is the initial price S0.

    Example:
        >>> paths = geometric_brownian_motion(100, 0.05, 0.2, 1.0, 252, 1000, seed=42)
        >>> paths.shape
        (1000, 253)

    See Also:
        heston: Stochastic volatility extension of GBM.
        jump_diffusion: GBM with Poisson-driven jumps.
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
    r"""Simulate the Heston (1993) stochastic volatility model.

    The Heston model captures the **leverage effect** (negative
    correlation between returns and volatility) and generates
    realistic volatility smiles.  The coupled SDEs are:

    .. math::

        dS_t &= \mu\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^1 \\
        dv_t &= \kappa(\theta - v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^2 \\
        \text{corr}(dW^1, dW^2) &= \rho

    Uses the **full truncation** scheme (replace negative variance
    with zero before computing diffusion) to ensure non-negative
    variance in the discretisation.

    Use Heston when you need to capture the volatility smile or when
    constant-volatility GBM is insufficient.

    Parameters:
        S0 (float): Initial price.
        v0 (float): Initial variance (e.g., 0.04 for 20% vol).
        mu (float): Drift rate (annualized).
        kappa (float): Mean reversion speed of variance.
        theta (float): Long-run variance level.
        sigma_v (float): Volatility of variance (vol of vol).
        rho (float): Correlation between price and variance Brownian
            motions.  Typically negative for equities (-0.7 to -0.3).
        T (float): Time horizon in years.
        n_steps (int): Number of time steps.
        n_paths (int): Number of simulation paths.
        seed (int | None): Random seed for reproducibility.

    Returns:
        tuple[ndarray, ndarray]: ``(price_paths, vol_paths)``, each of
            shape ``(n_paths, n_steps + 1)``.

    Example:
        >>> prices, vols = heston(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7, 1.0, 252, 1000, seed=42)
        >>> prices.shape
        (1000, 253)

    Notes:
        The Feller condition :math:`2\kappa\theta \geq \sigma_v^2`
        ensures variance stays strictly positive in continuous time.
        Even when violated, the full truncation scheme keeps the
        discretisation non-negative.

    See Also:
        geometric_brownian_motion: Constant-vol baseline.
        heston_characteristic: Analytical characteristic function for
            Fourier-based Heston option pricing.

    References:
        Heston, S.L. (1993). *A Closed-Form Solution for Options with
        Stochastic Volatility.* Review of Financial Studies 6(2).
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
    r"""Simulate the Merton (1976) jump-diffusion model.

    Extends GBM with random jumps driven by a compound Poisson
    process.  This captures sudden large moves (crashes, earnings
    surprises) that GBM cannot reproduce:

    .. math::

        \frac{dS}{S} = (\mu - \lambda k)\,dt + \sigma\,dW + J\,dN

    where :math:`N` is a Poisson process with intensity
    :math:`\lambda`, :math:`\ln(1+J) \sim N(\mu_J, \sigma_J^2)`,
    and :math:`k = E[e^J - 1]` is the drift compensation.

    Use jump diffusion when you need to model **fat tails** and
    **sudden discontinuities** in asset prices, or to capture the
    implied volatility smile steepening at short maturities.

    Parameters:
        S0 (float): Initial price.
        mu (float): Drift rate (annualized, before jump compensation).
        sigma (float): Diffusion volatility (annualized).
        lam (float): Jump intensity (expected number of jumps per year).
        jump_mean (float): Mean of log-jump size.  Negative values
            model downward crashes.
        jump_std (float): Standard deviation of log-jump size.
        T (float): Time horizon in years.
        n_steps (int): Number of time steps.
        n_paths (int): Number of simulation paths.
        seed (int | None): Random seed for reproducibility.

    Returns:
        ndarray: Array of shape ``(n_paths, n_steps + 1)`` with
            simulated price paths.

    Example:
        >>> paths = jump_diffusion(100, 0.05, 0.2, 1.0, -0.1, 0.15, 1.0, 252, 1000, seed=42)
        >>> paths.shape
        (1000, 253)

    See Also:
        geometric_brownian_motion: Diffusion-only baseline.
        heston: Stochastic volatility without jumps.

    References:
        Merton, R.C. (1976). *Option Pricing When Underlying Stock
        Returns Are Discontinuous.* Journal of Financial Economics 3.
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
    r"""Simulate the Ornstein-Uhlenbeck (OU) mean-reverting process.

    The OU process is the canonical continuous-time mean-reverting
    model, widely used for interest rates, pairs-trading spreads,
    and volatility dynamics:

    .. math::

        dX_t = \theta\,(\mu - X_t)\,dt + \sigma\,dW_t

    This implementation uses the **exact discretisation** (not Euler)
    so it is accurate for any step size.

    Use OU for modelling quantities that revert to a long-run level:
    interest rate spreads, log-volatility, or cointegrated pairs.

    Parameters:
        x0 (float): Initial value.
        theta (float): Mean reversion speed.  Higher values mean faster
            reversion.  The half-life is :math:`\ln(2) / \theta`.
        mu (float): Long-run mean level.
        sigma (float): Volatility (diffusion coefficient).
        T (float): Time horizon.
        n_steps (int): Number of time steps.
        n_paths (int): Number of simulation paths.
        seed (int | None): Random seed for reproducibility.

    Returns:
        ndarray: Array of shape ``(n_paths, n_steps + 1)`` with
            simulated paths.

    Example:
        >>> paths = ornstein_uhlenbeck(0.05, 5.0, 0.03, 0.01, 1.0, 252, 1000, seed=42)
        >>> paths.shape
        (1000, 253)

    See Also:
        cir_process: Mean-reverting process with non-negative constraint.
        simulate_vasicek: OU-based interest rate model with bond pricing.
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
    r"""Simulate the Cox-Ingersoll-Ross (CIR) process.

    The CIR process is the standard mean-reverting, non-negative
    model for interest rates and variance:

    .. math::

        dX_t = \kappa(\theta - X_t)\,dt + \sigma\sqrt{X_t}\,dW_t

    The :math:`\sqrt{X}` diffusion term ensures the volatility
    decreases as the process approaches zero, preventing negative
    values when the **Feller condition**
    :math:`2\kappa\theta \geq \sigma^2` holds.

    This is a plain-array simulation.  For analytics (bond prices,
    Feller diagnostics), use :func:`simulate_cir` instead.

    Parameters:
        x0 (float): Initial value (must be positive).
        kappa (float): Mean reversion speed.
        theta (float): Long-run mean level.
        sigma (float): Volatility coefficient.
        T (float): Time horizon.
        n_steps (int): Number of time steps.
        n_paths (int): Number of simulation paths.
        seed (int | None): Random seed for reproducibility.

    Returns:
        ndarray: Array of shape ``(n_paths, n_steps + 1)`` with
            simulated paths.  Values are kept non-negative via
            reflection.

    Example:
        >>> paths = cir_process(0.04, 2.0, 0.04, 0.1, 1.0, 252, 1000, seed=42)
        >>> paths.shape
        (1000, 253)

    See Also:
        simulate_cir: CIR with Feller diagnostics and bond pricing.
        ornstein_uhlenbeck: Gaussian mean-reverting process (can go
            negative).
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


def simulate_sabr(
    f0: float,
    sigma0: float,
    alpha: float,
    beta: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    r"""Simulate the SABR stochastic volatility model.

    The SABR model (Hagan et al., 2002) is the industry standard for
    modelling interest rate and FX volatility smiles:

    .. math::

        dF_t &= \sigma_t\,F_t^{\beta}\,dW_t^1 \\
        d\sigma_t &= \alpha\,\sigma_t\,dW_t^2 \\
        \text{corr}(dW^1, dW^2) &= \rho

    Key parameters:
      - :math:`\beta \in [0, 1]` controls the backbone: 0 = normal,
        1 = lognormal, in between = CEV.
      - :math:`\alpha` is the vol-of-vol.
      - :math:`\rho` controls the skew direction.

    **When to use SABR:** Primarily for interest rate derivatives
    (swaptions, caps/floors) and FX options where the smile shape
    is well-characterised by the SABR formula.

    Parameters:
        f0: Initial forward rate / price.
        sigma0: Initial stochastic volatility level.
        alpha: Volatility of volatility (vol-of-vol).
        beta: CEV exponent (0 = normal, 1 = lognormal).
        rho: Correlation between forward and vol Brownian motions.
        T: Time horizon in years.
        n_steps: Number of time steps.
        n_paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:

        * **forwards** -- simulated forward paths, shape
          ``(n_steps + 1, n_paths)``.
        * **vols** -- simulated stochastic vol paths, shape
          ``(n_steps + 1, n_paths)``.

    Example:
        >>> result = simulate_sabr(0.05, 0.3, 0.4, 0.5, -0.3, 1.0, 252, 1000, seed=42)
        >>> result['forwards'].shape
        (253, 1000)
        >>> result['vols'].shape
        (253, 1000)

    References:
        Hagan, P.S., Kumar, D., Lesniewski, A.S. & Woodward, D.E. (2002).
        *Managing Smile Risk.*  Wilmott Magazine, 84-108.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    forwards = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    vols = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    forwards[0] = f0
    vols[0] = sigma0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        # Correlate Brownian motions
        w1 = z1
        w2 = rho * z1 + np.sqrt(1.0 - rho**2) * z2

        sig = vols[t]
        f = forwards[t]

        # Forward: absorb at zero to prevent negative forwards
        f_pos = np.maximum(f, 1e-10)
        forwards[t + 1] = f + sig * f_pos**beta * sqrt_dt * w1
        forwards[t + 1] = np.maximum(forwards[t + 1], 0.0)

        # Vol: lognormal dynamics
        vols[t + 1] = sig * np.exp(-0.5 * alpha**2 * dt + alpha * sqrt_dt * w2)
        vols[t + 1] = np.maximum(vols[t + 1], 1e-12)

    return {
        "forwards": forwards,
        "vols": vols,
    }


def simulate_rough_bergomi(
    spot: float,
    xi: float,
    eta: float,
    H: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    r"""Simulate the rough Bergomi stochastic volatility model.

    The rough Bergomi model (Bayer, Friz & Gatheral, 2016) is the
    state-of-the-art for capturing the **rough nature of volatility**.
    Empirical studies show that log-volatility behaves like a fractional
    Brownian motion (fBM) with Hurst parameter H ~ 0.1, much rougher
    than classical models (H = 0.5 for standard BM).

    The model specifies:

    .. math::

        dS_t &= \sqrt{v_t}\,S_t\,dW_t^1 \\
        v_t &= \xi\,\mathcal{E}\bigl(\eta\,\tilde{W}_t^H\bigr)

    where :math:`\tilde{W}^H` is a (Riemann-Liouville) fractional
    Brownian motion with Hurst parameter :math:`H < 0.5`, and
    :math:`\mathcal{E}` is the Wick exponential
    :math:`\exp(x - \frac{1}{2}\text{Var}(x))`.

    The fBM is simulated via the Cholesky method on the exact covariance
    matrix, which is accurate but O(n_steps^2) in memory.

    **When to use rough Bergomi:** When you need to capture the term
    structure of at-the-money skew, which decays like :math:`T^{H-1/2}`
    as observed in markets.  Classical models (Heston, SABR) cannot
    reproduce this power-law behaviour.

    Parameters:
        spot: Initial spot price.
        xi: Forward variance level (flat forward variance curve).
        eta: Volatility of volatility.
        H: Hurst parameter of fractional Brownian motion.
            Must be in (0, 0.5) for the "rough" regime.
        rho: Correlation between the spot and vol driving noises.
        T: Time horizon in years.
        n_steps: Number of time steps.
        n_paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:

        * **prices** -- simulated asset price paths, shape
          ``(n_steps + 1, n_paths)``.
        * **variances** -- simulated instantaneous variance paths,
          shape ``(n_steps + 1, n_paths)``.

    Example:
        >>> result = simulate_rough_bergomi(100.0, 0.04, 1.9, 0.1, -0.7,
        ...                                1.0, 100, 500, seed=42)
        >>> result['prices'].shape
        (101, 500)
        >>> result['variances'].shape
        (101, 500)

    References:
        Bayer, C., Friz, P. & Gatheral, J. (2016). *Pricing Under Rough
        Volatility.*  Quantitative Finance 16(6), 887-904.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # ------------------------------------------------------------------
    # Build fBM covariance matrix and Cholesky factor
    # ------------------------------------------------------------------
    times = np.arange(1, n_steps + 1) * dt

    # Covariance of the Riemann-Liouville fBM increments
    # Using the exact covariance: Cov(W^H_s, W^H_t) =
    #   0.5 * (|s|^{2H} + |t|^{2H} - |t-s|^{2H})
    cov_matrix = np.empty((n_steps, n_steps), dtype=np.float64)
    for i in range(n_steps):
        for j in range(n_steps):
            ti = times[i]
            tj = times[j]
            cov_matrix[i, j] = 0.5 * (
                ti ** (2.0 * H) + tj ** (2.0 * H) - abs(ti - tj) ** (2.0 * H)
            )

    # Regularise for numerical stability
    cov_matrix += np.eye(n_steps) * 1e-10
    L = np.linalg.cholesky(cov_matrix)

    # ------------------------------------------------------------------
    # Simulate paths
    # ------------------------------------------------------------------
    prices = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    variances = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    prices[0] = spot
    variances[0] = xi

    for p in range(n_paths):
        # Generate correlated fBM and standard BM increments
        z_fbm = rng.standard_normal(n_steps)
        z_spot = rng.standard_normal(n_steps)

        # fBM path via Cholesky
        W_H = L @ z_fbm  # fBM values at each time step

        # Variance of W_H for the Wick exponential
        var_W_H = np.array([cov_matrix[i, i] for i in range(n_steps)])

        # Instantaneous variance: v_t = xi * exp(eta * W_H_t - 0.5 * eta^2 * Var(W_H_t))
        v = xi * np.exp(eta * W_H - 0.5 * eta**2 * var_W_H)
        variances[1:, p] = v

        # Correlated spot BM increments
        dW_spot = rho * z_fbm + np.sqrt(1.0 - rho**2) * z_spot

        # Simulate spot price (log-Euler)
        S = spot
        for t in range(n_steps):
            sqrt_v = np.sqrt(max(v[t], 0.0))
            S = S * np.exp(-0.5 * v[t] * dt + sqrt_v * sqrt_dt * dW_spot[t])
            prices[t + 1, p] = S

    return {
        "prices": prices,
        "variances": variances,
    }


def simulate_3_2_model(
    spot: float,
    v0: float,
    kappa: float,
    theta: float,
    epsilon: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    r"""Simulate the 3/2 stochastic volatility model.

    The 3/2 model features vol-of-vol that increases with the variance
    level, unlike Heston where vol-of-vol is constant.  This makes it
    better suited for markets where volatility becomes extremely volatile
    during high-vol regimes (e.g., VIX dynamics):

    .. math::

        dS_t &= \sqrt{v_t}\,S_t\,dW_t^1 \\
        dv_t &= \kappa\,v_t(\theta - v_t)\,dt
               + \varepsilon\,v_t^{3/2}\,dW_t^2 \\
        \text{corr}(dW^1, dW^2) &= \rho

    The key feature is the :math:`v^{3/2}` diffusion term: when variance
    is high, the variance process itself becomes much more volatile,
    capturing the empirical observation that vol-of-vol spikes during
    market crises.

    **When to use the 3/2 model:** For VIX derivatives, or when the
    Heston model under-estimates the vol-of-vol in high-vol regimes.
    Also useful when calibrating to options on variance swaps.

    Parameters:
        spot: Initial asset price.
        v0: Initial variance.
        kappa: Mean reversion speed (in the V dynamics).
        theta: Long-run variance target.
        epsilon: Vol-of-vol parameter (coefficient on V^{3/2}).
        rho: Correlation between asset and variance Brownian motions.
        T: Time horizon in years.
        n_steps: Number of time steps.
        n_paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:

        * **prices** -- simulated asset price paths, shape
          ``(n_steps + 1, n_paths)``.
        * **variances** -- simulated variance paths, shape
          ``(n_steps + 1, n_paths)``.

    Example:
        >>> result = simulate_3_2_model(100.0, 0.04, 2.0, 0.04, 0.5, -0.7,
        ...                             1.0, 252, 1000, seed=42)
        >>> result['prices'].shape
        (253, 1000)
        >>> result['variances'].shape
        (253, 1000)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    prices = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    variances = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    prices[0] = spot
    variances[0] = v0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w1 = z1
        w2 = rho * z1 + np.sqrt(1.0 - rho**2) * z2

        v = np.maximum(variances[t], 1e-12)
        sqrt_v = np.sqrt(v)

        # Variance dynamics: dV = kappa * V * (theta - V) dt + epsilon * V^{3/2} dW
        variances[t + 1] = (
            v + kappa * v * (theta - v) * dt + epsilon * v**1.5 * sqrt_dt * w2
        )
        variances[t + 1] = np.maximum(variances[t + 1], 1e-12)

        # Price dynamics (log-Euler)
        prices[t + 1] = prices[t] * np.exp(-0.5 * v * dt + sqrt_v * sqrt_dt * w1)

    return {
        "prices": prices,
        "variances": variances,
    }


def simulate_cir(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> dict[str, object]:
    r"""Simulate the Cox-Ingersoll-Ross short-rate model with analytics.

    The CIR model is the standard mean-reverting, non-negative process
    for interest rates:

    .. math::

        dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t

    The **Feller condition** :math:`2\kappa\theta \geq \sigma^2` ensures
    that the process stays strictly positive.  When violated, zero is
    accessible but reflecting.

    Unlike :func:`cir_process` (which returns a plain array), this
    function also computes:
    - The Feller condition diagnostic
    - Analytical zero-coupon bond prices P(0, T)
    - The mean reversion check

    **When to use CIR:** For modelling interest rates, credit intensities,
    or stochastic variance (it is the variance process in Heston).  The
    guaranteed non-negativity is crucial for these applications.

    Parameters:
        r0: Initial short rate (must be positive).
        kappa: Mean reversion speed.
        theta: Long-run mean level.
        sigma: Volatility coefficient.
        T: Time horizon in years.
        n_steps: Number of time steps.
        n_paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:

        * **paths** -- simulated rate paths, shape
          ``(n_steps + 1, n_paths)``.
        * **params** -- dictionary of model parameters including:
          - *kappa*, *theta*, *sigma* -- model parameters.
          - *feller_satisfied* -- whether 2*kappa*theta >= sigma^2.
          - *feller_ratio* -- 2*kappa*theta / sigma^2 (>= 1 means satisfied).

    Example:
        >>> result = simulate_cir(0.05, 0.5, 0.04, 0.1, 10.0, 252, 1000, seed=42)
        >>> result['paths'].shape
        (253, 1000)
        >>> result['params']['feller_satisfied']
        True
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    feller_ratio = 2.0 * kappa * theta / (sigma**2) if sigma > 0 else float("inf")
    feller_satisfied = feller_ratio >= 1.0

    paths = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    paths[0] = r0

    z = rng.standard_normal((n_steps, n_paths))

    for t in range(n_steps):
        r_pos = np.maximum(paths[t], 0.0)
        sqrt_r = np.sqrt(r_pos)
        paths[t + 1] = (
            paths[t]
            + kappa * (theta - r_pos) * dt
            + sigma * sqrt_r * np.sqrt(dt) * z[t]
        )
        # Reflection to keep non-negative
        paths[t + 1] = np.abs(paths[t + 1])

    return {
        "paths": paths,
        "params": {
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma,
            "feller_satisfied": feller_satisfied,
            "feller_ratio": feller_ratio,
        },
    }


def simulate_vasicek(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> dict[str, object]:
    r"""Simulate the Vasicek interest rate model with bond pricing.

    The Vasicek (1977) model is the simplest mean-reverting Gaussian
    model for the short rate:

    .. math::

        dr_t = \kappa(\theta - r_t)\,dt + \sigma\,dW_t

    Being Gaussian, the short rate can go negative -- this was historically
    seen as a drawback but is now relevant in the era of negative rates.

    The model admits closed-form solutions for zero-coupon bond prices:

    .. math::

        P(0, T) = A(T)\,\exp(-B(T)\,r_0)

    where:

    .. math::

        B(T) &= \frac{1 - e^{-\kappa T}}{\kappa} \\
        A(T) &= \exp\left[
            \left(\theta - \frac{\sigma^2}{2\kappa^2}\right)(B(T) - T)
            - \frac{\sigma^2}{4\kappa}B(T)^2
        \right]

    **When to use Vasicek:** Quick-and-dirty interest rate modelling,
    or when negative rates are possible/desired.  For non-negative
    rates, use CIR instead.

    Parameters:
        r0: Initial short rate.
        kappa: Mean reversion speed.
        theta: Long-run mean level.
        sigma: Volatility.
        T: Time horizon in years.
        n_steps: Number of time steps.
        n_paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:

        * **paths** -- simulated rate paths, shape
          ``(n_steps + 1, n_paths)``.
        * **bond_prices** -- analytical zero-coupon bond prices
          P(0, t) at each time step, shape ``(n_steps + 1,)``.
        * **yield_curve** -- continuously compounded yields
          y(0, t) = -ln(P(0,t))/t at each time step (excluding t=0),
          shape ``(n_steps,)``.

    Example:
        >>> result = simulate_vasicek(0.05, 0.5, 0.04, 0.01, 10.0, 100, 1000, seed=42)
        >>> result['paths'].shape
        (101, 1000)
        >>> len(result['bond_prices'])
        101
        >>> len(result['yield_curve'])
        100

    References:
        Vasicek, O. (1977). *An Equilibrium Characterisation of the Term
        Structure.*  Journal of Financial Economics 5(2), 177-188.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # ------------------------------------------------------------------
    # Simulate paths using exact discretisation of OU process
    # ------------------------------------------------------------------
    paths = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    paths[0] = r0

    exp_kdt = np.exp(-kappa * dt)
    mean_coeff = 1.0 - exp_kdt
    std_coeff = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))

    z = rng.standard_normal((n_steps, n_paths))

    for t in range(n_steps):
        paths[t + 1] = paths[t] * exp_kdt + theta * mean_coeff + std_coeff * z[t]

    # ------------------------------------------------------------------
    # Analytical bond prices and yield curve
    # ------------------------------------------------------------------
    times = np.arange(n_steps + 1) * dt

    bond_prices = np.ones(n_steps + 1, dtype=np.float64)
    for i in range(1, n_steps + 1):
        t_i = times[i]
        B = (1.0 - np.exp(-kappa * t_i)) / kappa
        A = np.exp(
            (theta - sigma**2 / (2.0 * kappa**2)) * (B - t_i)
            - sigma**2 / (4.0 * kappa) * B**2
        )
        bond_prices[i] = A * np.exp(-B * r0)

    # Yield curve: y = -ln(P)/t  (skip t=0)
    yield_curve = -np.log(bond_prices[1:]) / times[1:]

    return {
        "paths": paths,
        "bond_prices": bond_prices,
        "yield_curve": yield_curve,
    }
