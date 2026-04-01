"""Forward-Backward Stochastic Differential Equation (FBSDE) solvers.

FBSDEs are the modern mathematical framework for derivatives pricing.
The **forward SDE** models the dynamics of the underlying asset, while
the **backward SDE** (BSDE) simultaneously gives the option price *Y*
and the hedge ratio *Z* at every point in time.

The coupled system is:

    Forward:   dX_t = mu(t, X_t) dt + sigma(t, X_t) dW_t
    Backward:  dY_t = -f(t, X_t, Y_t, Z_t) dt + Z_t dW_t
    Terminal:  Y_T  = g(X_T)

By the **Feynman-Kac theorem**, the solution (Y_t, Z_t) of the BSDE
is related to the PDE solution u(t, x) of the associated parabolic PDE:

    Y_t = u(t, X_t),   Z_t = sigma(t, X_t) * nabla_x u(t, X_t)

In the Black-Scholes setting the driver f = -r*Y (risk-free discounting)
and the terminal condition g is the payoff.  The Z process is then
sigma * S * Delta, recovering the classical hedge ratio.

References:
    - El Karoui, Peng, Quenez (1997). *Backward Stochastic Differential
      Equations in Finance*.  Mathematical Finance 7(1), 1-71.
    - Pardoux & Peng (1990). *Adapted solution of a backward stochastic
      differential equation*.  Systems & Control Letters 14(1), 55-61.
    - E, Han, Jentzen (2017). *Deep Learning-Based Numerical Methods for
      High-Dimensional Parabolic PDEs and BSDEs*.  arXiv:1706.04702.

All implementations are pure numpy/scipy (with optional torch for deep BSDE).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from wraquant.core._coerce import coerce_array  # noqa: F401 — wired for type-system consistency

__all__ = [
    "fbsde_european",
    "deep_bsde",
    "reflected_bsde",
]


# ---------------------------------------------------------------------------
# FBSDE European option solver
# ---------------------------------------------------------------------------

def fbsde_european(
    spot: float,
    payoff_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    drift_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    vol_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    rf: float,
    T: float,
    n_steps: int = 100,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> dict[str, object]:
    r"""Solve a forward-backward SDE for European option pricing.

    The FBSDE system couples the asset dynamics (forward SDE) with the
    pricing equation (backward SDE):

    .. math::

        \text{Forward:} \quad dX_t &= \mu(X_t)\,dt + \sigma(X_t)\,dW_t \\
        \text{Backward:} \quad dY_t &= -f(t,X_t,Y_t,Z_t)\,dt + Z_t\,dW_t \\
        \text{Terminal:} \quad Y_T &= g(X_T)

    Under risk-neutral pricing the driver is ``f = -r * Y`` (discounting).
    The forward SDE is discretised with Euler-Maruyama and the backward
    component is solved via least-squares regression at each time step
    (Longstaff-Schwartz style), projecting the continuation value onto
    polynomial basis functions of the forward process.

    The **Z process** recovered from regression is proportional to
    ``sigma(X) * Delta``, giving the hedge ratio directly from the BSDE
    without any finite-difference bumping.

    When ``drift_fn = lambda x: r * x`` and ``vol_fn = lambda x: sigma * x``
    with constant sigma, this reduces to the Black-Scholes model and the
    price converges to the analytical formula.

    Parameters:
        spot: Initial value of the forward process X_0.
        payoff_fn: Terminal condition g(X_T).  Callable mapping an array of
            terminal values (shape ``(n_paths,)``) to payoffs.
        drift_fn: Drift coefficient mu(x) of the forward SDE.  Callable
            mapping an array of current values to drift values.
        vol_fn: Diffusion coefficient sigma(x) of the forward SDE.  Callable
            mapping an array of current values to volatility values.
        rf: Risk-free rate (used as the BSDE driver: f = -r * Y).
        T: Time to maturity in years.
        n_steps: Number of Euler-Maruyama time steps.
        n_paths: Number of Monte Carlo paths.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:

        * **price** -- estimated option price (mean of Y_0).
        * **delta** -- estimated hedge ratio at t=0 (mean of Z_0 / vol(X_0)).
        * **paths** -- forward process paths, shape ``(n_steps + 1, n_paths)``.
        * **price_process** -- Y process, shape ``(n_steps + 1, n_paths)``.

    Example:
        >>> import numpy as np
        >>> # Black-Scholes European call
        >>> S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        >>> payoff = lambda x: np.maximum(x - K, 0.0)
        >>> drift = lambda x: r * x
        >>> vol = lambda x: sigma * x
        >>> result = fbsde_european(S0, payoff, drift, vol, r, T,
        ...                         n_steps=100, n_paths=50000, seed=42)
        >>> 8.0 < result['price'] < 13.0
        True
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # ------------------------------------------------------------------
    # Forward pass: simulate X_t via Euler-Maruyama
    # ------------------------------------------------------------------
    X = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    X[0] = spot

    # Store Brownian increments for backward pass
    dW = np.empty((n_steps, n_paths), dtype=np.float64)

    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        dW[t] = sqrt_dt * z
        mu_val = drift_fn(X[t])
        sig_val = vol_fn(X[t])
        X[t + 1] = X[t] + mu_val * dt + sig_val * dW[t]

    # ------------------------------------------------------------------
    # Backward pass: solve BSDE via regression
    # ------------------------------------------------------------------
    Y = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    Z = np.empty((n_steps, n_paths), dtype=np.float64)

    # Terminal condition
    Y[n_steps] = payoff_fn(X[n_steps])

    for t in range(n_steps - 1, -1, -1):
        # Discounted continuation value
        Y_next = Y[t + 1]
        # The BSDE driver: dY = -f dt + Z dW  with f = -r*Y
        # Backward Euler: Y_t = Y_{t+1} + f(t+1,...) * dt - Z_t * dW_t
        # Rearranged: Y_t = Y_{t+1} - r * Y_{t+1} * dt - Z_t * dW_t
        #           = Y_{t+1} * (1 - r*dt) - Z_t * dW_t

        # Build polynomial basis of X_t for regression
        x = X[t]
        x_std = np.std(x)
        if x_std < 1e-12:
            x_std = 1.0
        x_norm = (x - np.mean(x)) / x_std

        # Polynomial basis: 1, x, x^2, x^3
        basis = np.column_stack([
            np.ones(n_paths),
            x_norm,
            x_norm ** 2,
            x_norm ** 3,
        ])

        # Regression target for Z: from dW relationship
        # Y_{t+1} = Y_t - f_t * dt + Z_t * dW_t
        # => Z_t * dW_t = Y_{t+1} - Y_t + f_t * dt
        # We estimate Z_t by regressing Y_{t+1} * dW_t / dt on basis
        # (since E[Y_{t+1} * dW_t | X_t] = Z_t * dt)
        z_target = Y_next * dW[t] / dt

        # Solve least squares for Z
        from wraquant.stats.regression import ols

        result_z = ols(z_target, basis, add_constant=False)
        coeffs_z = result_z["coefficients"]
        Z[t] = basis @ coeffs_z

        # Compute Y_t using the backward Euler scheme
        Y[t] = Y_next * (1.0 - rf * dt) - Z[t] * dW[t]

    # ------------------------------------------------------------------
    # Extract price and delta at t=0
    # ------------------------------------------------------------------
    price = float(np.mean(Y[0]))
    sig_0 = vol_fn(np.array([spot]))[0] if callable(vol_fn) else spot
    if abs(sig_0) > 1e-12:
        delta = float(np.mean(Z[0]) / sig_0)
    else:
        delta = 0.0

    return {
        "price": price,
        "delta": delta,
        "paths": X,
        "price_process": Y,
    }


# ---------------------------------------------------------------------------
# Deep BSDE solver
# ---------------------------------------------------------------------------

def deep_bsde(
    dim: int,
    payoff_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    drift_fn: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    vol_fn: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    rf: float,
    T: float,
    n_steps: int = 50,
    n_paths: int = 4096,
    n_epochs: int = 200,
    lr: float = 1e-3,
    seed: int | None = None,
) -> dict[str, object]:
    r"""Deep BSDE solver for high-dimensional derivative pricing.

    Implements the algorithm of E, Han & Jentzen (2017) for solving
    BSDEs using deep neural networks.  This is the state-of-the-art
    method for pricing derivatives on baskets of >3 assets, where
    PDE grid methods suffer from the curse of dimensionality.

    The key idea: parameterise the initial value Y_0 and the Z process
    at each time step with neural networks (or simpler function
    approximators).  Then minimise the terminal loss:

    .. math::

        \mathcal{L} = \mathbb{E}\bigl[\bigl|
            Y_T^{\theta} - g(X_T)
        \bigr|^2\bigr]

    where Y_T^theta is obtained by rolling forward the discretised BSDE
    with the learned Z networks.

    When ``torch`` is available, uses a proper neural network with Adam
    optimiser.  Otherwise falls back to a simplified linear approximation
    with ``scipy.optimize.minimize``.

    Use for problems with >3 dimensions where PDE methods fail due to
    the curse of dimensionality.

    Parameters:
        dim: Number of underlying assets (spatial dimension).
        payoff_fn: Terminal condition g(X_T).  Callable mapping an array
            of shape ``(n_paths, dim)`` to payoffs of shape ``(n_paths,)``.
        drift_fn: Drift mu(X).  Callable mapping ``(n_paths, dim)`` to
            ``(n_paths, dim)``.
        vol_fn: Diffusion sigma(X).  Callable mapping ``(n_paths, dim)``
            to ``(n_paths, dim)``.  Assumed diagonal diffusion.
        rf: Risk-free rate.
        T: Time to maturity in years.
        n_steps: Number of time steps.
        n_paths: Number of Monte Carlo paths per epoch.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:

        * **price** -- estimated option price (Y_0).
        * **delta** -- estimated hedge ratios at t=0, shape ``(dim,)``.
        * **loss_history** -- training loss at each epoch.

    Example:
        >>> import numpy as np
        >>> # 1D Black-Scholes call (use fbsde_european for 1D; this is for demo)
        >>> payoff = lambda x: np.maximum(np.mean(x, axis=1) - 100, 0.0)
        >>> drift = lambda x: 0.05 * x
        >>> vol = lambda x: 0.2 * x
        >>> result = deep_bsde(1, payoff, drift, vol, 0.05, 1.0,
        ...                    n_steps=20, n_paths=512, n_epochs=50, seed=42)
        >>> result['price'] > 0
        True

    References:
        E, W., Han, J., & Jentzen, A. (2017). *Deep Learning-Based
        Numerical Methods for High-Dimensional Parabolic Partial
        Differential Equations and Backward Stochastic Differential
        Equations.* Communications in Mathematics and Statistics, 5(4).
    """
    try:
        return _deep_bsde_torch(
            dim, payoff_fn, drift_fn, vol_fn, rf, T,
            n_steps, n_paths, n_epochs, lr, seed,
        )
    except ImportError:
        return _deep_bsde_scipy(
            dim, payoff_fn, drift_fn, vol_fn, rf, T,
            n_steps, n_paths, n_epochs, lr, seed,
        )


def _deep_bsde_torch(
    dim: int,
    payoff_fn: Callable,
    drift_fn: Callable,
    vol_fn: Callable,
    rf: float,
    T: float,
    n_steps: int,
    n_paths: int,
    n_epochs: int,
    lr: float,
    seed: int | None,
) -> dict[str, object]:
    """Deep BSDE implementation using PyTorch."""
    import torch
    import torch.nn as nn

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Learnable initial value Y_0
    y0_param = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    # Learnable Z networks: one small network per time step
    z_nets = nn.ModuleList([
        nn.Sequential(
            nn.Linear(dim, max(dim + 10, 16), dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(max(dim + 10, 16), dim, dtype=torch.float64),
        )
        for _ in range(n_steps)
    ])

    optimizer = torch.optim.Adam(
        list(z_nets.parameters()) + [y0_param], lr=lr,
    )

    loss_history: list[float] = []

    for epoch in range(n_epochs):
        # Simulate forward paths
        dW_np = rng.standard_normal((n_steps, n_paths, dim)) * sqrt_dt
        X_np = np.empty((n_steps + 1, n_paths, dim), dtype=np.float64)
        X_np[0] = np.full((n_paths, dim), 100.0)  # will be overridden below

        # Use first asset spot = 100 as default initial
        x0 = np.full((n_paths, dim), 100.0, dtype=np.float64)
        X_np[0] = x0

        for t in range(n_steps):
            mu_val = drift_fn(X_np[t])
            sig_val = vol_fn(X_np[t])
            X_np[t + 1] = X_np[t] + mu_val * dt + sig_val * dW_np[t]

        # Convert to torch
        dW_t = torch.tensor(dW_np, dtype=torch.float64)
        X_t = torch.tensor(X_np, dtype=torch.float64)

        # Roll forward the BSDE
        Y = y0_param.expand(n_paths)

        for t in range(n_steps):
            z_val = z_nets[t](X_t[t])  # (n_paths, dim)
            # Y_{t+1} = Y_t + r*Y_t*dt + Z_t . dW_t (driver f = -r*Y)
            Y = Y + rf * Y * dt + torch.sum(z_val * dW_t[t], dim=1)

        # Terminal loss
        terminal_payoff = torch.tensor(
            payoff_fn(X_np[n_steps]), dtype=torch.float64,
        )
        loss = torch.mean((Y - terminal_payoff) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.item()))

    # Extract delta from Z_0 network
    with torch.no_grad():
        x0_torch = torch.tensor(x0[:1], dtype=torch.float64)
        z0_val = z_nets[0](x0_torch).numpy().flatten()
        sig_0 = vol_fn(x0[:1]).flatten()
        delta = np.where(np.abs(sig_0) > 1e-12, z0_val / sig_0, 0.0)

    return {
        "price": float(y0_param.item()),
        "delta": delta,
        "loss_history": loss_history,
    }


def _deep_bsde_scipy(
    dim: int,
    payoff_fn: Callable,
    drift_fn: Callable,
    vol_fn: Callable,
    rf: float,
    T: float,
    n_steps: int,
    n_paths: int,
    n_epochs: int,
    lr: float,
    seed: int | None,
) -> dict[str, object]:
    """Fallback deep BSDE using scipy.optimize with linear Z approximation."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Pre-simulate paths for optimisation
    dW = rng.standard_normal((n_steps, n_paths, dim)) * sqrt_dt
    X = np.empty((n_steps + 1, n_paths, dim), dtype=np.float64)
    x0 = np.full((n_paths, dim), 100.0, dtype=np.float64)
    X[0] = x0

    for t in range(n_steps):
        mu_val = drift_fn(X[t])
        sig_val = vol_fn(X[t])
        X[t + 1] = X[t] + mu_val * dt + sig_val * dW[t]

    terminal_payoff = payoff_fn(X[n_steps])

    # Parameters: y0 (1) + z_coeffs per step (dim each) = 1 + n_steps * dim
    n_params = 1 + n_steps * dim

    def objective(params: npt.NDArray[np.float64]) -> float:
        y0_val = params[0]
        Y = np.full(n_paths, y0_val, dtype=np.float64)

        for t in range(n_steps):
            z_vals = params[1 + t * dim: 1 + (t + 1) * dim]
            # Z is constant across paths (linear approximation)
            z_dw = np.sum(z_vals * dW[t], axis=1)
            Y = Y + rf * Y * dt + z_dw

        return float(np.mean((Y - terminal_payoff) ** 2))

    # Initial guess
    p0 = np.zeros(n_params, dtype=np.float64)
    p0[0] = float(np.mean(terminal_payoff) * np.exp(-rf * T))

    result = minimize(objective, p0, method="L-BFGS-B",
                      options={"maxiter": n_epochs, "ftol": 1e-12})

    price = float(result.x[0])
    z0 = result.x[1: 1 + dim]
    sig_0 = vol_fn(x0[:1]).flatten()
    delta = np.where(np.abs(sig_0) > 1e-12, z0 / sig_0, 0.0)

    loss_history = [float(result.fun)]

    return {
        "price": price,
        "delta": delta,
        "loss_history": loss_history,
    }


# ---------------------------------------------------------------------------
# Reflected BSDE for American options
# ---------------------------------------------------------------------------

def reflected_bsde(
    spot: float,
    payoff_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    drift_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    vol_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    rf: float,
    T: float,
    n_steps: int = 100,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> dict[str, object]:
    r"""Solve a reflected BSDE (RBSDE) for American option pricing.

    American options require the price process Y_t to stay **above** the
    intrinsic value (the obstacle) at all times.  This is captured by
    adding a non-decreasing "reflection" process K_t to the BSDE:

    .. math::

        Y_t = g(X_T) + \int_t^T f(s, X_s, Y_s, Z_s)\,ds
              - \int_t^T Z_s\,dW_s + K_T - K_t

    with the constraint :math:`Y_t \geq h(t, X_t)` (obstacle) and
    :math:`K` increases only when :math:`Y_t = h(t, X_t)`.

    This implementation uses the **penalisation method**: at each backward
    step, the continuation value is lifted to at least the exercise value,
    effectively penalising paths that would violate the obstacle.  This
    converges to the true reflected BSDE solution as the time grid is
    refined.

    The resulting price is always >= the European option price, with the
    difference representing the early exercise premium.

    Parameters:
        spot: Initial value of the forward process X_0.
        payoff_fn: Payoff function g(x) used both as terminal condition
            and exercise (obstacle) value at each time step.
        drift_fn: Drift coefficient mu(x) of the forward SDE.
        vol_fn: Diffusion coefficient sigma(x) of the forward SDE.
        rf: Risk-free rate.
        T: Time to maturity in years.
        n_steps: Number of time steps.
        n_paths: Number of Monte Carlo paths.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:

        * **price** -- estimated American option price.
        * **exercise_boundary** -- array of exercise boundary estimates at
          each time step, shape ``(n_steps + 1,)``.
        * **optimal_stopping_time** -- mean optimal stopping time.

    Example:
        >>> import numpy as np
        >>> S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        >>> payoff = lambda x: np.maximum(K - x, 0.0)  # put
        >>> drift = lambda x: r * x
        >>> vol = lambda x: sigma * x
        >>> result = reflected_bsde(S0, payoff, drift, vol, r, T,
        ...                         n_steps=100, n_paths=20000, seed=42)
        >>> result['price'] > 0
        True
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # ------------------------------------------------------------------
    # Forward pass: simulate X_t
    # ------------------------------------------------------------------
    X = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    X[0] = spot

    dW = np.empty((n_steps, n_paths), dtype=np.float64)

    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        dW[t] = sqrt_dt * z
        mu_val = drift_fn(X[t])
        sig_val = vol_fn(X[t])
        X[t + 1] = X[t] + mu_val * dt + sig_val * dW[t]

    # ------------------------------------------------------------------
    # Backward pass: reflected BSDE via penalisation
    # ------------------------------------------------------------------
    Y = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    Z = np.empty((n_steps, n_paths), dtype=np.float64)

    # Terminal condition
    Y[n_steps] = payoff_fn(X[n_steps])

    # Track exercise boundary and stopping times
    exercise_boundary = np.full(n_steps + 1, np.nan, dtype=np.float64)
    stopped = np.zeros(n_paths, dtype=bool)
    stopping_times = np.full(n_paths, T, dtype=np.float64)

    for t in range(n_steps - 1, -1, -1):
        # Continuation value via regression
        Y_next = Y[t + 1]
        x = X[t]
        x_std = np.std(x)
        if x_std < 1e-12:
            x_std = 1.0
        x_norm = (x - np.mean(x)) / x_std

        basis = np.column_stack([
            np.ones(n_paths),
            x_norm,
            x_norm ** 2,
            x_norm ** 3,
        ])

        # Z regression
        z_target = Y_next * dW[t] / dt
        from wraquant.stats.regression import ols as _ols

        result_z = _ols(z_target, basis, add_constant=False)
        coeffs_z = result_z["coefficients"]
        Z[t] = basis @ coeffs_z

        # Continuation value (backward Euler)
        continuation = Y_next * (1.0 - rf * dt) - Z[t] * dW[t]

        # Obstacle: exercise value
        exercise_value = payoff_fn(X[t])

        # Reflection: Y_t = max(continuation, exercise_value)
        Y[t] = np.maximum(continuation, exercise_value)

        # Exercise boundary: approximate as the asset level where
        # exercise_value = continuation (find threshold)
        exercised = exercise_value >= continuation
        if np.any(exercised) and np.any(~exercised):
            # Boundary is approximately the max X where exercise is optimal
            exercise_boundary[t] = float(np.percentile(
                X[t][exercised], 95 if np.mean(X[t][exercised]) > spot else 5,
            ))

        # Track optimal stopping
        newly_stopped = exercised & ~stopped
        stopping_times[newly_stopped] = t * dt
        stopped |= exercised

    price = float(np.mean(Y[0]))
    mean_stopping = float(np.mean(stopping_times))

    return {
        "price": price,
        "exercise_boundary": exercise_boundary,
        "optimal_stopping_time": mean_stopping,
    }
