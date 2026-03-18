"""Bayesian online change-point detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def online_changepoint(
    data: pd.Series,
    hazard: float = 0.005,
) -> pd.Series:
    """Bayesian online change-point detection.

    Implements the algorithm from Adams & MacKay (2007). Uses a normal
    model with unknown mean and known variance (estimated from data).

    Parameters:
        data: Time series to monitor.
        hazard: Prior probability that a change-point occurs at each
            time step (constant hazard rate ``1/mean_run_length``).

    Returns:
        Series of estimated run lengths (the most probable run length
        at each time step). A drop to zero indicates a detected
        change-point.
    """
    clean = data.dropna()
    values = clean.values
    T = len(values)

    # Prior parameters (Normal-Gamma conjugate)
    mu0 = values.mean()
    kappa0 = 1.0
    alpha0 = 1.0
    beta0 = float(values.var()) if values.var() > 0 else 1.0

    # Run-length probabilities: R[t, r] = P(r_t = r | data_{1:t})
    # We only keep current and use log probabilities for stability.
    run_length_probs = np.zeros(T + 1)
    run_length_probs[0] = 1.0

    # Sufficient statistics for each run length
    mu_params = np.array([mu0])
    kappa_params = np.array([kappa0])
    alpha_params = np.array([alpha0])
    beta_params = np.array([beta0])

    max_run_lengths = np.zeros(T, dtype=int)

    for t in range(T):
        x = values[t]

        # Predictive probability under each run-length hypothesis
        # Student-t distribution
        df = 2 * alpha_params
        scale = beta_params * (kappa_params + 1) / (alpha_params * kappa_params)
        scale = np.maximum(scale, 1e-10)

        # Evaluate predictive log probability
        pred_probs = _student_t_pdf(x, mu_params, scale, df)

        # Growth probabilities
        growth_probs = run_length_probs[: len(pred_probs)] * pred_probs * (1 - hazard)

        # Change-point probability
        cp_prob = np.sum(run_length_probs[: len(pred_probs)] * pred_probs * hazard)

        # Update run length distribution
        new_run_length_probs = np.zeros(t + 2)
        new_run_length_probs[0] = cp_prob
        new_run_length_probs[1 : len(growth_probs) + 1] = growth_probs

        # Normalise
        total = new_run_length_probs.sum()
        if total > 0:
            new_run_length_probs /= total
        run_length_probs = new_run_length_probs

        # Update sufficient statistics
        new_mu = (kappa_params * mu_params + x) / (kappa_params + 1)
        new_kappa = kappa_params + 1
        new_alpha = alpha_params + 0.5
        new_beta = beta_params + kappa_params * (x - mu_params) ** 2 / (
            2 * (kappa_params + 1)
        )

        # Prepend prior for new run (r=0)
        mu_params = np.concatenate([[mu0], new_mu])
        kappa_params = np.concatenate([[kappa0], new_kappa])
        alpha_params = np.concatenate([[alpha0], new_alpha])
        beta_params = np.concatenate([[beta0], new_beta])

        max_run_lengths[t] = int(np.argmax(run_length_probs))

    return pd.Series(max_run_lengths, index=clean.index, name="run_length")


def _student_t_pdf(
    x: float,
    mu: np.ndarray,
    scale: np.ndarray,
    df: np.ndarray,
) -> np.ndarray:
    """Evaluate the Student-t PDF for vectorised parameters."""
    from scipy.special import gammaln

    z = (x - mu) ** 2 / scale
    log_prob = (
        gammaln((df + 1) / 2)
        - gammaln(df / 2)
        - 0.5 * np.log(np.pi * df * scale)
        - (df + 1) / 2 * np.log1p(z / df)
    )
    return np.exp(log_prob)
