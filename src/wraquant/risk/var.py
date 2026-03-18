"""Value-at-Risk and Conditional VaR estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Estimate Value-at-Risk (VaR).

    Parameters:
        returns: Simple return series.
        confidence: Confidence level (e.g., 0.95 for 95%).
        method: Estimation method — ``"historical"`` (default) or
            ``"parametric"`` (Gaussian assumption).

    Returns:
        VaR as a positive float representing the loss threshold
        (i.e., the absolute value of the quantile).

    Raises:
        ValueError: If *method* is not recognized.
    """
    clean = returns.dropna().values

    if method == "historical":
        var = float(np.percentile(clean, (1 - confidence) * 100))
    elif method == "parametric":
        mu = clean.mean()
        sigma = clean.std()
        var = float(sp_stats.norm.ppf(1 - confidence, loc=mu, scale=sigma))
    else:
        msg = f"Unknown VaR method: {method!r}"
        raise ValueError(msg)

    return -var


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Estimate Conditional VaR (Expected Shortfall / CVaR).

    Parameters:
        returns: Simple return series.
        confidence: Confidence level (e.g., 0.95 for 95%).
        method: Estimation method — ``"historical"`` (default) or
            ``"parametric"`` (Gaussian assumption).

    Returns:
        CVaR as a positive float representing the expected loss
        beyond the VaR threshold.

    Raises:
        ValueError: If *method* is not recognized.
    """
    clean = returns.dropna().values

    if method == "historical":
        cutoff = np.percentile(clean, (1 - confidence) * 100)
        tail = clean[clean <= cutoff]
        cvar = float(-tail.mean()) if len(tail) > 0 else 0.0
    elif method == "parametric":
        mu = clean.mean()
        sigma = clean.std()
        alpha = 1 - confidence
        z = sp_stats.norm.ppf(alpha)
        cvar = float(-(mu - sigma * sp_stats.norm.pdf(z) / alpha))
    else:
        msg = f"Unknown CVaR method: {method!r}"
        raise ValueError(msg)

    return cvar
