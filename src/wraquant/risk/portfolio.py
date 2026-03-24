"""Portfolio-level risk analytics."""

from __future__ import annotations

import numpy as np


def portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """Compute portfolio volatility from weights and covariance matrix.

    Parameters:
        weights: Asset weight vector (n,).
        cov_matrix: Covariance matrix (n, n).

    Returns:
        Portfolio volatility (standard deviation).
    """
    from wraquant.core._coerce import coerce_array

    weights = coerce_array(weights, name="weights")
    cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
    return float(np.sqrt(weights @ cov_matrix @ weights))


def risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """Compute each asset's risk contribution to portfolio volatility.

    Parameters:
        weights: Asset weight vector (n,).
        cov_matrix: Covariance matrix (n, n).

    Returns:
        Array of fractional risk contributions that sum to 1.
    """
    from wraquant.core._coerce import coerce_array

    weights = coerce_array(weights, name="weights")
    cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
    port_vol = portfolio_volatility(weights, cov_matrix)
    if port_vol == 0:
        return np.zeros_like(weights)
    marginal = cov_matrix @ weights / port_vol
    rc = weights * marginal
    return rc / rc.sum()


def diversification_ratio(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """Compute the diversification ratio.

    The diversification ratio is the ratio of the weighted average of
    individual volatilities to the portfolio volatility. A value of 1
    means no diversification benefit.

    Parameters:
        weights: Asset weight vector (n,).
        cov_matrix: Covariance matrix (n, n).

    Returns:
        Diversification ratio (>= 1).
    """
    from wraquant.core._coerce import coerce_array

    weights = coerce_array(weights, name="weights")
    cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
    individual_vols = np.sqrt(np.diag(cov_matrix))
    weighted_avg_vol = float(weights @ individual_vols)
    port_vol = portfolio_volatility(weights, cov_matrix)
    if port_vol == 0:
        return 1.0
    return weighted_avg_vol / port_vol
