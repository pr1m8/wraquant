"""Scenario analysis and Monte Carlo simulation for portfolio risk."""

from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    n_sims: int = 10000,
    confidence: float = 0.95,
) -> float:
    """Estimate portfolio VaR via Monte Carlo simulation.

    Draws from a multivariate normal distribution fitted to historical
    returns.

    Parameters:
        returns: DataFrame of asset returns (columns = assets).
        weights: Portfolio weight vector.
        n_sims: Number of simulation draws.
        confidence: Confidence level.

    Returns:
        VaR as a positive float.
    """
    mu = returns.mean().values
    cov = returns.cov().values

    rng = np.random.default_rng()
    sims = rng.multivariate_normal(mu, cov, size=n_sims)
    portfolio_returns = sims @ weights

    var_quantile = np.percentile(portfolio_returns, (1 - confidence) * 100)
    return float(-var_quantile)


def stress_test(
    returns: pd.DataFrame,
    weights: np.ndarray,
    shocks: dict[str, float],
) -> float:
    """Compute the portfolio loss under a stress scenario.

    Each entry in *shocks* maps an asset name to a shocked return.
    Assets not in *shocks* use their historical mean.

    Parameters:
        returns: DataFrame of asset returns (columns = assets).
        weights: Portfolio weight vector.
        shocks: Mapping of asset name to shocked return value.

    Returns:
        Portfolio return under the stress scenario (typically negative).
    """
    scenario = returns.mean().copy()
    for asset, shock_val in shocks.items():
        if asset in scenario.index:
            scenario[asset] = shock_val

    return float(scenario.values @ weights)
