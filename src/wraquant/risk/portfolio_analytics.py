"""Advanced portfolio risk analytics.

Extends the basic portfolio risk tools (volatility, risk contribution,
diversification ratio) with VaR decomposition, risk budgeting, and
benchmark-relative analytics. These functions are essential for
institutional portfolio management, risk budgeting, and performance
attribution.

Key concepts:
    - **Component VaR**: how much each asset contributes to portfolio VaR.
    - **Marginal VaR**: sensitivity of portfolio VaR to a small change in
      weight (used for position sizing).
    - **Incremental VaR**: change in portfolio VaR from adding/removing
      an asset entirely.
    - **Risk budgeting**: find weights that produce equal (or target) risk
      contributions.
    - **Tracking error**: active risk relative to a benchmark.
    - **Active share**: how different the portfolio is from the benchmark.

References:
    - Litterman (1996), "Hot Spots and Hedges" (Euler decomposition)
    - Maillard, Roncalli & Teiletche (2010), "The Properties of Equally
      Weighted Risk Contribution Portfolios"
    - Cremers & Petajisto (2009), "How Active Is Your Fund Manager?"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sp_stats

from wraquant.risk.metrics import max_drawdown as _max_drawdown
from wraquant.risk.portfolio import portfolio_volatility as _portfolio_volatility


def component_var(
    weights: np.ndarray,
    returns: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.Series:
    """Component Value-at-Risk: per-asset contribution to portfolio VaR.

    Decomposes portfolio VaR into additive per-asset contributions using
    the Euler (marginal) decomposition. The sum of component VaRs equals
    the portfolio VaR. This tells you *where* the tail risk is
    concentrated.

    When to use:
        Use component VaR for:
        - Identifying which assets dominate portfolio tail risk.
        - Setting per-asset risk limits.
        - Reporting risk contributions to portfolio managers and risk
          committees.

    Mathematical formulation:
        Component VaR_i = w_i * (partial VaR / partial w_i)

        Under the delta-normal approximation:
        CVaR_i = w_i * (Sigma @ w)_i / sigma_p * VaR_p

    Parameters:
        weights: Portfolio weight vector (n_assets,).
        returns: Multi-asset return DataFrame (columns = assets).
        alpha: Significance level (0.05 = 95% VaR).

    Returns:
        pd.Series of per-asset VaR contributions, indexed by asset names.
        Sum equals the portfolio VaR.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame({
        ...     "A": np.random.normal(0.0005, 0.01, 252),
        ...     "B": np.random.normal(0.0003, 0.015, 252),
        ... })
        >>> weights = np.array([0.6, 0.4])
        >>> cvar = component_var(weights, returns, alpha=0.05)
        >>> cvar.sum() > 0  # total VaR is positive
        True

    See Also:
        marginal_var: Sensitivity of VaR to weight changes.
        incremental_var: VaR change from adding/removing an asset.
    """
    cov = returns.cov().values
    z = sp_stats.norm.ppf(alpha)

    port_vol = _portfolio_volatility(weights, cov)
    if port_vol == 0:
        return pd.Series(np.zeros(len(weights)), index=returns.columns)

    # Marginal contribution
    marginal = cov @ weights / port_vol
    component = weights * marginal * (-z)

    return pd.Series(component, index=returns.columns, name="component_var")


def marginal_var(
    weights: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Marginal VaR: sensitivity of portfolio VaR to weight changes.

    Marginal VaR measures how much portfolio VaR changes for a small
    (infinitesimal) change in the weight of each asset. It is the
    gradient of portfolio VaR with respect to weights.

    When to use:
        Use marginal VaR for:
        - Position sizing: assets with high marginal VaR should have
          smaller positions.
        - Optimisation: marginal VaR should be equal across assets at
          the optimal portfolio (risk parity condition).
        - Hedging: the hedge ratio is proportional to the marginal VaR.

    Mathematical formulation:
        Marginal VaR_i = dVaR/dw_i = z_alpha * (Sigma @ w)_i / sigma_p

    Parameters:
        weights: Portfolio weight vector (n_assets,).
        cov: Covariance matrix (n_assets x n_assets).
        alpha: Significance level (0.05 = 95% VaR).

    Returns:
        np.ndarray of marginal VaR values per asset.

    Example:
        >>> import numpy as np
        >>> cov = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
        >>> weights = np.array([0.6, 0.4])
        >>> mvar = marginal_var(weights, cov, alpha=0.05)
        >>> len(mvar) == 2
        True

    See Also:
        component_var: Additive VaR decomposition (weight * marginal VaR).
    """
    z = sp_stats.norm.ppf(alpha)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    if port_vol == 0:
        return np.zeros_like(weights)

    return -z * (cov @ weights) / port_vol


def incremental_var(
    weights: np.ndarray,
    returns: pd.DataFrame,
    alpha: float = 0.05,
) -> np.ndarray:
    """Incremental VaR: change in portfolio VaR from adding each asset.

    For each asset, computes the difference between the portfolio VaR
    with and without that asset (reallocating its weight proportionally
    to remaining assets). This measures the *discrete* impact of each
    position on tail risk.

    When to use:
        Use incremental VaR when deciding whether to add or remove a
        position. Unlike marginal VaR (which is an infinitesimal measure),
        incremental VaR captures the full nonlinear impact including
        diversification effects.

    Parameters:
        weights: Portfolio weight vector (n_assets,).
        returns: Multi-asset return DataFrame (columns = assets).
        alpha: Significance level (0.05 = 95% VaR).

    Returns:
        np.ndarray of incremental VaR values per asset. Positive means
        adding the asset increases portfolio VaR (adds risk); negative
        means it reduces VaR (diversification benefit).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame({
        ...     "A": np.random.normal(0.001, 0.01, 252),
        ...     "B": np.random.normal(0.0005, 0.008, 252),
        ... })
        >>> weights = np.array([0.6, 0.4])
        >>> ivar = incremental_var(weights, returns, alpha=0.05)
        >>> len(ivar) == 2
        True

    See Also:
        component_var: Euler-based additive decomposition.
        marginal_var: Infinitesimal sensitivity.
    """
    cov = returns.cov().values
    z = sp_stats.norm.ppf(alpha)
    n = len(weights)

    port_var = -z * float(np.sqrt(weights @ cov @ weights))
    inc = np.zeros(n)

    for i in range(n):
        w_ex = weights.copy()
        w_ex[i]
        w_ex[i] = 0.0

        remaining_sum = w_ex.sum()
        if remaining_sum > 0:
            w_ex = w_ex / remaining_sum
        else:
            # Only one asset; VaR without it is 0
            inc[i] = port_var
            continue

        port_var_ex = -z * float(np.sqrt(w_ex @ cov @ w_ex))
        inc[i] = port_var - port_var_ex

    return inc


def risk_budgeting(
    cov: np.ndarray,
    target_risk: np.ndarray | None = None,
) -> dict[str, Any]:
    """Find portfolio weights that achieve target risk contributions.

    Risk budgeting finds the weights such that each asset's risk
    contribution (Euler decomposition) matches a target budget. With
    equal targets (default), this is the risk parity portfolio.

    When to use:
        Use risk budgeting for:
        - Risk parity portfolio construction (equal risk contribution).
        - Custom risk allocation (e.g., 60% risk from equities, 40%
          from bonds, regardless of capital allocation).
        - Avoiding concentration: risk-budgeted portfolios avoid
          overweighting high-volatility assets.

    Mathematical formulation:
        Find w such that: w_i * (Sigma @ w)_i / sigma_p = b_i * sigma_p

        where b_i is the target risk budget (sum to 1).

    Parameters:
        cov: Covariance matrix (n x n).
        target_risk: Target risk contribution vector (sums to 1). If None,
            uses equal risk contributions (1/n for each asset).

    Returns:
        Dictionary containing:
        - **weights** (*np.ndarray*) -- Optimal portfolio weights.
        - **risk_contributions** (*np.ndarray*) -- Achieved risk
          contributions (should match target).
        - **portfolio_vol** (*float*) -- Portfolio volatility.
        - **converged** (*bool*) -- Whether the optimiser converged.

    Example:
        >>> import numpy as np
        >>> cov = np.array([[0.04, 0.006], [0.006, 0.01]])
        >>> result = risk_budgeting(cov)
        >>> np.allclose(result["risk_contributions"], 0.5, atol=0.05)
        True

    See Also:
        wraquant.risk.portfolio.risk_contribution: Compute risk
            contributions for given weights.
        wraquant.opt.portfolio: Full portfolio optimisation suite.

    References:
        - Maillard, Roncalli & Teiletche (2010), "The Properties of
          Equally Weighted Risk Contribution Portfolios"
    """
    n = cov.shape[0]
    if target_risk is None:
        target_risk = np.ones(n) / n
    target_risk = np.asarray(target_risk, dtype=float)

    def objective(w: np.ndarray) -> float:
        """Sum of squared deviations from target risk contributions."""
        port_vol = _portfolio_volatility(w, cov)
        if port_vol < 1e-15:
            return 1e10
        marginal = cov @ w / port_vol
        rc = w * marginal
        rc_pct = rc / rc.sum()
        return float(np.sum((rc_pct - target_risk) ** 2))

    # Initial weights: inverse volatility
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    w0 = inv_vol / inv_vol.sum()

    # Bounds: weights > 0, sum to 1
    bounds = [(1e-6, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = optimize.minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    weights = result.x
    port_vol = _portfolio_volatility(weights, cov)
    if port_vol > 0:
        marginal = cov @ weights / port_vol
        rc = weights * marginal
        rc_pct = rc / rc.sum()
    else:
        rc_pct = np.zeros(n)

    return {
        "weights": weights,
        "risk_contributions": rc_pct,
        "portfolio_vol": port_vol,
        "converged": result.success,
    }


def diversification_ratio(
    weights: np.ndarray,
    cov: np.ndarray,
) -> float:
    """Diversification ratio of a portfolio.

    The diversification ratio is the ratio of the weighted average of
    individual asset volatilities to the portfolio volatility. It
    measures the diversification benefit captured by the portfolio.

    When to use:
        Use as a portfolio quality metric. Higher is better:
        - DR = 1.0: no diversification benefit (perfectly correlated).
        - DR = 1.5: good diversification.
        - DR > 2.0: excellent diversification.
        The Maximum Diversification Portfolio (Choueifaty & Coignard)
        maximises this ratio.

    Mathematical formulation:
        DR = (w' * sigma) / sqrt(w' * Sigma * w)

        where sigma is the vector of individual asset volatilities
        and Sigma is the covariance matrix.

    Parameters:
        weights: Portfolio weight vector (n_assets,).
        cov: Covariance matrix (n_assets x n_assets).

    Returns:
        Diversification ratio as a float (>= 1.0).

    Example:
        >>> import numpy as np
        >>> cov = np.array([[0.04, 0.006], [0.006, 0.01]])
        >>> weights = np.array([0.5, 0.5])
        >>> dr = diversification_ratio(weights, cov)
        >>> dr >= 1.0
        True

    See Also:
        concentration_ratio: Herfindahl-based concentration measure.
        risk_budgeting: Find weights for target risk contributions.

    References:
        - Choueifaty & Coignard (2008), "Toward Maximum Diversification"
    """
    individual_vols = np.sqrt(np.diag(cov))
    weighted_avg_vol = float(weights @ individual_vols)
    port_vol = _portfolio_volatility(weights, cov)
    if port_vol == 0:
        return 1.0
    return weighted_avg_vol / port_vol


def concentration_ratio(
    weights: np.ndarray,
    cov: np.ndarray,
) -> float:
    """Herfindahl concentration ratio of risk contributions.

    Measures how concentrated portfolio risk is across assets using the
    Herfindahl-Hirschman Index (HHI) of risk contributions. An equally
    risk-contributed portfolio has HHI = 1/n (minimum concentration).

    When to use:
        Use concentration ratio to:
        - Detect hidden risk concentrations even when capital weights
          look diversified. A portfolio with equal weights can still have
          concentrated risk if one asset is much more volatile.
        - Monitor risk concentration over time.
        - Compare portfolios: lower concentration ratio = more
          diversified risk.

    Mathematical formulation:
        CR = sum(rc_i^2) where rc_i is asset i's fractional risk
        contribution (sum to 1.0).

        CR = 1/n for equal risk contribution; CR = 1.0 for single-asset.

    Parameters:
        weights: Portfolio weight vector (n_assets,).
        cov: Covariance matrix (n_assets x n_assets).

    Returns:
        Herfindahl concentration ratio between 1/n and 1.0.

    Example:
        >>> import numpy as np
        >>> cov = np.array([[0.04, 0.0], [0.0, 0.04]])
        >>> weights = np.array([0.5, 0.5])
        >>> cr = concentration_ratio(weights, cov)
        >>> abs(cr - 0.5) < 0.01  # equal vol + equal weight -> equal risk
        True

    See Also:
        diversification_ratio: Alternative diversification metric.
    """
    port_vol = _portfolio_volatility(weights, cov)
    if port_vol == 0:
        return 1.0

    marginal = cov @ weights / port_vol
    rc = weights * marginal
    total_rc = rc.sum()
    if total_rc == 0:
        return 1.0

    rc_pct = rc / total_rc
    return float(np.sum(rc_pct**2))


def tracking_error(
    returns: pd.Series,
    benchmark: pd.Series,
) -> dict[str, Any]:
    """Active risk metrics relative to a benchmark.

    Tracking error (TE) is the standard deviation of the active return
    (portfolio return minus benchmark return). It measures how much the
    portfolio's performance deviates from the benchmark.

    When to use:
        Use tracking error for:
        - Index tracking: target TE < 50bp for passive strategies.
        - Active management: typical TE of 2-8% for active equity funds.
        - Risk budgeting: allocate TE budget across portfolio managers.

    Parameters:
        returns: Portfolio return series.
        benchmark: Benchmark return series (same frequency and index).

    Returns:
        Dictionary containing:
        - **tracking_error** (*float*) -- Annualized tracking error.
        - **information_ratio** (*float*) -- Annualized active return /
          tracking error.
        - **active_return** (*float*) -- Annualized mean active return.
        - **max_active_drawdown** (*float*) -- Worst cumulative active
          return drawdown.
        - **active_return_std** (*float*) -- Daily active return standard
          deviation (non-annualized).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> portfolio = pd.Series(np.random.normal(0.0005, 0.01, 252))
        >>> benchmark = pd.Series(np.random.normal(0.0004, 0.009, 252))
        >>> result = tracking_error(portfolio, benchmark)
        >>> result["tracking_error"] > 0
        True

    See Also:
        active_share: Weight-based difference from benchmark.
        wraquant.risk.metrics.information_ratio: Simpler IR calculation.
    """
    active = returns - benchmark
    active_clean = active.dropna()

    te_daily = float(active_clean.std())
    te_annual = te_daily * np.sqrt(252)

    active_mean_daily = float(active_clean.mean())
    active_mean_annual = active_mean_daily * 252

    ir = active_mean_annual / te_annual if te_annual > 0 else 0.0

    # Max active drawdown via shared metrics
    cum_active = (1 + active_clean).cumprod()
    max_active_dd = _max_drawdown(cum_active)

    return {
        "tracking_error": te_annual,
        "information_ratio": ir,
        "active_return": active_mean_annual,
        "max_active_drawdown": max_active_dd,
        "active_return_std": te_daily,
    }


def active_share(
    weights: np.ndarray,
    benchmark_weights: np.ndarray,
) -> float:
    """Active share: weight-based deviation from benchmark.

    Active share measures how different a portfolio's holdings are from
    its benchmark. It is computed as half the sum of absolute weight
    differences.

    When to use:
        Use active share to classify portfolio management style:
        - Active share < 20%: closet indexer (charging active fees for
          passive exposure).
        - 20-60%: moderate active.
        - 60-80%: genuinely active.
        - > 80%: concentrated active or different investment universe.

    Mathematical formulation:
        Active Share = (1/2) * sum_i |w_i - w_bench_i|

    Parameters:
        weights: Portfolio weight vector.
        benchmark_weights: Benchmark weight vector (same length).

    Returns:
        Active share as a float between 0 and 1.

    Example:
        >>> import numpy as np
        >>> portfolio = np.array([0.4, 0.3, 0.2, 0.1])
        >>> benchmark = np.array([0.25, 0.25, 0.25, 0.25])
        >>> as_ = active_share(portfolio, benchmark)
        >>> 0 <= as_ <= 1
        True

    See Also:
        tracking_error: Return-based deviation from benchmark.

    References:
        - Cremers & Petajisto (2009), "How Active Is Your Fund Manager?"
    """
    return float(0.5 * np.sum(np.abs(weights - benchmark_weights)))
