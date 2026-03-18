"""Portfolio optimization algorithms.

Implements common portfolio construction methods using scipy.optimize
(core dep). For more advanced solvers, see the convex module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize

from wraquant.opt.base import OptimizationResult


def _portfolio_stats(
    weights: npt.NDArray[np.floating],
    mean_returns: npt.NDArray[np.floating],
    cov_matrix: npt.NDArray[np.floating],
    periods_per_year: int = 252,
) -> tuple[float, float, float]:
    """Calculate portfolio return, volatility, and Sharpe ratio."""
    ret = float(np.dot(weights, mean_returns) * periods_per_year)
    vol = float(
        np.sqrt(np.dot(weights.T, np.dot(cov_matrix * periods_per_year, weights)))
    )
    sharpe = ret / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def mean_variance(
    returns: pd.DataFrame,
    target_return: float | None = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> OptimizationResult:
    """Mean-variance optimization (Markowitz).

    Parameters:
        returns: Asset return DataFrame (columns = assets).
        target_return: Target annualized return (None = max Sharpe).
        risk_free: Annual risk-free rate.
        periods_per_year: Trading periods per year.
        bounds: Weight bounds per asset (min, max).

    Returns:
        OptimizationResult with optimal weights.

    Example:
        >>> result = mean_variance(returns_df, target_return=0.10)  # doctest: +SKIP
    """
    n = returns.shape[1]
    mu = returns.mean().values
    cov = returns.cov().values
    assets = list(returns.columns)

    weight_bounds = [bounds] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_return is not None:
        ann_target = target_return / periods_per_year
        constraints.append({"type": "eq", "fun": lambda w: np.dot(w, mu) - ann_target})

    def neg_sharpe(w: npt.NDArray) -> float:
        ret = np.dot(w, mu) * periods_per_year
        vol = np.sqrt(np.dot(w.T, np.dot(cov * periods_per_year, w)))
        return -(ret - risk_free) / vol if vol > 0 else 0.0

    def portfolio_vol(w: npt.NDArray) -> float:
        return float(np.sqrt(np.dot(w.T, np.dot(cov * periods_per_year, w))))

    obj = neg_sharpe if target_return is None else portfolio_vol
    x0 = np.ones(n) / n

    result = optimize.minimize(
        obj, x0, method="SLSQP", bounds=weight_bounds, constraints=constraints
    )

    weights = result.x
    ret, vol, sharpe = _portfolio_stats(weights, mu, cov, periods_per_year)

    return OptimizationResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        asset_names=assets,
        metadata={"success": result.success, "message": result.message},
    )


def min_volatility(
    returns: pd.DataFrame,
    bounds: tuple[float, float] = (0.0, 1.0),
    periods_per_year: int = 252,
) -> OptimizationResult:
    """Minimum volatility portfolio.

    Parameters:
        returns: Asset return DataFrame.
        bounds: Weight bounds per asset.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with minimum variance weights.
    """
    n = returns.shape[1]
    mu = returns.mean().values
    cov = returns.cov().values
    assets = list(returns.columns)

    def portfolio_vol(w: npt.NDArray) -> float:
        return float(np.sqrt(np.dot(w.T, np.dot(cov * periods_per_year, w))))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.ones(n) / n

    result = optimize.minimize(
        portfolio_vol,
        x0,
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=constraints,
    )

    weights = result.x
    ret, vol, sharpe = _portfolio_stats(weights, mu, cov, periods_per_year)

    return OptimizationResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        asset_names=assets,
    )


def max_sharpe(
    returns: pd.DataFrame,
    risk_free: float = 0.0,
    bounds: tuple[float, float] = (0.0, 1.0),
    periods_per_year: int = 252,
) -> OptimizationResult:
    """Maximum Sharpe ratio portfolio.

    Parameters:
        returns: Asset return DataFrame.
        risk_free: Annual risk-free rate.
        bounds: Weight bounds per asset.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with maximum Sharpe weights.
    """
    return mean_variance(
        returns,
        target_return=None,
        risk_free=risk_free,
        bounds=bounds,
        periods_per_year=periods_per_year,
    )


def risk_parity(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> OptimizationResult:
    """Risk parity (equal risk contribution) portfolio.

    Parameters:
        returns: Asset return DataFrame.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with risk parity weights.
    """
    n = returns.shape[1]
    mu = returns.mean().values
    cov = returns.cov().values
    assets = list(returns.columns)

    target_risk = 1.0 / n

    def risk_budget_obj(w: npt.NDArray) -> float:
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        if port_vol == 0:
            return 0.0
        marginal_contrib = np.dot(cov, w) / port_vol
        risk_contrib = w * marginal_contrib
        risk_contrib_pct = risk_contrib / port_vol
        return float(np.sum((risk_contrib_pct - target_risk) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.ones(n) / n

    result = optimize.minimize(
        risk_budget_obj,
        x0,
        method="SLSQP",
        bounds=[(0.01, 1.0)] * n,
        constraints=constraints,
    )

    weights = result.x
    ret, vol, sharpe = _portfolio_stats(weights, mu, cov, periods_per_year)

    return OptimizationResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        asset_names=assets,
    )


def equal_weight(
    returns: pd.DataFrame, periods_per_year: int = 252
) -> OptimizationResult:
    """Equal weight portfolio (1/N).

    Parameters:
        returns: Asset return DataFrame.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with equal weights.
    """
    n = returns.shape[1]
    mu = returns.mean().values
    cov = returns.cov().values
    weights = np.ones(n) / n
    ret, vol, sharpe = _portfolio_stats(weights, mu, cov, periods_per_year)

    return OptimizationResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        asset_names=list(returns.columns),
    )


def inverse_volatility(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> OptimizationResult:
    """Inverse volatility weighted portfolio.

    Parameters:
        returns: Asset return DataFrame.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with inverse vol weights.
    """
    mu = returns.mean().values
    cov = returns.cov().values
    vols = returns.std().values
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    ret, vol, sharpe = _portfolio_stats(weights, mu, cov, periods_per_year)

    return OptimizationResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        asset_names=list(returns.columns),
    )


def hierarchical_risk_parity(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> OptimizationResult:
    """Hierarchical Risk Parity (HRP) by Lopez de Prado.

    Uses scipy hierarchical clustering and inverse-variance allocation.

    Parameters:
        returns: Asset return DataFrame.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with HRP weights.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    corr = returns.corr().values
    n = corr.shape[0]
    mu = returns.mean().values
    cov = returns.cov().values
    assets = list(returns.columns)

    # Distance matrix from correlation
    dist = np.sqrt((1 - corr) / 2)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist)
    link = linkage(condensed, method="single")
    sort_idx = leaves_list(link).tolist()

    # Recursive bisection
    weights = np.ones(n)

    def _recurse(items: list[int]) -> None:
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]

        cov_left = cov[np.ix_(left, left)]
        cov_right = cov[np.ix_(right, right)]

        inv_var_left = 1.0 / np.diag(cov_left).sum()
        inv_var_right = 1.0 / np.diag(cov_right).sum()
        alpha = inv_var_left / (inv_var_left + inv_var_right)

        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= 1 - alpha

        _recurse(left)
        _recurse(right)

    _recurse(sort_idx)
    weights = weights / weights.sum()
    ret, vol, sharpe = _portfolio_stats(weights, mu, cov, periods_per_year)

    return OptimizationResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        asset_names=assets,
    )


def black_litterman(
    returns: pd.DataFrame,
    views: dict[str, float],
    tau: float = 0.05,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> OptimizationResult:
    """Black-Litterman model.

    Parameters:
        returns: Asset return DataFrame.
        views: Dict mapping asset name to expected return view.
        tau: Uncertainty scaling parameter.
        risk_free: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with BL-adjusted weights.
    """
    n = returns.shape[1]
    assets = list(returns.columns)
    mu = returns.mean().values
    cov = returns.cov().values

    # Market cap weights proxy (equal weight as fallback)
    w_mkt = np.ones(n) / n

    # Implied equilibrium returns
    pi = tau * np.dot(cov, w_mkt) * periods_per_year

    # Build P (pick matrix) and Q (view returns) from views dict
    view_assets = [a for a in views if a in assets]
    k = len(view_assets)
    if k == 0:
        weights = w_mkt
    else:
        P = np.zeros((k, n))
        Q = np.zeros(k)
        for i, asset in enumerate(view_assets):
            j = assets.index(asset)
            P[i, j] = 1.0
            Q[i] = views[asset]

        # Omega = diag(P @ (tau * Sigma) @ P.T)
        omega = np.diag(np.diag(P @ (tau * cov) @ P.T))

        # BL combined return
        tau_cov = tau * cov
        inv_tau_cov = np.linalg.inv(tau_cov)
        inv_omega = np.linalg.inv(omega)
        bl_return = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P) @ (
            inv_tau_cov @ pi + P.T @ inv_omega @ Q
        )

        # BL weights via max Sharpe on BL returns
        def neg_sharpe(w: Any) -> float:
            ret = np.dot(w, bl_return)
            vol = np.sqrt(np.dot(w.T, np.dot(cov * periods_per_year, w)))
            return -(ret - risk_free / periods_per_year) / vol if vol > 0 else 0.0

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        result = optimize.minimize(
            neg_sharpe,
            w_mkt,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=constraints,
        )
        weights = result.x

    ret, vol, sharpe = _portfolio_stats(weights, mu, cov, periods_per_year)

    return OptimizationResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        asset_names=assets,
    )
