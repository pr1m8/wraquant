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
from wraquant.risk.portfolio import portfolio_volatility


def _portfolio_stats(
    weights: npt.NDArray[np.floating],
    mean_returns: npt.NDArray[np.floating],
    cov_matrix: npt.NDArray[np.floating],
    periods_per_year: int = 252,
) -> tuple[float, float, float]:
    """Calculate portfolio return, volatility, and Sharpe ratio."""
    ret = float(np.dot(weights, mean_returns) * periods_per_year)
    vol = portfolio_volatility(weights, cov_matrix * periods_per_year)
    sharpe = ret / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def mean_variance(
    returns: pd.DataFrame,
    target_return: float | None = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    bounds: tuple[float, float] = (0.0, 1.0),
    shrink: bool = False,
    shrinkage_method: str = "ledoit_wolf",
) -> OptimizationResult:
    """Mean-variance optimization (Markowitz).

    Use mean-variance optimization to find the portfolio that minimises
    risk for a given target return (efficient frontier), or maximises
    the Sharpe ratio when no target is specified.  This is the foundation
    of modern portfolio theory.

    Solves:
        min  w' Sigma w
        s.t. w' mu = target_return
             sum(w) = 1
             bounds[i][0] <= w[i] <= bounds[i][1]

    When ``target_return`` is None, maximises ``(w'mu - rf) / sqrt(w'Sigma w)``.

    Parameters:
        returns: Asset return DataFrame (columns = assets).  Must
            contain at least 2 assets and enough observations for a
            stable covariance estimate.
        target_return: Target annualised return (None = max Sharpe).
        risk_free: Annual risk-free rate for Sharpe calculation.
        periods_per_year: Trading periods per year (252 for daily).
        bounds: Weight bounds per asset (min, max).  Use ``(0, 1)`` for
            long-only; ``(-1, 1)`` to allow shorting.
        shrink: If ``True``, use a shrinkage estimator for the covariance
            matrix instead of the sample covariance.  Shrinkage produces
            a better-conditioned matrix when the number of assets is
            large relative to the number of observations.
        shrinkage_method: Shrinkage method when ``shrink=True`` --
            ``"ledoit_wolf"`` (default), ``"oas"``, or ``"basic"``.
            Forwarded to ``wraquant.stats.correlation.shrunk_covariance``.

    Returns:
        OptimizationResult with optimal weights, expected return,
        volatility, and Sharpe ratio.  Access ``result.weights`` for
        the allocation and ``result.sharpe_ratio`` for the risk-adjusted
        metric.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame(np.random.randn(252, 3) * 0.01,
        ...                        columns=['SPY', 'TLT', 'GLD'])
        >>> result = mean_variance(returns, target_return=0.05)
        >>> np.isclose(result.weights.sum(), 1.0)
        True

    Notes:
        Markowitz optimization is sensitive to estimation error in the
        mean return vector.  Consider ``black_litterman`` or
        ``hierarchical_risk_parity`` for more robust alternatives.

    See Also:
        max_sharpe: Convenience wrapper for max-Sharpe optimization.
        min_volatility: Minimum variance portfolio.
        risk_parity: Equal risk contribution portfolio.
    """
    n = returns.shape[1]
    mu = returns.mean().values
    if shrink:
        from wraquant.stats.correlation import shrunk_covariance

        cov = shrunk_covariance(returns, method=shrinkage_method).values
    else:
        cov = returns.cov().values
    assets = list(returns.columns)

    weight_bounds = [bounds] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_return is not None:
        ann_target = target_return / periods_per_year
        constraints.append({"type": "eq", "fun": lambda w: np.dot(w, mu) - ann_target})

    def neg_sharpe(w: npt.NDArray) -> float:
        ret = np.dot(w, mu) * periods_per_year
        vol = portfolio_volatility(w, cov * periods_per_year)
        return -(ret - risk_free) / vol if vol > 0 else 0.0

    def portfolio_vol(w: npt.NDArray) -> float:
        return portfolio_volatility(w, cov * periods_per_year)

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
    shrink: bool = False,
    shrinkage_method: str = "ledoit_wolf",
) -> OptimizationResult:
    """Minimum volatility portfolio.

    Use the minimum volatility portfolio when your primary objective is
    risk reduction rather than return maximisation.  This portfolio sits
    at the leftmost point of the efficient frontier and does not require
    a return estimate, making it more robust than mean-variance to
    estimation error in expected returns.

    Solves: min w' Sigma w, s.t. sum(w) = 1, bounds.

    Parameters:
        returns: Asset return DataFrame.
        bounds: Weight bounds per asset (default long-only ``(0, 1)``).
        periods_per_year: Trading periods per year.
        shrink: If ``True``, use a shrinkage covariance estimator.
        shrinkage_method: Shrinkage method (``"ledoit_wolf"``,
            ``"oas"``, or ``"basic"``).

    Returns:
        OptimizationResult with minimum variance weights.  The
        ``volatility`` field gives the lowest achievable portfolio
        standard deviation.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(0)
        >>> returns = pd.DataFrame(np.random.randn(252, 4) * 0.01,
        ...                        columns=['A', 'B', 'C', 'D'])
        >>> result = min_volatility(returns)
        >>> result.volatility > 0
        True

    See Also:
        mean_variance: Full mean-variance with target return.
        risk_parity: Equal risk contribution (also estimation-robust).
    """
    n = returns.shape[1]
    mu = returns.mean().values
    if shrink:
        from wraquant.stats.correlation import shrunk_covariance

        cov = shrunk_covariance(returns, method=shrinkage_method).values
    else:
        cov = returns.cov().values
    assets = list(returns.columns)

    def portfolio_vol(w: npt.NDArray) -> float:
        return portfolio_volatility(w, cov * periods_per_year)

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
    shrink: bool = False,
    shrinkage_method: str = "ledoit_wolf",
) -> OptimizationResult:
    """Maximum Sharpe ratio portfolio.

    Use max-Sharpe when you want the portfolio with the highest
    risk-adjusted return.  This is the tangency portfolio on the
    efficient frontier -- the point where a line from the risk-free
    rate is tangent to the frontier.

    Maximises: (w'mu - rf) / sqrt(w'Sigma w), s.t. sum(w) = 1, bounds.

    Parameters:
        returns: Asset return DataFrame.
        risk_free: Annual risk-free rate.
        bounds: Weight bounds per asset.
        periods_per_year: Trading periods per year.
        shrink: If ``True``, use a shrinkage covariance estimator.
        shrinkage_method: Shrinkage method (``"ledoit_wolf"``,
            ``"oas"``, or ``"basic"``).

    Returns:
        OptimizationResult with maximum Sharpe weights.  The
        ``sharpe_ratio`` field gives the optimal risk-adjusted return.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame(np.random.randn(252, 3) * 0.01,
        ...                        columns=['SPY', 'TLT', 'GLD'])
        >>> result = max_sharpe(returns, risk_free=0.04)
        >>> np.isclose(result.weights.sum(), 1.0)
        True

    See Also:
        mean_variance: Mean-variance with a target return constraint.
        min_volatility: Minimum risk portfolio.
    """
    return mean_variance(
        returns,
        target_return=None,
        risk_free=risk_free,
        bounds=bounds,
        periods_per_year=periods_per_year,
        shrink=shrink,
        shrinkage_method=shrinkage_method,
    )


def risk_parity(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> OptimizationResult:
    """Risk parity (equal risk contribution) portfolio.

    Use risk parity when you want each asset to contribute equally to
    total portfolio risk.  Unlike mean-variance, risk parity does not
    require expected return estimates, making it robust to estimation
    error.  It is the basis of many institutional "all-weather" strategies.

    Minimises: sum_i (RC_i / sigma_p - 1/N)^2

    where RC_i = w_i * (Sigma w)_i / sigma_p is asset i's risk
    contribution and sigma_p is portfolio volatility.

    Parameters:
        returns: Asset return DataFrame.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with risk parity weights.  Lower-volatility
        assets receive higher weights; higher-volatility assets receive
        lower weights.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame(np.random.randn(252, 3) * np.array([0.01, 0.02, 0.005]),
        ...                        columns=['Bonds', 'Equity', 'Gold'])
        >>> result = risk_parity(returns)
        >>> result.weights[0] > result.weights[1]  # bonds get more weight (lower vol)
        True

    References:
        - Maillard, Roncalli & Teiletche (2010), "The Properties of
          Equally Weighted Risk Contribution Portfolios"

    See Also:
        hierarchical_risk_parity: HRP (no inversion of covariance matrix).
        min_volatility: Minimum variance (not risk-balanced).
    """
    n = returns.shape[1]
    mu = returns.mean().values
    cov = returns.cov().values
    assets = list(returns.columns)

    target_risk = 1.0 / n

    # NOTE: Could use wraquant.risk.portfolio.risk_contribution here, but
    # the optimizer hot-loop benefits from inlined math to avoid function
    # call overhead on each iteration.
    def risk_budget_obj(w: npt.NDArray) -> float:
        port_vol = portfolio_volatility(w, cov)
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

    Use the equal-weight portfolio as a robust baseline.  Despite its
    simplicity, 1/N consistently outperforms many optimised portfolios
    out-of-sample because it avoids estimation error entirely.

    Parameters:
        returns: Asset return DataFrame.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with equal weights (each asset receives
        weight 1/N).

    Example:
        >>> import pandas as pd, numpy as np
        >>> returns = pd.DataFrame(np.random.randn(100, 4) * 0.01,
        ...                        columns=['A', 'B', 'C', 'D'])
        >>> result = equal_weight(returns)
        >>> np.allclose(result.weights, 0.25)
        True

    References:
        - DeMiguel, Garlappi & Uppal (2009), "Optimal Versus Naive
          Diversification"

    See Also:
        inverse_volatility: Simple vol-weighted alternative.
        risk_parity: Optimisation-based risk balancing.
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

    Use inverse-volatility weighting as a simple, estimation-light
    alternative to mean-variance.  Assets with lower volatility receive
    higher weights, producing a portfolio that tilts toward stability
    without requiring a full covariance estimate.

    Weight_i = (1 / sigma_i) / sum_j(1 / sigma_j)

    Parameters:
        returns: Asset return DataFrame.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with inverse vol weights.  Lower-volatility
        assets receive higher allocations.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(0)
        >>> returns = pd.DataFrame(
        ...     np.random.randn(252, 3) * np.array([0.005, 0.02, 0.01]),
        ...     columns=['Bonds', 'Equity', 'Gold'])
        >>> result = inverse_volatility(returns)
        >>> result.weights[0] > result.weights[1]  # Bonds > Equity
        True

    See Also:
        equal_weight: Uniform weighting (ignores vol entirely).
        risk_parity: Equalises risk contribution (uses covariance).
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

    Use HRP when you want a stable, estimation-robust portfolio that
    does not require covariance matrix inversion.  HRP applies
    hierarchical clustering to the correlation matrix, then allocates
    via recursive bisection using inverse variance.  This avoids the
    instability of mean-variance optimisation and produces portfolios
    that are naturally diversified across asset clusters.

    Algorithm:
        1. Compute correlation-based distance and hierarchical linkage.
        2. Quasi-diagonalise the covariance matrix.
        3. Recursively bisect the sorted assets, allocating by inverse
           variance of each cluster.

    Parameters:
        returns: Asset return DataFrame.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with HRP weights.  Weights are always
        positive (long-only) and sum to 1.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame(np.random.randn(252, 5) * 0.01,
        ...                        columns=['A', 'B', 'C', 'D', 'E'])
        >>> result = hierarchical_risk_parity(returns)
        >>> np.isclose(result.weights.sum(), 1.0)
        True

    References:
        - Lopez de Prado (2016), "Building Diversified Portfolios that
          Outperform Out-of-Sample"

    See Also:
        risk_parity: Equal risk contribution (requires covariance inversion).
        mean_variance: Classical Markowitz (more sensitive to estimation error).
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

    Use Black-Litterman when you have subjective views on expected
    returns for some assets and want to combine them with market
    equilibrium returns in a Bayesian framework.  BL produces more
    stable and intuitive portfolios than raw mean-variance because it
    starts from an equilibrium prior (implied by market capitalisation)
    and blends in your views proportionally to your confidence.

    The posterior expected return is:

        E[r] = [(tau Sigma)^{-1} + P' Omega^{-1} P]^{-1}
               * [(tau Sigma)^{-1} pi + P' Omega^{-1} Q]

    where pi = implied equilibrium returns, P = pick matrix, Q = view
    returns, Omega = view uncertainty.

    Parameters:
        returns: Asset return DataFrame.
        views: Dict mapping asset name to expected return view (e.g.,
            ``{'AAPL': 0.12}`` means you expect AAPL to return 12%
            annualised).
        tau: Uncertainty scaling parameter (typical range 0.01-0.1).
            Higher tau gives more weight to your views.
        risk_free: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        OptimizationResult with BL-adjusted weights.  The weights
        reflect a blend of market equilibrium and your views.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame(np.random.randn(252, 3) * 0.01,
        ...                        columns=['AAPL', 'MSFT', 'GOOG'])
        >>> views = {'AAPL': 0.15}  # bullish on AAPL
        >>> result = black_litterman(returns, views, tau=0.05)
        >>> result.weights[0] > 1 / 3  # AAPL gets more weight
        True

    References:
        - Black & Litterman (1992), "Global Portfolio Optimization"
        - He & Litterman (1999), "The Intuition Behind Black-Litterman"

    See Also:
        mean_variance: Pure mean-variance (no views prior).
        risk_parity: View-free risk-based allocation.
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
            vol = portfolio_volatility(w, cov * periods_per_year)
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
