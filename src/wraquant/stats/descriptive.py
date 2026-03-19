"""Descriptive statistics for financial return and price series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def summary_stats(returns: pd.Series) -> dict:
    """Compute summary statistics for a return series.

    Parameters:
        returns: Simple return series.

    Returns:
        Dictionary with mean, std, skew, kurtosis, min, max, and count.
    """
    return {
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "skew": float(sp_stats.skew(returns.dropna(), bias=False)),
        "kurtosis": float(sp_stats.kurtosis(returns.dropna(), bias=False)),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "count": int(returns.count()),
    }


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized return from a simple return series.

    Parameters:
        returns: Simple return series.
        periods_per_year: Number of periods per year (252 for daily).

    Returns:
        Annualized return as a float.
    """
    total = (1 + returns).prod()
    n = len(returns)
    return float(total ** (periods_per_year / n) - 1)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized volatility from a simple return series.

    Parameters:
        returns: Simple return series.
        periods_per_year: Number of periods per year (252 for daily).

    Returns:
        Annualized volatility as a float.
    """
    return float(returns.std() * np.sqrt(periods_per_year))


def max_drawdown(prices: pd.Series) -> float:
    """Compute maximum drawdown from a price series.

    Parameters:
        prices: Price series (not returns).

    Returns:
        Maximum drawdown as a negative float (e.g., -0.25 for 25% drawdown).
    """
    from wraquant.risk.metrics import max_drawdown as _canonical

    return _canonical(prices)


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute the Calmar ratio (annualized return / max drawdown).

    Parameters:
        returns: Simple return series.
        periods_per_year: Number of periods per year.

    Returns:
        Calmar ratio as a float.
    """
    prices = (1 + returns).cumprod()
    mdd = max_drawdown(prices)
    if mdd == 0:
        return 0.0
    ann_ret = annualized_return(returns, periods_per_year)
    return float(ann_ret / abs(mdd))


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Compute the Omega ratio.

    The Omega ratio is the probability-weighted ratio of gains versus
    losses relative to a threshold.

    Parameters:
        returns: Simple return series.
        threshold: Return threshold (default 0).

    Returns:
        Omega ratio as a float.
    """
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess <= 0].sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)


# ---------------------------------------------------------------------------
# Rolling Sharpe ratio
# ---------------------------------------------------------------------------


def rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute the rolling Sharpe ratio over a moving window.

    The Sharpe ratio is the most widely used risk-adjusted performance
    measure.  The rolling variant shows how risk-adjusted performance
    evolves over time, revealing periods of strong and weak
    risk-adjusted returns.

    When to use:
        - To monitor strategy performance stability over time.
        - To detect regime changes in risk-adjusted returns (e.g., a
          strategy that worked pre-2020 but degraded post-2020).
        - To compare two strategies' time-varying risk-adjusted
          performance.

    Mathematical formulation:
        For each window of length *w*:

        .. math::

            \\text{Sharpe}_t = \\frac{\\bar{r}_t - r_f}{\\sigma_t} \\cdot \\sqrt{P}

        where ``\\bar{r}_t`` and ``\\sigma_t`` are the rolling mean and
        standard deviation of returns, ``r_f`` is the per-period risk-free
        rate, and ``P`` is the annualisation factor (e.g., 252 for daily).

    How to interpret:
        - Sharpe > 1.0: good risk-adjusted performance (annualised).
        - Sharpe > 2.0: very strong.
        - Sharpe < 0.0: losing money on a risk-adjusted basis.
        - Large swings in rolling Sharpe indicate unstable performance.

    Parameters:
        returns: Simple return series.
        window: Rolling window size in periods (default 60, roughly
            3 months of daily data).
        risk_free_rate: Per-period risk-free rate (default 0.0).
        periods_per_year: Annualisation factor (252 for daily data).

    Returns:
        Rolling Sharpe ratio as a pd.Series.  First ``window - 1``
        values are NaN.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> ret = pd.Series(np.random.normal(0.001, 0.02, 252))
        >>> rs = rolling_sharpe(ret, window=60)
        >>> rs.dropna().shape[0]
        193

    See Also:
        annualized_volatility: Annualised standard deviation.
        calmar_ratio: Drawdown-based risk-adjusted return.
    """
    excess = returns - risk_free_rate
    rolling_mean = excess.rolling(window).mean()
    rolling_std = returns.rolling(window).std(ddof=1)

    sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)
    sharpe.name = "rolling_sharpe"
    return sharpe


# ---------------------------------------------------------------------------
# Rolling max drawdown
# ---------------------------------------------------------------------------


def rolling_drawdown(
    returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute the rolling maximum drawdown over a moving window.

    For each time step, the maximum drawdown is computed using only the
    most recent *window* observations.  This provides a time-varying
    measure of downside risk.

    When to use:
        - To monitor the worst-case loss over a recent period.
        - To detect periods of elevated tail risk that may not show up
          in rolling volatility.
        - As an input to risk overlays that tighten exposure when
          recent drawdowns are deep.
        - To compare the downside risk profile of different strategies
          over time.

    Mathematical formulation:
        For each window ``[t - w + 1, t]``, compute the cumulative
        return series from the window's returns, find the peak, and
        measure the maximum drop from peak to trough:

        .. math::

            \\text{MDD}_t = \\min_{s \\in [t-w+1, t]} \\frac{P_s - \\max_{u \\le s} P_u}{\\max_{u \\le s} P_u}

    How to interpret:
        - Values are negative (or zero).  More negative = deeper drawdown.
        - A rolling drawdown of -0.10 means the portfolio lost 10% from
          its peak within the window.
        - Compare to static max drawdown to see if the worst period is
          concentrated or distributed.

    Parameters:
        returns: Simple return series.
        window: Rolling window size in periods (default 60).

    Returns:
        Rolling maximum drawdown as a pd.Series.  First ``window - 1``
        values are NaN.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> ret = pd.Series(np.random.normal(0, 0.02, 252))
        >>> rd = rolling_drawdown(ret, window=60)
        >>> (rd.dropna() <= 0).all()
        True

    See Also:
        max_drawdown: Full-sample maximum drawdown.
        calmar_ratio: Return / drawdown ratio.
    """
    result = pd.Series(np.nan, index=returns.index, name="rolling_drawdown")

    for i in range(window - 1, len(returns)):
        window_returns = returns.iloc[i - window + 1 : i + 1]
        cum = (1 + window_returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        result.iloc[i] = float(dd.min())

    return result


# ---------------------------------------------------------------------------
# Return attribution (Brinson model)
# ---------------------------------------------------------------------------


def return_attribution(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """Decompose portfolio excess return using the Brinson-Fachler model.

    The Brinson model is the industry standard for performance
    attribution, decomposing the active return (portfolio minus benchmark)
    into three components: asset allocation, security selection, and
    interaction effects.

    When to use:
        - To explain *why* a portfolio outperformed or underperformed its
          benchmark.
        - To separate the contribution of top-down asset allocation
          decisions from bottom-up security selection.
        - For reporting to investors or risk committees.

    Mathematical formulation:
        For each asset/sector *i*:

        - **Allocation effect**:
          ``(w^P_i - w^B_i) * (r^B_i - r^B_total)``
        - **Selection effect**:
          ``w^B_i * (r^P_i - r^B_i)``
        - **Interaction effect**:
          ``(w^P_i - w^B_i) * (r^P_i - r^B_i)``
        - **Total active return**:
          ``sum(allocation + selection + interaction) = r^P_total - r^B_total``

    How to interpret:
        - Positive allocation: the portfolio overweighted sectors that
          outperformed the benchmark.
        - Positive selection: within each sector, the portfolio held
          better-performing securities.
        - The interaction term captures the joint effect.
        - The three components sum to the total excess return.

    Parameters:
        portfolio_weights: Portfolio weights per asset/sector.
        benchmark_weights: Benchmark weights per asset/sector (same index).
        portfolio_returns: Portfolio returns per asset/sector.
        benchmark_returns: Benchmark returns per asset/sector (same index).

    Returns:
        Dictionary with:
        - ``allocation``: total allocation effect (float).
        - ``selection``: total selection effect (float).
        - ``interaction``: total interaction effect (float).
        - ``total_excess``: total excess return (float).
        - ``detail``: DataFrame with per-asset breakdown.

    Example:
        >>> import pandas as pd
        >>> pw = pd.Series({"Tech": 0.4, "Fin": 0.3, "Health": 0.3})
        >>> bw = pd.Series({"Tech": 0.3, "Fin": 0.4, "Health": 0.3})
        >>> pr = pd.Series({"Tech": 0.05, "Fin": 0.02, "Health": 0.03})
        >>> br = pd.Series({"Tech": 0.04, "Fin": 0.03, "Health": 0.03})
        >>> result = return_attribution(pw, bw, pr, br)
        >>> abs(result["total_excess"] - (pw @ pr - bw @ br)) < 1e-10
        True

    See Also:
        risk_contribution: Per-asset risk decomposition.
    """
    common = portfolio_weights.index.intersection(benchmark_weights.index)
    wp = portfolio_weights.loc[common]
    wb = benchmark_weights.loc[common]
    rp = portfolio_returns.loc[common]
    rb = benchmark_returns.loc[common]

    rb_total = float(wb @ rb)

    alloc = (wp - wb) * (rb - rb_total)
    select = wb * (rp - rb)
    interact = (wp - wb) * (rp - rb)

    detail = pd.DataFrame(
        {
            "allocation": alloc,
            "selection": select,
            "interaction": interact,
        }
    )

    return {
        "allocation": float(alloc.sum()),
        "selection": float(select.sum()),
        "interaction": float(interact.sum()),
        "total_excess": float(alloc.sum() + select.sum() + interact.sum()),
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Per-asset risk contribution
# ---------------------------------------------------------------------------


def risk_contribution(
    weights: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
) -> pd.Series:
    """Compute marginal risk contributions per asset.

    Risk contribution measures how much each asset contributes to the
    total portfolio risk (standard deviation).  This is the foundation
    of risk-parity and risk-budgeting portfolio construction.

    When to use:
        - To understand *where* portfolio risk comes from.
        - To build risk-parity portfolios where each asset contributes
          equally to total risk.
        - For risk monitoring: detect when a single position dominates
          portfolio risk.
        - To compare intended risk budgets with realised risk allocation.

    Mathematical formulation:
        The marginal contribution to risk (MCR) for asset *i* is:

        .. math::

            \\text{MCR}_i = w_i \\cdot \\frac{(\\Sigma w)_i}{\\sigma_p}

        where ``\\Sigma`` is the covariance matrix, ``w`` is the weight
        vector, and ``\\sigma_p = \\sqrt{w' \\Sigma w}`` is the portfolio
        standard deviation.

        The marginal contributions sum to the total portfolio risk:

        .. math::

            \\sum_i \\text{MCR}_i = \\sigma_p

    How to interpret:
        - Values are in the same units as portfolio standard deviation.
        - Each value represents the portion of total portfolio risk
          attributable to that asset.
        - Negative contributions are possible for assets that hedge
          overall portfolio risk.

    Parameters:
        weights: Portfolio weights (1-D array or Series).
        cov_matrix: Covariance matrix (2-D array or DataFrame).

    Returns:
        pd.Series of marginal risk contributions per asset.

    Example:
        >>> import pandas as pd, numpy as np
        >>> w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        >>> cov = pd.DataFrame(
        ...     np.diag([0.04, 0.09, 0.01]),
        ...     index=["A", "B", "C"], columns=["A", "B", "C"],
        ... )
        >>> rc = risk_contribution(w, cov)
        >>> abs(rc.sum() - np.sqrt(w.values @ cov.values @ w.values)) < 1e-10
        True

    See Also:
        return_attribution: Return decomposition (Brinson model).
        shrunk_covariance: Better covariance input for risk contributions.
    """
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(cov_matrix, dtype=float)

    port_var = float(w @ cov @ w)
    port_std = np.sqrt(port_var)

    if port_std <= 0:
        rc = np.zeros_like(w)
    else:
        # MCR_i = w_i * (Sigma @ w)_i / sigma_p
        marginal = cov @ w
        rc = w * marginal / port_std

    if isinstance(weights, pd.Series):
        return pd.Series(rc, index=weights.index, name="risk_contribution")
    return pd.Series(rc, name="risk_contribution")
