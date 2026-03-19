"""Tail risk analytics for non-normal return distributions.

Standard risk measures (VaR, volatility) assume or approximate normality.
In practice, financial returns are fat-tailed (excess kurtosis) and
left-skewed. This module provides tail-aware risk measures that account
for higher moments and drawdown-based risk.

Functions:

1. **Cornish-Fisher VaR** -- adjusts the normal VaR quantile for skewness
   and kurtosis using the Cornish-Fisher expansion.
2. **ES decomposition** -- per-asset contribution to Expected Shortfall.
3. **Conditional Drawdown at Risk (CDaR)** -- the average of worst-alpha%
   drawdowns (analogous to CVaR but for drawdowns).
4. **Tail ratio analysis** -- 95th/5th percentile ratio with diagnostics.
5. **Drawdown at Risk (DaR)** -- worst alpha-quantile drawdown.

References:
    - Cornish & Fisher (1937), "Moments and Cumulants in the Specification
      of Distributions"
    - Chekhlov, Uryasev & Zabarankin (2005), "Drawdown Measure in
      Portfolio Optimization"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from wraquant.risk.var import value_at_risk as _value_at_risk


def cornish_fisher_var(
    returns: pd.Series,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Cornish-Fisher expansion VaR (skewness and kurtosis adjusted).

    The Cornish-Fisher expansion modifies the standard normal quantile
    to account for skewness (S) and excess kurtosis (K) of the return
    distribution. This produces a more accurate VaR than parametric
    (Gaussian) VaR for non-normal distributions.

    When to use:
        Use Cornish-Fisher VaR when:
        - Returns are detectably non-normal (skewness != 0 or kurtosis != 3).
        - You want a quick analytical adjustment without fitting a full
          distribution (e.g., Student-t or EVT).
        - The sample is too short for reliable historical VaR but long
          enough to estimate skewness/kurtosis (>100 observations).

    Mathematical formulation:
        z_cf = z + (z^2 - 1) * S/6 + (z^3 - 3z) * K/24 - (2z^3 - 5z) * S^2/36

        CF-VaR = -(mu + sigma * z_cf)

        where z = Phi^{-1}(alpha), S = skewness, K = excess kurtosis.

    How to interpret:
        Compare ``cf_var`` to ``normal_var``. If cf_var > normal_var, the
        distribution has fatter left tails than normal (typical for
        equities). The ``adjustment_factor`` (cf_var / normal_var) tells
        you how much the normal VaR underestimates tail risk.

    Parameters:
        returns: Simple return series.
        alpha: Significance level (0.05 = 95% VaR).

    Returns:
        Dictionary containing:
        - **cf_var** (*float*) -- Cornish-Fisher adjusted VaR (positive
          number = loss).
        - **normal_var** (*float*) -- Standard parametric (Gaussian) VaR.
        - **z_cf** (*float*) -- Adjusted quantile.
        - **z_normal** (*float*) -- Standard normal quantile.
        - **skewness** (*float*) -- Sample skewness.
        - **excess_kurtosis** (*float*) -- Sample excess kurtosis.
        - **adjustment_factor** (*float*) -- cf_var / normal_var.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0, 0.01, 1000))
        >>> result = cornish_fisher_var(returns, alpha=0.05)
        >>> result["cf_var"] > 0
        True

    See Also:
        wraquant.risk.var.value_at_risk: Historical and parametric VaR.
        tail_ratio_analysis: Tail shape diagnostics.

    References:
        - Cornish & Fisher (1937), "Moments and Cumulants in the
          Specification of Distributions"
        - Maillard (2012), "A User's Guide to the Cornish Fisher Expansion"
    """
    clean = returns.dropna().values
    mu = float(np.mean(clean))
    sigma = float(np.std(clean, ddof=1))
    s = float(sp_stats.skew(clean))
    k = float(sp_stats.kurtosis(clean))  # excess kurtosis

    z = sp_stats.norm.ppf(alpha)

    # Cornish-Fisher expansion
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )

    cf_var = -(mu + sigma * z_cf)
    normal_var = -(mu + sigma * z)

    adjustment = cf_var / normal_var if normal_var != 0 else 1.0

    return {
        "cf_var": float(cf_var),
        "normal_var": float(normal_var),
        "z_cf": float(z_cf),
        "z_normal": float(z),
        "skewness": s,
        "excess_kurtosis": k,
        "adjustment_factor": float(adjustment),
    }


def expected_shortfall_decomposition(
    weights: np.ndarray,
    returns: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.Series:
    """Decompose Expected Shortfall (CVaR) into per-asset contributions.

    Each asset's contribution to portfolio ES is computed as its average
    return on the days when the portfolio return is in the worst alpha
    tail. These contributions are additive (they sum to total portfolio
    ES).

    When to use:
        Use ES decomposition for:
        - Identifying which assets drive tail losses.
        - Setting per-asset ES limits.
        - Comparing tail-risk concentration to normal-market risk
          concentration.

    Mathematical formulation:
        ES_i = w_i * E[r_i | r_p <= VaR_alpha(r_p)]

        where r_p = w' @ r is the portfolio return.

    Parameters:
        weights: Portfolio weight vector (n_assets,).
        returns: Multi-asset return DataFrame (columns = assets).
        alpha: Significance level (0.05 = worst 5% of days).

    Returns:
        pd.Series of per-asset ES contributions. Sum equals portfolio ES
        (as a positive number).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame({
        ...     "A": np.random.normal(0.0005, 0.01, 500),
        ...     "B": np.random.normal(0.0003, 0.015, 500),
        ... })
        >>> weights = np.array([0.6, 0.4])
        >>> es = expected_shortfall_decomposition(weights, returns, alpha=0.05)
        >>> es.sum() > 0  # positive = loss
        True

    See Also:
        component_var: Euler decomposition of VaR.
        cornish_fisher_var: Skewness-adjusted VaR.
    """
    clean = returns.dropna()
    port_returns = clean.values @ weights

    cutoff = np.percentile(port_returns, alpha * 100)
    tail_mask = port_returns <= cutoff

    if tail_mask.sum() == 0:
        return pd.Series(np.zeros(len(weights)), index=returns.columns)

    tail_returns = clean.values[tail_mask]
    avg_tail = tail_returns.mean(axis=0)

    contributions = -weights * avg_tail

    return pd.Series(contributions, index=returns.columns, name="es_contribution")


def conditional_drawdown_at_risk(
    returns: pd.Series,
    alpha: float = 0.05,
) -> float:
    """Conditional Drawdown at Risk (CDaR).

    CDaR is the average of the worst alpha fraction of drawdowns in the
    return series. It is analogous to CVaR (Expected Shortfall) but
    operates on drawdowns rather than returns. CDaR is a coherent risk
    measure and is used in drawdown-constrained portfolio optimisation.

    When to use:
        Use CDaR when drawdown is a primary risk constraint (e.g., hedge
        funds with max drawdown mandates). CDaR penalises sustained
        drawdowns, not just point-in-time losses. A portfolio optimised
        to minimise CDaR will have better drawdown recovery properties
        than one optimised for VaR.

    Parameters:
        returns: Simple return series.
        alpha: Fraction of worst drawdowns to average (0.05 = worst 5%).

    Returns:
        CDaR as a positive float (e.g., 0.15 = average worst-5% drawdown
        is 15%).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0.0005, 0.01, 500))
        >>> cdar = conditional_drawdown_at_risk(returns, alpha=0.05)
        >>> cdar >= 0
        True

    See Also:
        drawdown_at_risk: Quantile-based drawdown measure (DaR).
        wraquant.risk.metrics.max_drawdown: Single worst drawdown.

    References:
        - Chekhlov, Uryasev & Zabarankin (2005), "Drawdown Measure in
          Portfolio Optimization"
    """
    clean = returns.dropna()
    cum = (1 + clean).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max  # negative values

    # CDaR: mean of worst alpha fraction
    dd_values = drawdowns.values
    n_tail = max(1, int(len(dd_values) * alpha))
    sorted_dd = np.sort(dd_values)[:n_tail]  # most negative first

    return float(-np.mean(sorted_dd))


def tail_ratio_analysis(returns: pd.Series) -> dict[str, Any]:
    """Tail ratio analysis with interpretation.

    The tail ratio is the ratio of the right tail (gains) to the absolute
    value of the left tail (losses) at a given percentile. A ratio > 1
    means the distribution has fatter right tails (gains are larger than
    losses at the extremes). A ratio < 1 means fatter left tails (losses
    are larger than gains).

    When to use:
        Use tail ratio analysis to:
        - Assess payoff asymmetry: trend-following should have tail ratio > 1
          (large gains, small frequent losses).
        - Detect negative skew: mean-reversion and short vol strategies
          typically have tail ratio < 1.
        - Compare strategies beyond Sharpe ratio.

    Parameters:
        returns: Simple return series.

    Returns:
        Dictionary containing:
        - **tail_ratio** (*float*) -- 95th percentile / abs(5th percentile).
        - **right_tail** (*float*) -- 95th percentile return.
        - **left_tail** (*float*) -- 5th percentile return.
        - **tail_ratio_99** (*float*) -- 99th/1st percentile ratio.
        - **skewness** (*float*) -- Sample skewness.
        - **excess_kurtosis** (*float*) -- Sample excess kurtosis.
        - **interpretation** (*str*) -- Human-readable assessment.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0, 0.01, 1000))
        >>> result = tail_ratio_analysis(returns)
        >>> result["tail_ratio"] > 0
        True

    See Also:
        cornish_fisher_var: Skewness-adjusted VaR.
    """
    clean = returns.dropna().values

    p5 = float(np.percentile(clean, 5))
    p95 = float(np.percentile(clean, 95))
    p1 = float(np.percentile(clean, 1))
    p99 = float(np.percentile(clean, 99))

    tail_ratio_95 = p95 / abs(p5) if abs(p5) > 1e-15 else float("inf")
    tail_ratio_99 = p99 / abs(p1) if abs(p1) > 1e-15 else float("inf")

    skew = float(sp_stats.skew(clean))
    kurt = float(sp_stats.kurtosis(clean))

    if tail_ratio_95 > 1.2:
        interp = "Right-skewed: gains are larger than losses at the tails. Favorable for trend-following."
    elif tail_ratio_95 < 0.8:
        interp = "Left-skewed: losses are larger than gains at the tails. Typical for short-vol or mean-reversion."
    else:
        interp = (
            "Approximately symmetric tails. Consistent with near-normal distribution."
        )

    return {
        "tail_ratio": float(tail_ratio_95),
        "right_tail": p95,
        "left_tail": p5,
        "tail_ratio_99": float(tail_ratio_99),
        "skewness": skew,
        "excess_kurtosis": kurt,
        "interpretation": interp,
    }


def drawdown_at_risk(
    returns: pd.Series,
    alpha: float = 0.05,
) -> float:
    """Drawdown at Risk (DaR): worst alpha-quantile drawdown.

    DaR is to drawdowns what VaR is to returns. It is the alpha-percentile
    of the drawdown distribution -- i.e., the drawdown that is exceeded
    only alpha% of the time.

    When to use:
        Use DaR when setting drawdown limits:
        - "With 95% confidence, the drawdown will not exceed DaR."
        - Useful for fund prospectuses and investor communications.
        - More intuitive than VaR for many stakeholders because
          drawdowns are easier to understand than daily P&L.

    Parameters:
        returns: Simple return series.
        alpha: Significance level (0.05 = 95th percentile drawdown).

    Returns:
        DaR as a positive float (e.g., 0.12 = 12% drawdown).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0.0005, 0.01, 500))
        >>> dar = drawdown_at_risk(returns, alpha=0.05)
        >>> dar >= 0
        True

    See Also:
        conditional_drawdown_at_risk: Average of worst drawdowns (CDaR).
        wraquant.risk.metrics.max_drawdown: Single worst drawdown.
    """
    clean = returns.dropna()
    cum = (1 + clean).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max  # negative values

    # DaR: alpha-percentile of drawdowns
    dar = float(np.percentile(drawdowns.values, alpha * 100))
    return -dar
