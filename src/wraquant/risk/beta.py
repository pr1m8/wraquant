"""Beta estimation models for systematic risk measurement.

Beta measures the sensitivity of an asset's returns to a benchmark (typically
the market). A beta of 1.0 means the asset moves in lockstep with the
benchmark; >1.0 means amplified moves; <1.0 means dampened moves; <0 means
the asset moves inversely.

This module provides six beta estimators, each suited to different situations:

1. **Rolling OLS beta** (``rolling_beta``) -- the workhorse; shows how beta
   evolves over time. Use for regime analysis and dynamic hedging.
2. **Blume adjustment** (``blume_adjusted_beta``) -- shrinks raw beta toward
   1.0 using the empirical regression-to-mean relationship.
3. **Vasicek adjustment** (``vasicek_adjusted_beta``) -- Bayesian shrinkage
   that incorporates estimation uncertainty.
4. **Dimson beta** (``dimson_beta``) -- sums lagged betas to correct for
   non-synchronous trading in illiquid assets.
5. **Conditional beta** (``conditional_beta``) -- separate up-market and
   down-market betas to capture asymmetric sensitivity.
6. **EWMA beta** (``ewma_beta``) -- exponentially weighted beta that adapts
   quickly to recent market conditions.

References:
    - Blume (1971), "On the Assessment of Risk"
    - Vasicek (1973), "A Note on Using Cross-Sectional Information in
      Bayesian Estimation of Security Betas"
    - Dimson (1979), "Risk Measurement When Shares are Subject to
      Infrequent Trading"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def rolling_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling OLS beta of asset returns against a benchmark.

    Computes the ordinary least squares regression slope of *returns* on
    *benchmark* over a rolling window. This is the standard approach for
    tracking how an asset's market sensitivity evolves over time.

    When to use:
        Use rolling beta to:
        - Monitor regime changes in market exposure (beta rising during
          sell-offs indicates contagion).
        - Calibrate dynamic hedging ratios (e.g., beta-hedge a long
          position with index futures).
        - Detect structural breaks in a strategy's factor exposure.

    Mathematical formulation:
        beta_t = Cov(r, b; t-w:t) / Var(b; t-w:t)

        where r is the asset return, b is the benchmark return, and
        w is the rolling window size.

    Parameters:
        returns: Asset return series (e.g., daily simple returns).
        benchmark: Benchmark return series (same frequency and aligned
            index). Typically a broad market index (S&P 500, MSCI World).
        window: Rolling window size in periods. 60 trading days (~3 months)
            is standard for equity beta. Use 120-252 for more stable
            estimates; use 20-40 for faster-reacting estimates.

    Returns:
        pd.Series of rolling beta values, indexed to match *returns*.
        The first ``window - 1`` values are NaN (insufficient data).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> market = pd.Series(np.random.normal(0.0005, 0.01, 252))
        >>> stock = 1.2 * market + np.random.normal(0, 0.005, 252)
        >>> beta = rolling_beta(stock, market, window=60)
        >>> abs(beta.iloc[-1] - 1.2) < 0.3
        True

    See Also:
        ewma_beta: Exponentially weighted alternative (no fixed window).
        conditional_beta: Separate up/down market betas.

    References:
        - Ang & Chen (2007), "Asymmetric Correlations of Equity Portfolios"
    """
    aligned = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()

    cov = aligned["r"].rolling(window).cov(aligned["b"])
    var = aligned["b"].rolling(window).var()

    beta = cov / var
    beta.name = "rolling_beta"
    return beta


def blume_adjusted_beta(raw_beta: float) -> float:
    """Blume-adjusted beta (mean-reversion adjustment).

    Blume (1971) documented that betas regress toward 1.0 over time.
    The adjustment applies the empirical relationship:

        adjusted_beta = 0.33 + 0.67 * raw_beta

    This is the adjustment used by Bloomberg and most commercial risk
    systems when reporting "adjusted beta."

    When to use:
        Use Blume adjustment for forward-looking beta estimates (e.g.,
        cost of equity in CAPM). The raw historical beta is a biased
        predictor of future beta; the Blume adjustment reduces this bias.

    Parameters:
        raw_beta: Historical OLS beta estimate.

    Returns:
        Adjusted beta as a float, shrunk toward 1.0.

    Example:
        >>> blume_adjusted_beta(1.5)
        1.335
        >>> blume_adjusted_beta(0.5)
        0.665
        >>> blume_adjusted_beta(1.0)
        1.0

    See Also:
        vasicek_adjusted_beta: Bayesian shrinkage with uncertainty.
        rolling_beta: Source of the raw beta input.

    References:
        - Blume (1971), "On the Assessment of Risk", *Journal of Finance*
        - Blume (1975), "Betas and Their Regression Tendencies"
    """
    return 0.33 + 0.67 * raw_beta


def vasicek_adjusted_beta(
    raw_beta: float,
    cross_sectional_mean: float = 1.0,
    raw_se: float = 0.2,
    prior_se: float = 0.3,
) -> float:
    """Vasicek Bayesian shrinkage beta adjustment.

    Combines the sample beta with a prior (typically the cross-sectional
    mean beta of 1.0) using a precision-weighted average. Assets with
    imprecise beta estimates (high standard error) are shrunk more
    toward the prior.

    When to use:
        Use Vasicek adjustment when you have an estimate of beta's
        standard error (e.g., from OLS regression). It is more
        principled than Blume's fixed-weight adjustment because the
        shrinkage intensity adapts to estimation uncertainty.

    Mathematical formulation:
        adjusted_beta = (prior_se^2 / (prior_se^2 + raw_se^2)) * raw_beta
                      + (raw_se^2 / (prior_se^2 + raw_se^2)) * cross_sectional_mean

    Parameters:
        raw_beta: Historical OLS beta estimate.
        cross_sectional_mean: Prior mean beta (cross-sectional average).
            Typically 1.0 for market beta.
        raw_se: Standard error of the raw beta estimate from OLS
            regression. Higher values cause more shrinkage.
        prior_se: Standard deviation of the cross-sectional beta
            distribution. Represents uncertainty in the prior.

    Returns:
        Vasicek-adjusted beta as a float.

    Example:
        >>> vasicek_adjusted_beta(1.5, cross_sectional_mean=1.0, raw_se=0.2, prior_se=0.3)
        1.3461538461538463
        >>> # High SE -> more shrinkage toward prior
        >>> vasicek_adjusted_beta(1.5, raw_se=0.5, prior_se=0.3)
        1.1323529411764706

    See Also:
        blume_adjusted_beta: Simpler fixed-weight adjustment.

    References:
        - Vasicek (1973), "A Note on Using Cross-Sectional Information
          in Bayesian Estimation of Security Betas"
    """
    prior_var = prior_se**2
    raw_var = raw_se**2
    total_var = prior_var + raw_var

    weight_raw = prior_var / total_var
    weight_prior = raw_var / total_var

    return weight_raw * raw_beta + weight_prior * cross_sectional_mean


def dimson_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    lags: int = 1,
) -> dict[str, Any]:
    r"""Dimson beta for illiquid or thinly traded assets.

    Standard OLS beta underestimates the true beta of assets that trade
    infrequently, because non-synchronous trading introduces measurement
    error. The Dimson (1979) correction runs a multiple regression of
    asset returns on contemporaneous and lagged benchmark returns, then
    sums all coefficients to recover the "true" beta.

    When to use:
        Use Dimson beta for:
        - Small-cap and micro-cap stocks with thin trading.
        - Private equity or real estate benchmarked against a public index.
        - Emerging market assets with liquidity constraints.
        A significant difference between ``total_beta`` and the
        contemporaneous beta suggests non-synchronous trading effects.

    Mathematical formulation:
        r_t = alpha + beta_0 * b_t + beta_1 * b_{t-1} + ... + beta_k * b_{t-k} + eps

        Dimson beta = sum(beta_0, beta_1, ..., beta_k)

    Parameters:
        returns: Asset return series.
        benchmark: Benchmark return series (same frequency and index).
        lags: Number of lagged benchmark terms to include. 1 is standard
            for daily data; use 2-3 for very illiquid assets.

    Returns:
        Dictionary containing:
        - **total_beta** (*float*) -- Sum of all lag coefficients (the
          Dimson-adjusted beta).
        - **lag_betas** (*list[float]*) -- Individual coefficients for
          each lag (index 0 = contemporaneous).
        - **alpha** (*float*) -- Regression intercept.
        - **r_squared** (*float*) -- R-squared of the multiple regression.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> benchmark = pd.Series(np.random.normal(0.0005, 0.01, 300))
        >>> # Illiquid asset: reacts with a lag
        >>> returns = 0.5 * benchmark + 0.4 * benchmark.shift(1).fillna(0) + \\
        ...     np.random.normal(0, 0.005, 300)
        >>> result = dimson_beta(returns, benchmark, lags=1)
        >>> result["total_beta"] > result["lag_betas"][0]
        True

    See Also:
        rolling_beta: Standard OLS beta (assumes synchronous trading).
        ewma_beta: Exponentially weighted beta.

    References:
        - Dimson (1979), "Risk Measurement When Shares are Subject to
          Infrequent Trading", *Journal of Financial Economics*
    """
    aligned = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()

    # Build design matrix with contemporaneous + lagged benchmark returns
    X_cols = [aligned["b"].values]
    for lag in range(1, lags + 1):
        X_cols.append(aligned["b"].shift(lag).values)

    # Stack and drop rows with NaN from lagging
    X = np.column_stack(X_cols)
    y = aligned["r"].values

    # Remove NaN rows
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    y = y[valid]

    # OLS via shared regression module
    from wraquant.stats.regression import ols as _ols

    ols_result = _ols(y, X, add_constant=True)
    alpha = float(ols_result["coefficients"][0])
    lag_betas = [float(c) for c in ols_result["coefficients"][1:]]
    r_squared = ols_result["r_squared"]

    return {
        "total_beta": sum(lag_betas),
        "lag_betas": lag_betas,
        "alpha": alpha,
        "r_squared": r_squared,
    }


def conditional_beta(
    returns: pd.Series,
    benchmark: pd.Series,
) -> dict[str, Any]:
    """Conditional (asymmetric) beta: separate up-market and down-market betas.

    Standard beta assumes symmetric sensitivity to the benchmark. In
    practice, many assets have higher beta in down markets than up markets
    (the "leverage effect" and flight-to-quality dynamics). Conditional
    beta splits the regression into up-market days (benchmark > 0) and
    down-market days (benchmark <= 0).

    When to use:
        Use conditional beta to:
        - Assess downside protection: an asset with low downside beta
          and high upside beta is a desirable portfolio component.
        - Detect asymmetric risk exposure: if downside_beta >> upside_beta,
          the asset amplifies losses more than gains.
        - Evaluate hedge fund or options-like payoff profiles.

    Mathematical formulation:
        Up-market: r_t = alpha_up + beta_up * b_t + eps, for b_t > 0
        Down-market: r_t = alpha_down + beta_down * b_t + eps, for b_t <= 0

    Parameters:
        returns: Asset return series.
        benchmark: Benchmark return series (same frequency and index).

    Returns:
        Dictionary containing:
        - **upside_beta** (*float*) -- Beta in up-market periods.
        - **downside_beta** (*float*) -- Beta in down-market periods.
        - **beta_asymmetry** (*float*) -- downside_beta - upside_beta.
          Positive means the asset is more sensitive to down moves.
        - **n_up** (*int*) -- Number of up-market observations.
        - **n_down** (*int*) -- Number of down-market observations.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> market = pd.Series(np.random.normal(0.0005, 0.01, 500))
        >>> stock = 1.0 * market + np.random.normal(0, 0.005, 500)
        >>> result = conditional_beta(stock, market)
        >>> isinstance(result["upside_beta"], float)
        True

    See Also:
        rolling_beta: Time-varying beta (not conditional on direction).
        ewma_beta: Exponentially weighted beta.

    References:
        - Pettengill, Sundaram & Mathur (1995), "The Conditional Relation
          Between Beta and Returns"
        - Ang & Chen (2002), "Asymmetric Correlations of Equity Portfolios"
    """
    aligned = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()

    up_mask = aligned["b"] > 0
    down_mask = ~up_mask

    up_data = aligned[up_mask]
    down_data = aligned[down_mask]

    def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
        """Simple OLS slope."""
        if len(x) < 3:
            return float("nan")
        cov = np.cov(x, y, ddof=1)
        var_x = cov[0, 0]
        if var_x == 0:
            return 0.0
        return float(cov[0, 1] / var_x)

    upside_beta = _ols_beta(up_data["r"].values, up_data["b"].values)
    downside_beta = _ols_beta(down_data["r"].values, down_data["b"].values)

    return {
        "upside_beta": upside_beta,
        "downside_beta": downside_beta,
        "beta_asymmetry": downside_beta - upside_beta,
        "n_up": int(up_mask.sum()),
        "n_down": int(down_mask.sum()),
    }


def ewma_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    halflife: int = 60,
) -> pd.Series:
    """Exponentially weighted moving average (EWMA) beta.

    Uses exponentially weighted covariance and variance to compute a
    time-varying beta that adapts to recent market conditions faster
    than a fixed rolling window. More recent observations receive
    exponentially higher weight.

    When to use:
        Use EWMA beta when you need a smooth, responsive beta estimate
        that adapts quickly to regime changes. Compared to rolling beta:
        - EWMA has no "cliff effect" (old observations do not drop out
          abruptly).
        - EWMA adapts faster to structural breaks (smaller halflife).
        - EWMA is smoother (no window-edge artifacts).

    Mathematical formulation:
        beta_t = EWCov(r, b; lambda) / EWVar(b; lambda)

        where lambda = 1 - exp(-ln(2) / halflife) is the decay factor.

    Parameters:
        returns: Asset return series.
        benchmark: Benchmark return series (same frequency and index).
        halflife: Decay halflife in periods. 60 days is standard.
            Shorter halflife (20-30) reacts faster but is noisier.
            Longer halflife (90-120) is smoother but lags.

    Returns:
        pd.Series of EWMA beta values.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> market = pd.Series(np.random.normal(0.0005, 0.01, 252))
        >>> stock = 1.3 * market + np.random.normal(0, 0.005, 252)
        >>> beta = ewma_beta(stock, market, halflife=60)
        >>> abs(beta.iloc[-1] - 1.3) < 0.4
        True

    See Also:
        rolling_beta: Fixed-window alternative.
        conditional_beta: Direction-dependent beta.

    References:
        - RiskMetrics Technical Document (1996), J.P. Morgan
    """
    aligned = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()

    ewm = aligned.ewm(halflife=halflife, min_periods=max(10, halflife // 2))
    cov_rb = ewm.cov()

    # Extract the cross-covariance and benchmark variance
    n = len(aligned)
    betas = np.full(n, np.nan)

    for i in range(n):
        idx = aligned.index[i]
        try:
            cov_matrix = cov_rb.loc[idx]
            cov_val = cov_matrix.loc["r", "b"]
            var_val = cov_matrix.loc["b", "b"]
            if var_val > 0:
                betas[i] = cov_val / var_val
        except (KeyError, ValueError):
            continue

    result = pd.Series(betas, index=aligned.index, name="ewma_beta")
    return result
