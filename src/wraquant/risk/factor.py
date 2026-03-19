"""Factor risk models for return attribution and risk decomposition.

Factor models decompose portfolio risk into systematic (factor-driven) and
idiosyncratic (asset-specific) components. This is fundamental to
understanding *where* portfolio risk comes from and whether factor
exposures are intentional or accidental.

This module provides four approaches:

1. **Fundamental factor model** (``factor_risk_model``) -- regress returns
   on user-supplied factors (e.g., Fama-French, macro factors). Use when
   you know which factors matter.
2. **Statistical factor model** (``statistical_factor_model``) -- extract
   latent factors via PCA. Use when you do not have a prior on which
   factors drive returns.
3. **Fama-French regression** (``fama_french_regression``) -- specialised
   for the classic Fama-French framework with named factors (MKT, SMB,
   HML, etc.).
4. **Factor contribution** (``factor_contribution``) -- given portfolio
   weights and factor exposures, decompose portfolio risk into factor
   contributions.

References:
    - Fama & French (1993), "Common Risk Factors in the Returns on Stocks
      and Bonds"
    - Connor & Korajczyk (1986), "Performance Measurement with the
      Arbitrage Pricing Theory"
    - Menchero (2011), "The Barra Risk Model Handbook"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def factor_risk_model(
    returns: pd.Series | pd.DataFrame,
    factors: pd.DataFrame,
) -> dict[str, Any]:
    """Regress asset returns on factors and decompose total risk.

    Fits a multivariate OLS regression of returns on the provided factor
    returns, then decomposes total variance into the portion explained by
    factors (systematic risk) and the residual (specific/idiosyncratic
    risk).

    When to use:
        Use this function when you have a set of candidate factors (market,
        value, momentum, macro variables) and want to understand how much
        of the return variation they explain. The ``factor_risk`` /
        ``specific_risk`` split guides hedging decisions: hedge systematic
        risk with factor instruments; accept specific risk if you believe
        in the alpha.

    Mathematical formulation:
        r_t = alpha + B * f_t + eps_t

        Total variance = B' * Sigma_f * B + sigma_eps^2
        Factor risk share = B' * Sigma_f * B / Total variance
        Specific risk share = sigma_eps^2 / Total variance

    Parameters:
        returns: Asset return series (pd.Series for one asset,
            pd.DataFrame for multiple assets -- uses first column).
        factors: DataFrame of factor returns with columns as factor names
            and a compatible index.

    Returns:
        Dictionary containing:
        - **betas** (*dict[str, float]*) -- Factor loadings (regression
          coefficients). Positive beta = positive exposure.
        - **alpha** (*float*) -- Regression intercept (excess return not
          explained by factors).
        - **factor_risk** (*float*) -- Fraction of total variance explained
          by factors (0 to 1).
        - **specific_risk** (*float*) -- Fraction of total variance from
          idiosyncratic sources (1 - factor_risk).
        - **r_squared** (*float*) -- R-squared of the regression.
        - **residual_vol** (*float*) -- Annualized volatility of residuals.
        - **contributions** (*dict[str, float]*) -- Each factor's
          individual contribution to systematic variance.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> mkt = np.random.normal(0.0005, 0.01, 252)
        >>> smb = np.random.normal(0, 0.005, 252)
        >>> stock = 1.1 * mkt + 0.3 * smb + np.random.normal(0, 0.005, 252)
        >>> result = factor_risk_model(
        ...     pd.Series(stock),
        ...     pd.DataFrame({"MKT": mkt, "SMB": smb}),
        ... )
        >>> result["factor_risk"] > 0.5
        True

    See Also:
        statistical_factor_model: PCA-based (no prior on factors).
        fama_french_regression: Specialised Fama-French interface.

    References:
        - Menchero (2011), "The Barra Risk Model Handbook"
    """
    if isinstance(returns, pd.DataFrame):
        y = returns.iloc[:, 0].copy()
    else:
        y = returns.copy()

    # Align
    aligned = pd.concat([y.rename("y"), factors], axis=1).dropna()
    y_vals = aligned["y"].values
    factor_names = [c for c in aligned.columns if c != "y"]
    X = aligned[factor_names].values

    # OLS via shared regression module
    from wraquant.stats.regression import ols as _ols

    ols_result = _ols(y_vals, X, add_constant=True)
    coeffs = ols_result["coefficients"]
    alpha_val = float(coeffs[0])
    betas = {name: float(coeffs[i + 1]) for i, name in enumerate(factor_names)}
    residuals = ols_result["residuals"]
    r_squared = ols_result["r_squared"]

    # Risk decomposition
    total_var = float(np.var(y_vals, ddof=1))
    residual_var = float(np.var(residuals, ddof=1))

    factor_var = total_var - residual_var
    factor_risk = max(0.0, factor_var / total_var) if total_var > 0 else 0.0
    specific_risk = 1.0 - factor_risk

    # Factor covariance for individual contributions
    factor_cov = np.cov(X, rowvar=False, ddof=1)
    beta_vec = np.array([betas[n] for n in factor_names])
    if factor_cov.ndim == 0:
        factor_cov = np.array([[float(factor_cov)]])

    beta_vec @ factor_cov @ beta_vec
    contributions = {}
    for i, name in enumerate(factor_names):
        # Marginal contribution: beta_i * (Sigma_f @ beta)_i
        marginal = beta_vec[i] * (factor_cov @ beta_vec)[i]
        contributions[name] = float(marginal / total_var) if total_var > 0 else 0.0

    residual_vol = float(np.sqrt(residual_var) * np.sqrt(252))

    return {
        "betas": betas,
        "alpha": alpha_val,
        "factor_risk": factor_risk,
        "specific_risk": specific_risk,
        "r_squared": r_squared,
        "residual_vol": residual_vol,
        "contributions": contributions,
    }


def statistical_factor_model(
    returns: pd.DataFrame,
    n_factors: int = 3,
) -> dict[str, Any]:
    """PCA-based statistical factor model with risk decomposition.

    Extracts latent factors from the cross-section of asset returns using
    Principal Component Analysis (PCA). The first principal component
    typically captures market-wide movements; subsequent components
    capture sector, style, and other systematic effects.

    When to use:
        Use statistical factor models when you do not have a prior on
        which factors drive returns. PCA discovers the dominant sources
        of covariation. Useful for:
        - Constructing factor-mimicking portfolios.
        - Dimensionality reduction before portfolio optimisation.
        - Identifying hidden risk concentrations.

    Parameters:
        returns: DataFrame of asset returns (columns = assets, rows = dates).
            Should have at least ``n_factors + 1`` columns.
        n_factors: Number of principal components to extract. 3-5 is
            typical for equity portfolios.

    Returns:
        Dictionary containing:
        - **factors** (*pd.DataFrame*) -- Extracted factor return series
          (columns: PC1, PC2, ...).
        - **loadings** (*np.ndarray*) -- Factor loadings matrix (n_assets x
          n_factors).
        - **explained_variance** (*np.ndarray*) -- Variance explained by each
          factor.
        - **explained_variance_ratio** (*np.ndarray*) -- Fraction of total
          variance explained by each factor.
        - **cumulative_variance_ratio** (*np.ndarray*) -- Cumulative fraction
          of variance explained.
        - **factor_risk** (*float*) -- Total fraction of variance explained by
          all extracted factors.
        - **specific_risk** (*float*) -- Fraction of variance not explained.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> market = np.random.normal(0, 0.01, 252)
        >>> returns = pd.DataFrame({
        ...     f"asset_{i}": market * (0.5 + i * 0.2) + np.random.normal(0, 0.005, 252)
        ...     for i in range(5)
        ... })
        >>> result = statistical_factor_model(returns, n_factors=2)
        >>> result["factor_risk"] > 0.3
        True

    See Also:
        factor_risk_model: When you know which factors to use.

    References:
        - Connor & Korajczyk (1986), "Performance Measurement with the
          Arbitrage Pricing Theory"
    """
    clean = returns.dropna()
    X = clean.values
    n_obs, n_assets = X.shape

    # Demean
    means = X.mean(axis=0)
    X_centered = X - means

    # SVD-based PCA
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Eigenvalues (variance explained)
    eigenvalues = (S**2) / (n_obs - 1)
    total_var = eigenvalues.sum()

    n_factors = min(n_factors, min(n_obs, n_assets))

    # Factor returns: projections onto principal components
    factor_returns = X_centered @ Vt[:n_factors].T
    factor_names = [f"PC{i+1}" for i in range(n_factors)]
    factors_df = pd.DataFrame(factor_returns, index=clean.index, columns=factor_names)

    # Loadings
    loadings = Vt[:n_factors].T  # (n_assets x n_factors)

    explained_var = eigenvalues[:n_factors]
    explained_ratio = explained_var / total_var
    cumulative_ratio = np.cumsum(explained_ratio)

    factor_risk = float(cumulative_ratio[-1])
    specific_risk = 1.0 - factor_risk

    return {
        "factors": factors_df,
        "loadings": loadings,
        "explained_variance": explained_var,
        "explained_variance_ratio": explained_ratio,
        "cumulative_variance_ratio": cumulative_ratio,
        "factor_risk": factor_risk,
        "specific_risk": specific_risk,
    }


def fama_french_regression(
    returns: pd.Series,
    factors_df: pd.DataFrame,
) -> dict[str, Any]:
    r"""Fama-French factor regression with full diagnostics.

    Regresses asset returns on named Fama-French factors (e.g., Mkt-RF,
    SMB, HML, RMW, CMA, Mom). Reports alpha, betas, t-statistics, and
    R-squared. The alpha represents the return not explained by factor
    exposures -- a positive, statistically significant alpha indicates
    genuine skill.

    When to use:
        Use for performance attribution and alpha measurement. The classic
        3-factor model (Mkt, SMB, HML) is the minimum; the 5-factor model
        adds RMW (profitability) and CMA (investment). Add Mom (momentum)
        for the 6-factor model.

    Parameters:
        returns: Asset or portfolio return series (excess of risk-free rate
            if the factors are excess returns).
        factors_df: DataFrame of factor returns. Column names should be
            descriptive (e.g., "Mkt-RF", "SMB", "HML").

    Returns:
        Dictionary containing:
        - **alpha** (*float*) -- Jensen's alpha (intercept).
        - **betas** (*dict[str, float]*) -- Factor loadings.
        - **t_stats** (*dict[str, float]*) -- t-statistics for each
          coefficient (including alpha under key "alpha").
        - **p_values** (*dict[str, float]*) -- p-values for each
          coefficient.
        - **r_squared** (*float*) -- R-squared.
        - **adj_r_squared** (*float*) -- Adjusted R-squared.
        - **residual_vol** (*float*) -- Annualized residual volatility.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> mkt = np.random.normal(0.0005, 0.01, 252)
        >>> smb = np.random.normal(0, 0.005, 252)
        >>> hml = np.random.normal(0, 0.005, 252)
        >>> stock = 0.0001 + 1.1 * mkt + 0.3 * smb - 0.2 * hml + \\
        ...     np.random.normal(0, 0.003, 252)
        >>> factors = pd.DataFrame({"Mkt-RF": mkt, "SMB": smb, "HML": hml})
        >>> result = fama_french_regression(pd.Series(stock), factors)
        >>> abs(result["betas"]["Mkt-RF"] - 1.1) < 0.2
        True

    See Also:
        factor_risk_model: General factor regression with risk decomposition.

    References:
        - Fama & French (1993), "Common Risk Factors in the Returns on
          Stocks and Bonds"
        - Fama & French (2015), "A Five-Factor Asset Pricing Model"
    """
    aligned = pd.concat([returns.rename("y"), factors_df], axis=1).dropna()

    y = aligned["y"].values
    factor_names = [c for c in aligned.columns if c != "y"]
    X = aligned[factor_names].values

    # OLS via shared regression module
    from wraquant.stats.regression import ols as _ols

    ols_result = _ols(y, X, add_constant=True)
    coeffs = ols_result["coefficients"]
    t_stats_arr = ols_result["t_stats"]
    p_values_arr = ols_result["p_values"]
    residuals = ols_result["residuals"]
    r_squared = ols_result["r_squared"]
    adj_r_squared = ols_result["adj_r_squared"]

    alpha_val = float(coeffs[0])
    betas = {name: float(coeffs[i + 1]) for i, name in enumerate(factor_names)}
    t_stats_dict = {"alpha": float(t_stats_arr[0])}
    p_values_dict = {"alpha": float(p_values_arr[0])}

    for i, name in enumerate(factor_names):
        t_stats_dict[name] = float(t_stats_arr[i + 1])
        p_values_dict[name] = float(p_values_arr[i + 1])

    residual_vol = float(np.std(residuals, ddof=1) * np.sqrt(252))

    return {
        "alpha": alpha_val,
        "betas": betas,
        "t_stats": t_stats_dict,
        "p_values": p_values_dict,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "residual_vol": residual_vol,
    }


def factor_contribution(
    weights: np.ndarray,
    factor_betas: np.ndarray,
    factor_cov: np.ndarray,
) -> dict[str, Any]:
    """Decompose portfolio factor risk into per-factor contributions.

    Given portfolio weights, a matrix of factor loadings, and the factor
    covariance matrix, computes how much each factor contributes to
    total portfolio factor risk (variance).

    When to use:
        Use after estimating a factor model to understand which factors
        dominate portfolio risk. This guides factor hedging decisions:
        if 80% of portfolio risk comes from the market factor, you
        can hedge with index futures to dramatically reduce risk.

    Mathematical formulation:
        Portfolio factor variance = w' * B * Sigma_f * B' * w

        Factor i contribution = w' * B_i * (Sigma_f * B' * w)_i / total_var

    Parameters:
        weights: Portfolio weight vector (n_assets,).
        factor_betas: Factor loading matrix (n_assets x n_factors).
            Each row is an asset's factor exposures.
        factor_cov: Factor covariance matrix (n_factors x n_factors).

    Returns:
        Dictionary containing:
        - **total_factor_var** (*float*) -- Total portfolio factor variance.
        - **total_factor_vol** (*float*) -- Square root of factor variance.
        - **factor_contributions** (*np.ndarray*) -- Each factor's
          variance contribution (sums to total_factor_var).
        - **factor_pct_contributions** (*np.ndarray*) -- Percentage
          contributions (sum to 1.0).

    Example:
        >>> import numpy as np
        >>> weights = np.array([0.3, 0.3, 0.4])
        >>> betas = np.array([[1.0, 0.5], [1.2, -0.3], [0.8, 0.1]])
        >>> factor_cov = np.array([[0.0004, 0.00005], [0.00005, 0.0001]])
        >>> result = factor_contribution(weights, betas, factor_cov)
        >>> result["total_factor_var"] > 0
        True

    See Also:
        factor_risk_model: Estimate factor betas from return data.
        statistical_factor_model: Extract latent factors via PCA.
    """
    # Portfolio factor exposure: B' @ w -> (n_factors,)
    portfolio_beta = factor_betas.T @ weights  # (n_factors,)

    # Total factor variance: beta_p' @ Sigma_f @ beta_p
    total_factor_var = float(portfolio_beta @ factor_cov @ portfolio_beta)
    total_factor_vol = float(np.sqrt(max(0, total_factor_var)))

    # Per-factor marginal contribution
    # Euler decomposition: contribution_i = beta_p_i * (Sigma_f @ beta_p)_i
    marginal = factor_cov @ portfolio_beta
    contributions = portfolio_beta * marginal

    if total_factor_var > 0:
        pct_contributions = contributions / total_factor_var
    else:
        pct_contributions = np.zeros_like(contributions)

    return {
        "total_factor_var": total_factor_var,
        "total_factor_vol": total_factor_vol,
        "factor_contributions": contributions,
        "factor_pct_contributions": pct_contributions,
    }
