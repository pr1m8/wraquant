"""Statistical factor analysis for financial data.

Provides PCA-based factor extraction, factor loadings estimation, varimax
rotation, factor-mimicking portfolios, and risk decomposition.  All
implementations use pure numpy/scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

__all__ = [
    "pca_factors",
    "factor_loadings",
    "scree_plot_data",
    "varimax_rotation",
    "factor_mimicking_portfolios",
    "risk_factor_decomposition",
    "factor_correlation",
    "common_factors",
    "fama_french_factors",
    "factor_exposure",
    "factor_risk_decomposition",
]


# ---------------------------------------------------------------------------
# PCA factor extraction
# ---------------------------------------------------------------------------


def pca_factors(
    returns: pd.DataFrame,
    n_components: int = 3,
    method: str = "svd",
) -> dict:
    """Extract statistical factors from a returns matrix using PCA.

    Parameters:
        returns: DataFrame of asset returns with shape ``(T, N)`` where
            *T* is the number of observations and *N* is the number of
            assets.
        n_components: Number of principal components to retain.
        method: Decomposition method.  ``"svd"`` (default) uses
            ``numpy.linalg.svd`` on the demeaned returns; ``"eig"`` uses
            eigendecomposition of the covariance matrix.

    Returns:
        Dictionary with:
        - ``factors``: DataFrame of extracted factors ``(T, n_components)``.
        - ``loadings``: DataFrame of factor loadings ``(N, n_components)``.
        - ``explained_variance``: array of variance explained by each
          component.
        - ``explained_variance_ratio``: array of fraction of total
          variance explained.
    """
    X = np.asarray(returns, dtype=float)
    T, N = X.shape
    mean = X.mean(axis=0)
    X_centered = X - mean

    if method == "eig":
        cov = np.cov(X_centered, rowvar=False, ddof=1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        loadings_arr = eigenvectors[:, :n_components]
        factors_arr = X_centered @ loadings_arr
        explained_variance = eigenvalues[:n_components]
    else:
        # SVD method (default)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        loadings_arr = Vt[:n_components].T  # (N, n_components)
        factors_arr = X_centered @ loadings_arr  # (T, n_components)
        eigenvalues = (S ** 2) / (T - 1)
        explained_variance = eigenvalues[:n_components]

    total_variance = np.sum(np.var(X_centered, axis=0, ddof=1))
    explained_variance_ratio = explained_variance / total_variance

    component_names = [f"PC{i+1}" for i in range(n_components)]
    index = returns.index if isinstance(returns, pd.DataFrame) else range(T)
    columns = returns.columns if isinstance(returns, pd.DataFrame) else range(N)

    factors_df = pd.DataFrame(factors_arr, index=index, columns=component_names)
    loadings_df = pd.DataFrame(loadings_arr, index=columns, columns=component_names)

    return {
        "factors": factors_df,
        "loadings": loadings_df,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
    }


# ---------------------------------------------------------------------------
# Factor loadings via regression
# ---------------------------------------------------------------------------


def factor_loadings(
    returns: pd.DataFrame | np.ndarray,
    factors: pd.DataFrame | np.ndarray,
) -> dict:
    """Compute factor loadings by regressing each asset's returns on factors.

    For each asset column in *returns*, an OLS regression is run against
    *factors* (with intercept) to obtain betas (loadings), alphas, and
    R-squared values.

    Parameters:
        returns: Asset returns matrix ``(T, N)``.
        factors: Factor returns matrix ``(T, K)``.

    Returns:
        Dictionary with:
        - ``loadings``: ndarray ``(N, K)`` of factor betas.
        - ``alphas``: ndarray ``(N,)`` of intercepts.
        - ``r_squared``: ndarray ``(N,)`` of regression R-squared values.
        - ``residuals``: ndarray ``(T, N)`` of regression residuals.
    """
    from wraquant.stats.regression import ols as _ols

    Y = np.asarray(returns, dtype=float)
    F = np.asarray(factors, dtype=float)
    T, N = Y.shape
    K = F.shape[1] if F.ndim == 2 else 1
    if F.ndim == 1:
        F = F.reshape(-1, 1)

    # Regress each asset on the factors via the shared OLS module
    loadings_arr = np.zeros((N, K))
    alphas = np.zeros(N)
    r_squared = np.zeros(N)
    residuals = np.zeros((T, N))

    for i in range(N):
        result = _ols(Y[:, i], F, add_constant=True)
        alphas[i] = result["coefficients"][0]
        loadings_arr[i, :] = result["coefficients"][1:]
        r_squared[i] = result["r_squared"]
        residuals[:, i] = result["residuals"]

    return {
        "loadings": loadings_arr,
        "alphas": alphas,
        "r_squared": r_squared,
        "residuals": residuals,
    }


# ---------------------------------------------------------------------------
# Scree plot data
# ---------------------------------------------------------------------------


def scree_plot_data(returns: pd.DataFrame | np.ndarray) -> dict:
    """Return eigenvalues and explained variance ratios for a scree plot.

    Parameters:
        returns: Asset returns matrix ``(T, N)``.

    Returns:
        Dictionary with:
        - ``eigenvalues``: array of eigenvalues in descending order.
        - ``explained_variance_ratio``: array of fraction of total
          variance explained by each component.
        - ``cumulative_variance_ratio``: cumulative sum of explained
          variance ratios.
    """
    X = np.asarray(returns, dtype=float)
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # descending
    total = eigenvalues.sum()
    ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    cumulative = np.cumsum(ratio)

    return {
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": ratio,
        "cumulative_variance_ratio": cumulative,
    }


# ---------------------------------------------------------------------------
# Varimax rotation
# ---------------------------------------------------------------------------


def varimax_rotation(
    loadings: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> dict:
    """Apply varimax rotation to a factor loadings matrix.

    Varimax maximises the sum of variances of squared loadings within
    each factor, producing a simpler (more interpretable) structure.

    Parameters:
        loadings: Factor loadings matrix ``(N, K)`` where *N* is the
            number of variables and *K* is the number of factors.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on the rotation criterion change.

    Returns:
        Dictionary with:
        - ``rotated_loadings``: ndarray ``(N, K)`` of rotated loadings.
        - ``rotation_matrix``: ndarray ``(K, K)`` orthogonal rotation
          matrix.
        - ``n_iter``: number of iterations performed.
    """
    A = np.array(loadings, dtype=float, copy=True)
    p, k = A.shape

    if k < 2:
        return {
            "rotated_loadings": A,
            "rotation_matrix": np.eye(k),
            "n_iter": 0,
        }

    rotation = np.eye(k)
    d = 0.0

    for it in range(max_iter):
        d_old = d
        B = A @ rotation
        # Varimax criterion update via SVD of A' * (B^3 - B @ diag(sum(B^2)))
        B2 = B ** 2
        D = np.diag((B2).sum(axis=0) / p)
        C = A.T @ (B ** 3 - B @ D)
        U, _S, Vt = np.linalg.svd(C)
        rotation = U @ Vt
        d = np.sum(_S)
        if abs(d - d_old) < tol:
            break

    rotated = A @ rotation

    return {
        "rotated_loadings": rotated,
        "rotation_matrix": rotation,
        "n_iter": it + 1,
    }


# ---------------------------------------------------------------------------
# Factor-mimicking portfolios
# ---------------------------------------------------------------------------


def factor_mimicking_portfolios(
    returns: pd.DataFrame,
    characteristics: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Build factor-mimicking portfolios via long-short quantile sorts.

    For each characteristic column, assets are sorted into quantiles at
    each time step. The factor-mimicking return is the difference between
    the mean return of the top quantile and the mean return of the bottom
    quantile (long top, short bottom).

    Parameters:
        returns: DataFrame of asset returns ``(T, N)``.
        characteristics: DataFrame of asset characteristics ``(T, N)``
            or ``(N,)`` for a static sort.  If a single row / Series,
            the same sort is applied to every period.  If multiple
            columns, each column is treated as a separate characteristic,
            with returns as a single panel.
        n_quantiles: Number of quantile buckets (default 5).

    Returns:
        DataFrame of factor-mimicking portfolio returns ``(T, K)`` where
        *K* is the number of characteristics.
    """
    ret = returns.copy()
    T = len(ret)

    # If characteristics is 1-D or single-row, treat as static sort
    if characteristics.ndim == 1 or len(characteristics) == 1:
        chars_vals = np.asarray(characteristics).flatten()
        if len(chars_vals) != ret.shape[1]:
            msg = (
                f"characteristics length ({len(chars_vals)}) must match "
                f"number of assets ({ret.shape[1]})"
            )
            raise ValueError(msg)
        # Static quantile assignment
        quantiles = pd.qcut(chars_vals, q=n_quantiles, labels=False, duplicates="drop")
        top = quantiles == quantiles.max()
        bottom = quantiles == quantiles.min()
        long_short = ret.values[:, top].mean(axis=1) - ret.values[:, bottom].mean(axis=1)
        return pd.DataFrame(
            {"factor_1": long_short},
            index=ret.index,
        )

    # Time-varying characteristics: characteristics has same index/columns as returns
    if set(characteristics.columns) == set(returns.columns):
        # Single characteristic, time-varying
        ls_returns = np.full(T, np.nan)
        for t_idx in range(T):
            row_chars = characteristics.iloc[t_idx].dropna()
            valid_assets = row_chars.index.intersection(ret.columns)
            if len(valid_assets) < n_quantiles:
                continue
            c = row_chars.loc[valid_assets]
            r = ret.iloc[t_idx].loc[valid_assets]
            q = pd.qcut(c, q=n_quantiles, labels=False, duplicates="drop")
            top_mask = q == q.max()
            bottom_mask = q == q.min()
            if top_mask.sum() > 0 and bottom_mask.sum() > 0:
                ls_returns[t_idx] = float(r[top_mask].mean() - r[bottom_mask].mean())
        return pd.DataFrame({"factor_1": ls_returns}, index=ret.index)

    # Multiple characteristics as columns, returns as single panel
    # characteristics: (N, K), returns: (T, N)
    result: dict[str, np.ndarray] = {}
    for col in characteristics.columns:
        chars_vals = characteristics[col].values
        valid_mask = ~np.isnan(chars_vals)
        if valid_mask.sum() < n_quantiles:
            result[col] = np.full(T, np.nan)
            continue
        quantiles = pd.qcut(
            chars_vals[valid_mask],
            q=n_quantiles,
            labels=False,
            duplicates="drop",
        )
        # Map back to full asset array
        full_q = np.full(len(chars_vals), np.nan)
        full_q[valid_mask] = quantiles

        top = full_q == np.nanmax(full_q)
        bottom = full_q == np.nanmin(full_q)
        if top.sum() > 0 and bottom.sum() > 0:
            ls = ret.values[:, top].mean(axis=1) - ret.values[:, bottom].mean(axis=1)
        else:
            ls = np.full(T, np.nan)
        result[col] = ls

    return pd.DataFrame(result, index=ret.index)


# ---------------------------------------------------------------------------
# Risk factor decomposition
# ---------------------------------------------------------------------------


def risk_factor_decomposition(
    portfolio_returns: pd.Series | np.ndarray,
    factor_returns: pd.DataFrame | np.ndarray,
) -> dict:
    """Decompose portfolio risk into factor risk and idiosyncratic risk.

    Runs an OLS regression of *portfolio_returns* on *factor_returns* and
    computes the variance attributable to each factor and the residual
    (idiosyncratic) variance.

    Parameters:
        portfolio_returns: Portfolio return series ``(T,)``.
        factor_returns: Factor return matrix ``(T, K)``.

    Returns:
        Dictionary with:
        - ``total_variance``: total variance of portfolio returns.
        - ``factor_variance``: variance explained by the factor model.
        - ``idiosyncratic_variance``: residual variance.
        - ``factor_risk_share``: fraction of total variance from factors.
        - ``idiosyncratic_risk_share``: fraction of total variance from
          idiosyncratic risk.
        - ``betas``: regression coefficients (loadings) on each factor.
        - ``factor_marginal_contributions``: variance contribution of
          each factor.
    """
    from wraquant.stats.regression import ols as _ols

    y = np.asarray(portfolio_returns, dtype=float).ravel()
    F = np.asarray(factor_returns, dtype=float)
    if F.ndim == 1:
        F = F.reshape(-1, 1)

    K = F.shape[1]

    # OLS via shared regression module
    ols_result = _ols(y, F, add_constant=True)
    betas = ols_result["coefficients"][1:]  # factor betas
    residuals = ols_result["residuals"]

    total_var = float(np.var(y, ddof=1))
    idio_var = float(np.var(residuals, ddof=1))
    factor_var = total_var - idio_var

    # Marginal contribution: beta_i^2 * var(factor_i) + covariance terms
    cov_factors = np.cov(F, rowvar=False, ddof=1)
    if K == 1:
        cov_factors = cov_factors.reshape(1, 1)
    factor_marginal = np.array(
        [float(betas[i] * np.dot(cov_factors[i, :], betas)) for i in range(K)]
    )

    return {
        "total_variance": total_var,
        "factor_variance": max(factor_var, 0.0),
        "idiosyncratic_variance": idio_var,
        "factor_risk_share": max(factor_var, 0.0) / total_var if total_var > 0 else 0.0,
        "idiosyncratic_risk_share": idio_var / total_var if total_var > 0 else 0.0,
        "betas": betas,
        "factor_marginal_contributions": factor_marginal,
    }


# ---------------------------------------------------------------------------
# Factor correlation with significance
# ---------------------------------------------------------------------------


def factor_correlation(
    factor_returns: pd.DataFrame | np.ndarray,
) -> dict:
    """Compute the correlation matrix of factors with significance tests.

    For each pair of factors, the Pearson correlation and a two-sided
    *p*-value (testing H0: rho = 0) are computed.

    Parameters:
        factor_returns: Factor return matrix ``(T, K)``.

    Returns:
        Dictionary with:
        - ``correlation``: ndarray ``(K, K)`` correlation matrix.
        - ``p_values``: ndarray ``(K, K)`` of p-values for each pair.
    """
    F = np.asarray(factor_returns, dtype=float)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    K = F.shape[1]

    corr = np.corrcoef(F, rowvar=False)
    pvals = np.zeros((K, K))
    T = F.shape[0]

    for i in range(K):
        for j in range(K):
            if i == j:
                pvals[i, j] = 0.0
            else:
                r = corr[i, j]
                # t-statistic for correlation
                if abs(r) >= 1.0:
                    pvals[i, j] = 0.0
                else:
                    t_stat = r * np.sqrt((T - 2) / (1 - r ** 2))
                    pvals[i, j] = float(2.0 * sp_stats.t.sf(abs(t_stat), df=T - 2))

    return {
        "correlation": corr,
        "p_values": pvals,
    }


# ---------------------------------------------------------------------------
# Common factors across asset classes
# ---------------------------------------------------------------------------


def common_factors(
    returns_list: list[pd.DataFrame | np.ndarray],
    n_components: int = 3,
) -> dict:
    """Find common factors shared across multiple asset classes.

    Extracts PCA factors from each asset class independently, then
    performs canonical correlation analysis on the stacked factor scores
    to identify shared latent factors.

    Parameters:
        returns_list: List of returns matrices, one per asset class.
            Each has shape ``(T, N_i)`` and all must share the same
            number of time observations *T*.
        n_components: Number of PCA components to extract per asset
            class before cross-analysis.

    Returns:
        Dictionary with:
        - ``individual_factors``: list of factor arrays, one per asset
          class.
        - ``common_factor_scores``: ndarray ``(T, n_components)`` of
          shared factor scores obtained from a second-level PCA on
          concatenated individual factors.
        - ``cross_correlations``: correlation matrix between the
          individual factor sets.
    """
    factor_arrays: list[np.ndarray] = []

    for ret in returns_list:
        X = np.asarray(ret, dtype=float)
        X_c = X - X.mean(axis=0)
        T = X_c.shape[0]
        n_comp = min(n_components, X_c.shape[1])
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        scores = X_c @ Vt[:n_comp].T  # (T, n_comp)
        factor_arrays.append(scores)

    # Stack all factor scores and run a second PCA
    stacked = np.hstack(factor_arrays)  # (T, sum of n_comp_i)
    stacked_c = stacked - stacked.mean(axis=0)
    n_final = min(n_components, stacked_c.shape[1])
    U2, S2, Vt2 = np.linalg.svd(stacked_c, full_matrices=False)
    common_scores = stacked_c @ Vt2[:n_final].T  # (T, n_final)

    # Cross-correlation between factor sets
    cross_corr = np.corrcoef(stacked, rowvar=False)

    return {
        "individual_factors": factor_arrays,
        "common_factor_scores": common_scores,
        "cross_correlations": cross_corr,
    }


# ---------------------------------------------------------------------------
# Fama-French style factor construction
# ---------------------------------------------------------------------------


def fama_french_factors(
    returns: pd.DataFrame,
    characteristics: pd.DataFrame,
    n_quantiles: int = 3,
) -> pd.DataFrame:
    """Construct Fama-French style factors from a cross-section of returns.

    Sorts assets into portfolios based on each characteristic at each time
    period, then computes long-short (top-minus-bottom quantile) factor
    returns.  This replicates the standard methodology used by Fama and
    French to construct SMB, HML, and related factors.

    When to use:
        - To create custom factors from firm characteristics (e.g.,
          book-to-market, momentum, profitability, investment).
        - To replicate or extend the Fama-French factor zoo.
        - To test whether a new characteristic is a priced risk factor
          (construct the factor, then test its risk premium via
          ``fama_macbeth``).

    Mathematical formulation:
        For each period *t* and each characteristic *c*:

        1. Sort assets into *q* quantiles based on the characteristic.
        2. Compute the equal-weighted mean return for the top and bottom
           quantile.
        3. Factor return = ``mean(top quantile returns) - mean(bottom
           quantile returns)``.

        This is a zero-cost, long-short portfolio that isolates the
        return premium associated with the characteristic.

    How to interpret:
        - A consistently positive factor return means that assets with
          high values of the characteristic outperform those with low
          values (and vice versa for negative).
        - The *t*-statistic of the mean factor return tests whether the
          premium is significantly different from zero.
        - Standard Fama-French uses terciles (3 groups) for the main
          sort and independent double sorts for intersections (e.g.,
          size x value).

    Parameters:
        returns: DataFrame of asset returns ``(T, N)`` with a DatetimeIndex
            and asset names as columns.
        characteristics: DataFrame of asset characteristics ``(T, N)``
            with the same index and columns as *returns*, or a DataFrame
            with ``(N, K)`` where *N* assets are the index and *K*
            characteristics are the columns (static sort).
        n_quantiles: Number of quantile buckets (default 3 = terciles,
            matching Fama-French convention).

    Returns:
        DataFrame of factor returns ``(T, K)`` where *K* is the number
        of characteristics.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> T, N = 100, 30
        >>> dates = pd.bdate_range("2020-01-01", periods=T)
        >>> assets = [f"s{i}" for i in range(N)]
        >>> ret = pd.DataFrame(rng.normal(0, 0.02, (T, N)), index=dates, columns=assets)
        >>> bm = pd.DataFrame(rng.uniform(0.5, 3.0, (T, N)), index=dates, columns=assets)
        >>> ff = fama_french_factors(ret, bm)
        >>> ff.shape[0] == T
        True

    See Also:
        factor_mimicking_portfolios: General factor-mimicking portfolio
            construction.
        factor_exposure: Regress returns on constructed factors.
    """
    T = len(returns)

    # Case 1: characteristics has same index/columns as returns (time-varying)
    if (
        isinstance(characteristics, pd.DataFrame)
        and set(characteristics.columns) == set(returns.columns)
        and len(characteristics) == T
    ):
        factor_ret = np.full(T, np.nan)
        for t_idx in range(T):
            row_chars = characteristics.iloc[t_idx].dropna()
            valid = row_chars.index.intersection(returns.columns)
            if len(valid) < n_quantiles:
                continue
            c = row_chars.loc[valid]
            r = returns.iloc[t_idx].loc[valid]
            try:
                q = pd.qcut(c, q=n_quantiles, labels=False, duplicates="drop")
            except ValueError:
                continue
            top = q == q.max()
            bottom = q == q.min()
            if top.sum() > 0 and bottom.sum() > 0:
                factor_ret[t_idx] = float(r[top].mean() - r[bottom].mean())

        return pd.DataFrame({"factor_1": factor_ret}, index=returns.index)

    # Case 2: characteristics is (N, K) — multiple static characteristics
    result: dict[str, np.ndarray] = {}
    char_cols = characteristics.columns

    for col in char_cols:
        chars_vals = characteristics[col].values.astype(float)
        valid_mask = ~np.isnan(chars_vals)
        if valid_mask.sum() < n_quantiles:
            result[col] = np.full(T, np.nan)
            continue

        try:
            q = pd.qcut(
                chars_vals[valid_mask],
                q=n_quantiles,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            result[col] = np.full(T, np.nan)
            continue

        full_q = np.full(len(chars_vals), np.nan)
        full_q[valid_mask] = q
        top = full_q == np.nanmax(full_q)
        bottom = full_q == np.nanmin(full_q)

        if top.sum() > 0 and bottom.sum() > 0:
            # Map characteristics index to returns columns
            char_assets = characteristics.index
            ret_subset = returns[char_assets]
            ls = ret_subset.values[:, top].mean(axis=1) - ret_subset.values[:, bottom].mean(axis=1)
        else:
            ls = np.full(T, np.nan)

        result[col] = ls

    return pd.DataFrame(result, index=returns.index)


# ---------------------------------------------------------------------------
# Factor exposure (betas) via regression
# ---------------------------------------------------------------------------


def factor_exposure(
    returns: pd.DataFrame | pd.Series,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Regress returns on factor returns to estimate factor exposures (betas).

    For each asset (or a single portfolio), runs an OLS regression of
    returns on factor returns (with intercept) and reports the factor
    betas, t-statistics, and R-squared.

    When to use:
        - To estimate a portfolio's or asset's exposure to known factors
          (e.g., market, size, value, momentum).
        - As the first step in factor-based risk decomposition.
        - To validate that a factor-neutral strategy truly has zero
          exposure to target factors.

    Mathematical formulation:
        For each asset *i*:

        .. math::

            r_i = \\alpha_i + \\sum_{k=1}^K \\beta_{ik} f_k + \\epsilon_i

        The betas are the OLS coefficients on the factor returns.

    How to interpret:
        - ``beta > 0``: positive exposure to the factor (moves with it).
        - ``beta = 0``: no exposure.
        - ``|t_stat| > 2``: the exposure is statistically significant.
        - ``R_squared``: fraction of return variance explained by the
          factor model.  Higher R-squared means the model is a good fit.

    Parameters:
        returns: Asset returns.  A Series for a single asset or a
            DataFrame ``(T, N)`` for multiple assets.
        factor_returns: DataFrame of factor returns ``(T, K)``.

    Returns:
        DataFrame with one row per asset and columns:
        ``alpha``, ``beta_<factor_name>`` for each factor,
        ``t_<factor_name>`` for each factor's t-statistic,
        and ``r_squared``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> T = 200
        >>> mkt = pd.Series(rng.normal(0, 0.01, T), name="MKT")
        >>> smb = pd.Series(rng.normal(0, 0.005, T), name="SMB")
        >>> factors = pd.DataFrame({"MKT": mkt, "SMB": smb})
        >>> ret = 1.2 * mkt + 0.5 * smb + rng.normal(0, 0.003, T)
        >>> result = factor_exposure(pd.Series(ret, name="fund"), factors)
        >>> abs(result.loc["fund", "beta_MKT"] - 1.2) < 0.3
        True

    See Also:
        factor_loadings: Lower-level loadings estimation.
        factor_risk_decomposition: Risk decomposition from exposures.
    """
    import statsmodels.api as sm

    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    common = returns.index.intersection(factor_returns.index)
    ret_aligned = returns.loc[common]
    fact_aligned = factor_returns.loc[common]
    factor_names = list(fact_aligned.columns)

    X = sm.add_constant(fact_aligned.values.astype(float))
    rows: list[dict] = []

    for col in ret_aligned.columns:
        y = ret_aligned[col].values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < len(factor_names) + 2:
            continue

        model = sm.OLS(y[mask], X[mask]).fit()

        row: dict = {"asset": col, "alpha": float(model.params[0])}
        for i, fname in enumerate(factor_names):
            row[f"beta_{fname}"] = float(model.params[i + 1])
            row[f"t_{fname}"] = float(model.tvalues[i + 1])
        row["r_squared"] = float(model.rsquared)
        rows.append(row)

    result_df = pd.DataFrame(rows).set_index("asset")
    return result_df


# ---------------------------------------------------------------------------
# Factor risk decomposition
# ---------------------------------------------------------------------------


def factor_risk_decomposition(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> dict:
    """Decompose total risk into systematic (factor) and idiosyncratic components.

    Runs a factor model regression, then separates the total variance of
    the return series into the portion explained by the factors
    (systematic risk) and the unexplained residual (idiosyncratic risk).

    When to use:
        - To understand what fraction of a portfolio's risk comes from
          common factor exposures vs. stock-specific bets.
        - For risk budgeting: allocate risk limits to systematic and
          idiosyncratic components.
        - To evaluate diversification: a well-diversified portfolio has
          low idiosyncratic risk relative to total risk.

    Mathematical formulation:
        .. math::

            \\text{Var}(r) = \\beta' \\Sigma_f \\beta + \\sigma^2_\\epsilon

        where ``\\beta`` is the vector of factor exposures, ``\\Sigma_f``
        is the factor covariance matrix, and ``\\sigma^2_\\epsilon`` is
        the idiosyncratic variance.

        The R-squared of the regression gives the systematic risk share:

        .. math::

            R^2 = 1 - \\frac{\\sigma^2_\\epsilon}{\\text{Var}(r)}

    Parameters:
        returns: Return series for a single asset or portfolio.
        factor_returns: DataFrame of factor returns ``(T, K)``.

    Returns:
        Dictionary with:
        - ``systematic_risk``: variance explained by factors.
        - ``idiosyncratic_risk``: residual variance.
        - ``total_risk``: total return variance.
        - ``R_squared``: fraction of risk that is systematic.
        - ``betas``: factor exposure coefficients.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> mkt = rng.normal(0, 0.01, 200)
        >>> ret = pd.Series(1.2 * mkt + rng.normal(0, 0.005, 200))
        >>> factors = pd.DataFrame({"MKT": mkt})
        >>> result = factor_risk_decomposition(ret, factors)
        >>> result["R_squared"] > 0.3
        True

    See Also:
        risk_factor_decomposition: Lower-level decomposition with
            marginal contributions.
        factor_exposure: Factor exposure estimation.
    """
    import statsmodels.api as sm

    common = returns.index.intersection(factor_returns.index) if hasattr(returns, 'index') and hasattr(factor_returns, 'index') else range(min(len(returns), len(factor_returns)))

    if isinstance(common, pd.Index):
        y = returns.loc[common].values.astype(float)
        X = factor_returns.loc[common].values.astype(float)
    else:
        y = np.asarray(returns, dtype=float)[:len(common)]
        X = np.asarray(factor_returns, dtype=float)[:len(common)]

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    total_var = float(np.var(y, ddof=1))
    resid_var = float(np.var(model.resid, ddof=1))
    systematic_var = max(total_var - resid_var, 0.0)

    betas = model.params[1:]  # exclude intercept

    return {
        "systematic_risk": systematic_var,
        "idiosyncratic_risk": resid_var,
        "total_risk": total_var,
        "R_squared": float(model.rsquared),
        "betas": betas,
    }
