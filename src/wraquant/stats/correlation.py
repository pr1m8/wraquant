"""Correlation and covariance estimation for financial data.

Accurate correlation and covariance estimation is fundamental to
portfolio construction, risk management, and factor modelling. The
sample covariance matrix is a poor estimator when the number of assets
(p) is comparable to or exceeds the number of observations (T) --
a common situation in finance. This module provides shrinkage estimators
that regularise the covariance matrix for more stable and better-
conditioned estimates.

Key concepts:
    - **Shrinkage** blends the noisy sample covariance with a structured
      target (identity, diagonal, or constant correlation) to reduce
      estimation error.
    - The optimal shrinkage intensity balances bias (too much shrinkage)
      against variance (too little shrinkage).
    - All methods here produce positive semi-definite matrices, which is
      required for downstream use in optimisation.

References:
    - Ledoit & Wolf (2004), "A well-conditioned estimator for large-
      dimensional covariance matrices"
    - Chen, Wiesel, Eldar & Hero (2010), "Shrinkage Algorithms for MMSE
      Covariance Estimation" (OAS)
"""

from __future__ import annotations

import pandas as pd
from sklearn.covariance import OAS, LedoitWolf, ShrunkCovariance


def correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute a correlation matrix from asset returns.

    Pearson correlation measures linear dependence, Spearman measures
    monotonic rank dependence, and Kendall measures concordance of
    pairs. For financial returns, Pearson is standard but understates
    co-movement in the tails; Spearman and Kendall are more robust to
    outliers and non-linearity.

    When to use:
        - ``"pearson"`` (default): standard linear correlation. Use for
          most portfolio and factor analyses.
        - ``"spearman"``: rank correlation. Use when you suspect
          non-linear but monotonic relationships, or when returns have
          heavy tails / outliers.
        - ``"kendall"``: concordance-based. More robust than Spearman
          for small samples. Also connects naturally to copula models
          (Kendall's tau has a direct relationship to copula parameters).

    Parameters:
        returns: DataFrame of asset returns (columns = assets).
        method: Correlation method -- ``"pearson"``, ``"spearman"``,
            or ``"kendall"``.

    Returns:
        Correlation matrix as a DataFrame (p x p, symmetric, diagonal
        entries = 1.0).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
        >>> corr = correlation_matrix(returns)
        >>> corr.shape
        (3, 3)

    See Also:
        shrunk_covariance: Regularised covariance estimation.
        rolling_correlation: Time-varying pairwise correlation.
        wraquant.risk.copulas.rank_correlation: Kendall/Spearman for
            copula analysis.
    """
    return returns.corr(method=method)


def shrunk_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """Compute a shrinkage-estimated covariance matrix.

    Shrinkage estimators blend the sample covariance with a structured
    target to reduce estimation error, producing a better-conditioned
    matrix that is especially valuable when the number of assets (p)
    is large relative to the number of observations (T).

    When to use:
        Always prefer shrinkage over the raw sample covariance for
        portfolio optimisation. The improvement is largest when p/T
        is close to or exceeds 1 (e.g., 500 stocks with 252 daily
        observations).

        - ``"ledoit_wolf"`` (default): analytically optimal shrinkage
          toward a structured target. Best general-purpose choice.
          Automatically determines the optimal shrinkage intensity.
        - ``"oas"`` (Oracle Approximating Shrinkage): assumes the
          underlying distribution is Gaussian and computes the oracle-
          approximating shrinkage intensity. Slightly better than
          Ledoit-Wolf when normality holds.
        - ``"basic"``: simple shrinkage toward the diagonal with a
          fixed (non-optimal) shrinkage coefficient. Use only as a
          baseline.

    Mathematical formulation:
        Sigma_shrunk = (1 - alpha) * S + alpha * F

        where S is the sample covariance, F is the shrinkage target
        (e.g., identity or diagonal), and alpha is the shrinkage
        intensity (0 = no shrinkage, 1 = full shrinkage to target).

    How to interpret:
        The returned matrix is guaranteed positive semi-definite. Its
        eigenvalues are more dispersed than the sample covariance (less
        extreme), leading to more stable portfolio weights. Compare
        the condition number (ratio of max to min eigenvalue) before
        and after shrinkage to see the regularisation effect.

    Parameters:
        returns: DataFrame of asset returns (columns = assets).
        method: Shrinkage method -- ``"ledoit_wolf"`` (default),
            ``"oas"``, or ``"basic"``.

    Returns:
        Shrunk covariance matrix as a DataFrame (p x p).

    Raises:
        ValueError: If *method* is not recognized.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame(np.random.randn(100, 5), columns=list("ABCDE"))
        >>> cov = shrunk_covariance(returns, method="ledoit_wolf")
        >>> cov.shape
        (5, 5)

    See Also:
        correlation_matrix: Correlation (standardised covariance).
        wraquant.ml.preprocessing.denoised_correlation: Random matrix
            theory-based denoising.

    References:
        - Ledoit & Wolf (2004), "A well-conditioned estimator for
          large-dimensional covariance matrices"
        - Chen et al. (2010), "Shrinkage Algorithms for MMSE
          Covariance Estimation"
    """
    clean = returns.dropna()
    assets = clean.columns

    if method == "ledoit_wolf":
        estimator = LedoitWolf().fit(clean.values)
    elif method == "oas":
        estimator = OAS().fit(clean.values)
    elif method == "basic":
        estimator = ShrunkCovariance().fit(clean.values)
    else:
        msg = f"Unknown shrinkage method: {method!r}"
        raise ValueError(msg)

    return pd.DataFrame(estimator.covariance_, index=assets, columns=assets)


def rolling_correlation(
    x: pd.Series,
    y: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling Pearson correlation between two series.

    Rolling correlation reveals how the linear relationship between two
    assets evolves over time. Stable correlation is a key assumption in
    portfolio construction; large swings in rolling correlation indicate
    that static portfolio weights may be suboptimal.

    When to use:
        Use rolling correlation to:
        - Monitor diversification benefit over time (correlation
          rising toward 1.0 means diversification is eroding).
        - Detect correlation regime changes for pairs trading or
          hedging ratio adjustment.
        - Validate the stationarity assumption of portfolio
          optimisation inputs.

    How to interpret:
        - Values near +1.0: strong positive co-movement (little
          diversification benefit).
        - Values near 0.0: approximately uncorrelated.
        - Values near -1.0: strong negative co-movement (excellent
          diversification or natural hedge).
        - Spikes toward +1.0 during sell-offs are typical ("correlation
          goes to 1 in a crisis").

    Parameters:
        x: First return series.
        y: Second return series (same index).
        window: Rolling window size (e.g., 60 for ~3 months of
            daily data).

    Returns:
        Rolling Pearson correlation series. First ``window - 1``
        values are NaN.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> x = pd.Series(np.random.randn(200))
        >>> y = pd.Series(0.5 * x + np.random.randn(200) * 0.5)
        >>> rc = rolling_correlation(x, y, window=60)
        >>> rc.dropna().iloc[0] > 0
        True

    See Also:
        correlation_matrix: Full cross-asset correlation matrix.
        wraquant.risk.dcc.rolling_correlation_dcc: DCC-GARCH-based
            dynamic correlation.
    """
    return x.rolling(window).corr(y)
