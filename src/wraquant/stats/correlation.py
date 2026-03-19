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

import numpy as np
import pandas as pd
from sklearn.covariance import OAS, LedoitWolf, ShrunkCovariance

from wraquant.core._coerce import coerce_array, coerce_dataframe, coerce_series


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
    returns = coerce_dataframe(returns, "returns")
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
    x = coerce_series(x, "x")
    y = coerce_series(y, "y")
    return x.rolling(window).corr(y)


# ---------------------------------------------------------------------------
# Partial correlation
# ---------------------------------------------------------------------------


def partial_correlation(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the partial correlation matrix, controlling for all other variables.

    Partial correlation measures the linear association between two variables
    after removing the linear effect of all other variables in the dataset.
    This is essential in finance for understanding *direct* relationships
    between assets, factors, or macro variables --- as opposed to
    associations that are mediated through a common driver.

    When to use:
        Use partial correlation when you suspect that the observed
        correlation between two assets (or factors) is driven by a shared
        exposure to a third variable.  For example, two energy stocks may
        appear highly correlated, but partial correlation can reveal that
        after controlling for oil prices, the direct relationship is weak.

    Mathematical formulation:
        For each pair ``(i, j)``, regress both ``X_i`` and ``X_j`` on all
        remaining variables, then compute the Pearson correlation of the
        residuals:

        .. math::

            \\rho_{ij \\cdot \\text{rest}} = \\text{corr}(e_i, e_j)

        where ``e_i = X_i - \\hat{X}_i`` is the residual from regressing
        ``X_i`` on all other columns.

        Equivalently, partial correlations can be obtained from the inverse
        of the correlation matrix (the precision matrix):

        .. math::

            \\rho_{ij \\cdot \\text{rest}} = -\\frac{P_{ij}}{\\sqrt{P_{ii} P_{jj}}}

        where ``P = R^{-1}`` is the precision matrix.

    How to interpret:
        - Values near 0 indicate no *direct* linear relationship once
          shared drivers are removed.
        - A large drop from raw correlation to partial correlation signals
          that the association is mostly indirect (mediated).
        - The diagonal is always 1.0.

    Parameters:
        data: DataFrame with columns as variables (assets, factors, etc.)
            and rows as observations.  Must have at least 3 columns.

    Returns:
        Partial correlation matrix as a DataFrame (p x p, symmetric,
        diagonal = 1.0).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> z = np.random.randn(200)
        >>> data = pd.DataFrame({
        ...     "A": z + np.random.randn(200) * 0.3,
        ...     "B": z + np.random.randn(200) * 0.3,
        ...     "C": np.random.randn(200),
        ... })
        >>> pcorr = partial_correlation(data)
        >>> pcorr.shape
        (3, 3)

    See Also:
        correlation_matrix: Standard (marginal) correlation matrix.
        mutual_information: Non-linear dependence measure.
    """
    clean = data.dropna()
    cols = clean.columns
    p = len(cols)

    corr = clean.corr().values
    try:
        precision = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(corr)

    # Partial correlation from precision matrix
    diag = np.sqrt(np.diag(precision))
    outer = np.outer(diag, diag)
    pcorr = -precision / outer
    np.fill_diagonal(pcorr, 1.0)

    return pd.DataFrame(pcorr, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# Distance correlation
# ---------------------------------------------------------------------------


def distance_correlation(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> float:
    """Compute the Brownian distance correlation between two variables.

    Distance correlation (Szekely, Rizzo & Bakirov, 2007) is a measure of
    dependence that equals zero if and only if the two variables are
    independent --- unlike Pearson correlation, which only captures linear
    dependence.  This makes it invaluable for detecting nonlinear
    relationships in financial data (e.g., option-like payoffs, regime-
    dependent correlations, or tail dependence).

    When to use:
        - You suspect a nonlinear relationship that Pearson/Spearman will
          miss (e.g., a U-shaped or threshold relationship).
        - You want a single-number summary of *any* type of dependence.
        - You need a test statistic for independence that is consistent
          against all alternatives with finite first moments.

    Mathematical formulation:
        1. Compute the pairwise Euclidean distance matrices
           ``a_{kl} = |X_k - X_l|`` and ``b_{kl} = |Y_k - Y_l|``.
        2. Double-center each matrix:
           ``A_{kl} = a_{kl} - \\bar{a}_{k\\cdot} - \\bar{a}_{\\cdot l} + \\bar{a}_{\\cdot\\cdot}``
        3. Distance covariance squared:
           ``\\text{dCov}^2(X, Y) = \\frac{1}{n^2} \\sum_{k,l} A_{kl} B_{kl}``
        4. Distance correlation:
           ``\\text{dCor}(X, Y) = \\frac{\\text{dCov}(X, Y)}{\\sqrt{\\text{dVar}(X) \\cdot \\text{dVar}(Y)}}``

    How to interpret:
        - 0.0: independence (no dependence of any kind).
        - 1.0: perfect dependence (deterministic relationship).
        - Values between 0 and 1 indicate partial dependence.
        - Distance correlation >= |Pearson correlation|, so it always
          detects at least as much dependence.

    Parameters:
        x: First variable (1-D array or Series).
        y: Second variable (1-D array or Series, same length).

    Returns:
        Distance correlation as a float in [0, 1].

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> x = rng.normal(0, 1, 200)
        >>> y = x ** 2 + rng.normal(0, 0.3, 200)  # nonlinear
        >>> dcor = distance_correlation(x, y)
        >>> dcor > 0.3  # detects nonlinear dependence
        True

    References:
        Szekely, G. J., Rizzo, M. L. & Bakirov, N. K. (2007).
        "Measuring and testing dependence by correlation of distances."
        *Annals of Statistics*, 35(6), 2769-2794.

    See Also:
        correlation_matrix: Linear (Pearson) correlation.
        mutual_information: Information-theoretic dependence measure.
    """
    x_arr = coerce_array(x, "x")
    y_arr = coerce_array(y, "y")

    # Remove NaN pairs
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = len(x_arr)

    if n < 2:
        return 0.0

    def _dcov_sq(a: np.ndarray, b: np.ndarray) -> float:
        """Compute squared distance covariance."""
        a_dist = np.abs(a[:, None] - a[None, :])
        b_dist = np.abs(b[:, None] - b[None, :])

        # Double-center
        a_row_mean = a_dist.mean(axis=1, keepdims=True)
        a_col_mean = a_dist.mean(axis=0, keepdims=True)
        a_grand_mean = a_dist.mean()
        A = a_dist - a_row_mean - a_col_mean + a_grand_mean

        b_row_mean = b_dist.mean(axis=1, keepdims=True)
        b_col_mean = b_dist.mean(axis=0, keepdims=True)
        b_grand_mean = b_dist.mean()
        B = b_dist - b_row_mean - b_col_mean + b_grand_mean

        return float(np.mean(A * B))

    dcov_xy = _dcov_sq(x_arr, y_arr)
    dcov_xx = _dcov_sq(x_arr, x_arr)
    dcov_yy = _dcov_sq(y_arr, y_arr)

    denom = np.sqrt(dcov_xx * dcov_yy)
    if denom <= 0:
        return 0.0

    dcor_sq = dcov_xy / denom
    return float(np.sqrt(max(dcor_sq, 0.0)))


# ---------------------------------------------------------------------------
# Kendall's tau-b
# ---------------------------------------------------------------------------


def kendall_tau(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> dict:
    """Compute Kendall's tau-b rank correlation coefficient with p-value.

    Kendall's tau measures the ordinal association between two variables.
    It counts the number of concordant and discordant pairs: a pair
    ``(x_i, y_i), (x_j, y_j)`` is concordant if the ranks agree and
    discordant if they disagree.  The tau-b variant adjusts for ties.

    When to use:
        - When data has heavy tails or outliers that distort Pearson
          correlation.
        - For small samples where Spearman is less reliable.
        - When you need a rank-based measure that connects naturally to
          copula parameters (Kendall's tau has a one-to-one mapping to
          the parameter of many copula families).

    Mathematical formulation:
        .. math::

            \\tau_b = \\frac{C - D}{\\sqrt{(C + D + T_x)(C + D + T_y)}}

        where *C* = concordant pairs, *D* = discordant pairs, *T_x* and
        *T_y* = pairs tied only on *x* or *y*.

    How to interpret:
        - +1: perfect concordance (monotonically increasing relationship).
        - -1: perfect discordance (monotonically decreasing).
        - 0: no ordinal association.
        - |tau| > 0.3 is generally considered a moderate association in
          financial data.
        - The p-value tests H0: tau = 0 (independence).

    Parameters:
        x: First variable (1-D array or Series).
        y: Second variable (1-D array or Series, same length).

    Returns:
        Dictionary with:
        - ``tau``: Kendall's tau-b statistic.
        - ``p_value``: two-sided p-value for H0: tau = 0.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> x = rng.normal(0, 1, 100)
        >>> y = 0.8 * x + rng.normal(0, 0.5, 100)
        >>> result = kendall_tau(x, y)
        >>> result["tau"] > 0
        True
        >>> result["p_value"] < 0.05
        True

    See Also:
        correlation_matrix: Pearson/Spearman/Kendall full matrix.
        distance_correlation: Nonlinear dependence measure.
    """
    from scipy import stats as sp_stats

    x_arr = coerce_array(x, "x")
    y_arr = coerce_array(y, "y")

    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    tau, p_value = sp_stats.kendalltau(x_arr, y_arr, variant="b")

    return {
        "tau": float(tau),
        "p_value": float(p_value),
    }


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------


def mutual_information(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    n_bins: int = 20,
    method: str = "binning",
) -> float:
    """Estimate mutual information between two continuous variables.

    Mutual information (MI) quantifies the amount of information obtained
    about one variable by observing the other.  Unlike correlation, MI
    captures *any* type of statistical dependence --- linear, nonlinear,
    or even purely distributional.

    When to use:
        - Feature selection for ML-based trading models: MI identifies
          features with *any* predictive signal, not just linear ones.
        - Comparing the information content of different alpha signals.
        - Measuring the ``true'' dependence between returns and macro
          indicators that may have complex, non-monotonic relationships.

    Mathematical formulation:
        .. math::

            I(X; Y) = \\sum_{x} \\sum_{y} p(x, y) \\log \\frac{p(x, y)}{p(x) p(y)}

        For continuous variables, the sums become integrals.  The
        ``"binning"`` method discretises both variables into *n_bins*
        bins and computes MI on the resulting contingency table.  The
        ``"kde"`` method uses kernel density estimation for the joint
        and marginal densities.

    How to interpret:
        - MI = 0: independence (knowing X tells you nothing about Y).
        - MI > 0: some dependence exists.
        - MI is measured in nats (when using natural log) and is
          non-negative.
        - There is no upper bound in general, but normalised MI
          (MI / sqrt(H(X)*H(Y))) can be used for comparisons.

    Parameters:
        x: First continuous variable.
        y: Second continuous variable (same length).
        n_bins: Number of bins for discretisation (``"binning"`` method).
            More bins capture finer structure but need more data.
        method: Estimation method -- ``"binning"`` (default) or ``"kde"``.

    Returns:
        Estimated mutual information in nats (>= 0).

    Raises:
        ValueError: If *method* is not recognized.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> x = rng.normal(0, 1, 500)
        >>> y = x + rng.normal(0, 0.5, 500)
        >>> mi = mutual_information(x, y)
        >>> mi > 0
        True

    See Also:
        distance_correlation: Another non-linear dependence measure.
        correlation_matrix: Linear dependence only.
    """
    x_arr = coerce_array(x, "x")
    y_arr = coerce_array(y, "y")

    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = len(x_arr)

    if n < 10:
        return 0.0

    if method == "binning":
        # Discretise and compute MI from 2D histogram
        hist_2d, _, _ = np.histogram2d(x_arr, y_arr, bins=n_bins)
        # Convert to probabilities
        pxy = hist_2d / n
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

        return max(float(mi), 0.0)

    elif method == "kde":
        from scipy.stats import gaussian_kde

        # Joint KDE
        data_2d = np.vstack([x_arr, y_arr])
        kde_joint = gaussian_kde(data_2d)
        kde_x = gaussian_kde(x_arr)
        kde_y = gaussian_kde(y_arr)

        # Evaluate at data points
        log_joint = np.log(kde_joint(data_2d))
        log_marginal_x = np.log(kde_x(x_arr))
        log_marginal_y = np.log(kde_y(y_arr))

        mi = float(np.mean(log_joint - log_marginal_x - log_marginal_y))
        return max(mi, 0.0)

    else:
        msg = f"Unknown MI estimation method: {method!r}. Use 'binning' or 'kde'."
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Correlation significance test
# ---------------------------------------------------------------------------


def correlation_significance(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    method: str = "pearson",
    confidence: float = 0.95,
) -> dict:
    """Test whether the correlation between two variables is significantly non-zero.

    Computes the sample correlation, performs a t-test for the null
    hypothesis ``H0: rho = 0``, and constructs a confidence interval
    using Fisher's z-transformation.

    When to use:
        - To confirm that an observed correlation is statistically
          significant and not just sampling noise.
        - To obtain confidence intervals for reporting correlation
          estimates with uncertainty.
        - Before using a correlation estimate in portfolio construction
          or risk models --- insignificant correlations may be unreliable.

    Mathematical formulation:
        Test statistic:

        .. math::

            t = r \\sqrt{\\frac{n - 2}{1 - r^2}}

        which follows a t-distribution with ``n - 2`` degrees of freedom
        under H0.

        Confidence interval via Fisher z-transformation:

        .. math::

            z = \\text{arctanh}(r), \\quad SE = \\frac{1}{\\sqrt{n - 3}}

        The interval ``[z - z_{\\alpha/2} \\cdot SE, z + z_{\\alpha/2} \\cdot SE]``
        is back-transformed via ``tanh()`` to the correlation scale.

    Parameters:
        x: First variable.
        y: Second variable (same length).
        method: Correlation method -- ``"pearson"`` (default) or
            ``"spearman"``.
        confidence: Confidence level for the interval (default 0.95).

    Returns:
        Dictionary with:
        - ``r``: sample correlation coefficient.
        - ``t_stat``: t-test statistic.
        - ``p_value``: two-sided p-value for H0: rho = 0.
        - ``ci_lower``: lower bound of the confidence interval.
        - ``ci_upper``: upper bound of the confidence interval.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> x = rng.normal(0, 1, 100)
        >>> y = 0.5 * x + rng.normal(0, 1, 100)
        >>> result = correlation_significance(x, y)
        >>> result["p_value"] < 0.05
        True
        >>> result["ci_lower"] < result["r"] < result["ci_upper"]
        True

    See Also:
        correlation_matrix: Compute correlations without significance test.
        kendall_tau: Rank correlation with p-value.
    """
    from scipy import stats as sp_stats

    x_arr = coerce_array(x, "x")
    y_arr = coerce_array(y, "y")

    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = len(x_arr)

    if method == "spearman":
        r, p_value = sp_stats.spearmanr(x_arr, y_arr)
    else:
        r, p_value = sp_stats.pearsonr(x_arr, y_arr)

    r = float(r)
    p_value = float(p_value)

    # t-statistic
    if abs(r) >= 1.0:
        t_stat = float("inf") if r > 0 else float("-inf")
    else:
        t_stat = float(r * np.sqrt((n - 2) / (1 - r ** 2)))

    # Fisher z-transformation confidence interval
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3) if n > 3 else float("inf")
    alpha = 1.0 - confidence
    z_crit = float(sp_stats.norm.ppf(1 - alpha / 2))
    ci_lower = float(np.tanh(z - z_crit * se))
    ci_upper = float(np.tanh(z + z_crit * se))

    return {
        "r": r,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# ---------------------------------------------------------------------------
# Minimum spanning tree of correlation matrix
# ---------------------------------------------------------------------------


def minimum_spanning_tree_correlation(
    corr_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the minimum spanning tree (MST) of a correlation matrix.

    The MST is a connected, acyclic subgraph that connects all assets
    with the minimum total distance, where distance is derived from
    correlation.  It reveals the hierarchical structure of the market:
    which assets are the most "central" and how clusters of correlated
    assets are organized.

    When to use:
        - To visualise market structure and identify clusters of related
          assets (sectors, factor groups, regimes).
        - As input to hierarchical risk parity (HRP) portfolio
          construction (Lopez de Prado, 2016).
        - To detect changes in market structure over time by comparing
          MSTs across different periods.
        - To reduce dimensionality of the correlation matrix for
          network-based analysis.

    Mathematical formulation:
        The correlation matrix is converted to a distance matrix:

        .. math::

            d_{ij} = \\sqrt{2(1 - \\rho_{ij})}

        This metric satisfies the triangle inequality and maps perfect
        correlation (rho = 1) to zero distance and zero correlation
        (rho = 0) to distance sqrt(2).

        Prim's or Kruskal's algorithm is then applied to find the MST
        of the complete weighted graph.

    How to interpret:
        The returned adjacency matrix has non-zero entries only for
        edges in the MST.  The values are the correlation-derived
        distances.  Assets connected by short edges are highly correlated;
        the "hub" asset with the most edges is the most central.

    Parameters:
        corr_matrix: Correlation matrix as a DataFrame (p x p, symmetric).

    Returns:
        Adjacency matrix of the MST as a DataFrame (p x p).  Non-zero
        entries indicate edges in the tree, with values equal to the
        distance ``sqrt(2 * (1 - rho))``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD"))
        >>> corr = returns.corr()
        >>> mst = minimum_spanning_tree_correlation(corr)
        >>> mst.shape
        (4, 4)
        >>> (mst.values > 0).sum()  # MST has p-1 edges, each counted twice
        6

    References:
        - Mantegna, R. N. (1999). "Hierarchical structure in financial
          markets."
        - Lopez de Prado, M. (2016). "Building diversified portfolios
          that outperform out-of-sample."

    See Also:
        correlation_matrix: Compute the input correlation matrix.
        shrunk_covariance: Regularised covariance for more stable MSTs.
    """
    cols = corr_matrix.columns
    p = len(cols)
    corr_vals = corr_matrix.values.copy()

    # Distance matrix: d_ij = sqrt(2 * (1 - rho_ij))
    dist = np.sqrt(2.0 * (1.0 - corr_vals))
    np.fill_diagonal(dist, 0.0)

    # Prim's algorithm for MST
    in_tree = np.zeros(p, dtype=bool)
    adj = np.zeros((p, p), dtype=float)
    in_tree[0] = True

    for _ in range(p - 1):
        min_dist = float("inf")
        u_best, v_best = -1, -1

        for u in range(p):
            if not in_tree[u]:
                continue
            for v in range(p):
                if in_tree[v]:
                    continue
                if dist[u, v] < min_dist:
                    min_dist = dist[u, v]
                    u_best, v_best = u, v

        if u_best >= 0 and v_best >= 0:
            in_tree[v_best] = True
            adj[u_best, v_best] = min_dist
            adj[v_best, u_best] = min_dist

    return pd.DataFrame(adj, index=cols, columns=cols)
