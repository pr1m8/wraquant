"""Distribution fitting and tail analysis for financial data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def fit_distribution(data: pd.Series, dist: str = "norm") -> dict:
    """Fit a parametric distribution to data.

    Parameters:
        data: Data series to fit.
        dist: Name of a ``scipy.stats`` distribution (e.g., ``"norm"``,
            ``"t"``, ``"lognorm"``).

    Returns:
        Dictionary with ``params`` (tuple of fitted parameters),
        ``ks_statistic``, and ``ks_pvalue`` from a Kolmogorov-Smirnov
        goodness-of-fit test.

    Raises:
        AttributeError: If *dist* is not a valid scipy distribution.
    """
    clean = data.dropna().values
    distribution = getattr(sp_stats, dist)
    params = distribution.fit(clean)
    ks_stat, ks_p = sp_stats.kstest(clean, dist, args=params)
    return {
        "params": params,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


def tail_ratio(returns: pd.Series, quantile: float = 0.05) -> float:
    """Compute the tail ratio (right tail / left tail).

    A tail ratio > 1 indicates a fatter right tail (more extreme gains)
    relative to the left tail.

    Parameters:
        returns: Return series.
        quantile: Quantile for tail measurement (default 5%).

    Returns:
        Tail ratio as a float.
    """
    clean = returns.dropna()
    right = abs(clean.quantile(1 - quantile))
    left = abs(clean.quantile(quantile))
    if left == 0:
        return float("inf")
    return float(right / left)


def hurst_exponent(data: pd.Series) -> float:
    """Estimate the Hurst exponent via rescaled range (R/S) analysis.

    The Hurst exponent characterises the long-term memory of a series:

    - ``H < 0.5``: mean-reverting
    - ``H = 0.5``: random walk
    - ``H > 0.5``: trending / persistent

    Parameters:
        data: Time series (prices or returns).

    Returns:
        Estimated Hurst exponent as a float.
    """
    clean = data.dropna().values
    n = len(clean)

    max_k = max(2, n // 2)
    sizes = []
    rs_values = []

    size = 8
    while size <= max_k:
        sizes.append(size)
        n_chunks = n // size
        rs_list = []
        for i in range(n_chunks):
            chunk = clean[i * size : (i + 1) * size]
            mean = chunk.mean()
            deviations = chunk - mean
            cumdev = np.cumsum(deviations)
            r = cumdev.max() - cumdev.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append(np.mean(rs_list))
        size *= 2

    if len(sizes) < 2:
        return 0.5

    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)
    slope, _intercept = np.polyfit(log_sizes, log_rs, 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Extended distribution analysis
# ---------------------------------------------------------------------------


def fit_stable_distribution(data: pd.Series | np.ndarray) -> dict:
    """Fit a stable (Levy) distribution to data.

    Uses scipy's ``levy_stable`` distribution to estimate the four
    parameters: alpha (stability), beta (skewness), loc, and scale.

    Parameters:
        data: Data array or series.

    Returns:
        Dictionary with:
        - ``alpha``: stability parameter (0, 2].
        - ``beta``: skewness parameter [-1, 1].
        - ``loc``: location parameter.
        - ``scale``: scale parameter.
        - ``ks_statistic``: Kolmogorov-Smirnov statistic.
        - ``ks_pvalue``: KS test p-value.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]

    params = sp_stats.levy_stable.fit(clean)
    alpha, beta, loc, scale = params

    ks_stat, ks_p = sp_stats.kstest(
        clean, "levy_stable", args=(alpha, beta, loc, scale)
    )

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "loc": float(loc),
        "scale": float(scale),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


def tail_index(
    data: pd.Series | np.ndarray,
    method: str = "hill",
    threshold_quantile: float = 0.9,
) -> dict:
    """Estimate the tail index of a distribution.

    The tail index characterises the heaviness of distribution tails.
    A finite tail index indicates power-law tails (Pareto-like).

    Parameters:
        data: Data array or series.
        method: Estimation method.  One of ``"hill"`` (Hill estimator),
            ``"pickands"`` (Pickands estimator), or ``"moment"``
            (moment estimator of Dekkers-Einmahl-de Haan).
        threshold_quantile: Quantile above which tail observations are
            used (default 0.9, i.e. top 10%).

    Returns:
        Dictionary with:
        - ``tail_index``: estimated tail index (xi).
        - ``method``: method used.
        - ``n_tail``: number of observations in the tail.

    Raises:
        ValueError: If *method* is not one of the supported estimators.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    # Work with absolute values for tail analysis
    abs_data = np.abs(clean)
    threshold = float(np.quantile(abs_data, threshold_quantile))
    tail_obs = np.sort(abs_data[abs_data > threshold])[::-1]
    n_tail = len(tail_obs)

    if n_tail < 4:
        return {"tail_index": float("nan"), "method": method, "n_tail": n_tail}

    if method == "hill":
        # Hill estimator: 1/xi = (1/k) * sum(log(X_(i)) - log(X_(k+1)))
        # where X_(1) >= X_(2) >= ... are order statistics
        k = n_tail - 1
        log_ratio = np.log(tail_obs[:k]) - np.log(tail_obs[k])
        hill_est = float(np.mean(log_ratio))
        xi = hill_est  # gamma = 1/xi for Pareto, but xi itself is the tail index

    elif method == "pickands":
        # Pickands estimator using quartiles of tail
        m = n_tail // 4
        if m < 1:
            return {"tail_index": float("nan"), "method": method, "n_tail": n_tail}
        x1 = tail_obs[m - 1]
        x2 = tail_obs[2 * m - 1]
        x4 = tail_obs[4 * m - 1] if 4 * m <= n_tail else tail_obs[-1]
        denom = x2 - x4
        if denom <= 0:
            xi = 0.0
        else:
            xi = float(np.log((x1 - x2) / denom) / np.log(2.0))

    elif method == "moment":
        # Moment estimator (Dekkers-Einmahl-de Haan)
        k = n_tail - 1
        log_ratio = np.log(tail_obs[:k]) - np.log(tail_obs[k])
        m1 = float(np.mean(log_ratio))
        m2 = float(np.mean(log_ratio ** 2))
        xi = m1 + 1.0 - 0.5 / (1.0 - m1 ** 2 / m2) if m2 > 0 else m1

    else:
        msg = f"Unknown method '{method}'. Use 'hill', 'pickands', or 'moment'."
        raise ValueError(msg)

    return {
        "tail_index": float(xi),
        "method": method,
        "n_tail": n_tail,
    }


def qqplot_data(
    data: pd.Series | np.ndarray,
    dist: str = "norm",
) -> dict:
    """Generate quantile-quantile plot data.

    Computes theoretical and sample quantiles for constructing a Q-Q
    plot against a reference distribution.

    Parameters:
        data: Data array or series.
        dist: Name of a ``scipy.stats`` distribution to use as the
            theoretical reference (default ``"norm"``).

    Returns:
        Dictionary with:
        - ``theoretical_quantiles``: array of theoretical quantiles.
        - ``sample_quantiles``: array of ordered sample values.
        - ``slope``: slope of the best-fit line through the Q-Q plot.
        - ``intercept``: intercept of the best-fit line.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    clean = np.sort(clean)
    n = len(clean)

    distribution = getattr(sp_stats, dist)
    # Fit distribution to data for theoretical quantiles
    params = distribution.fit(clean)
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical = distribution.ppf(probabilities, *params)

    # Best-fit line
    slope, intercept = np.polyfit(theoretical, clean, 1)

    return {
        "theoretical_quantiles": theoretical,
        "sample_quantiles": clean,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def jarque_bera(data: pd.Series | np.ndarray) -> dict:
    """Perform the Jarque-Bera test for normality.

    Tests the null hypothesis that the data is normally distributed,
    based on sample skewness and kurtosis.

    Parameters:
        data: Data array or series.

    Returns:
        Dictionary with:
        - ``statistic``: Jarque-Bera test statistic.
        - ``p_value``: p-value of the test.
        - ``skewness``: sample skewness.
        - ``kurtosis``: sample excess kurtosis.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    n = len(clean)

    skew = float(sp_stats.skew(clean, bias=False))
    kurt = float(sp_stats.kurtosis(clean, bias=False))  # excess kurtosis

    jb_stat = (n / 6.0) * (skew ** 2 + (kurt ** 2) / 4.0)
    p_value = float(1.0 - sp_stats.chi2.cdf(jb_stat, df=2))

    return {
        "statistic": float(jb_stat),
        "p_value": p_value,
        "skewness": skew,
        "kurtosis": kurt,
    }


def kolmogorov_smirnov(
    data: pd.Series | np.ndarray,
    dist: str = "norm",
) -> dict:
    """Perform the Kolmogorov-Smirnov goodness-of-fit test.

    Tests the null hypothesis that *data* was drawn from the specified
    distribution.  The distribution parameters are first estimated via
    MLE.

    Parameters:
        data: Data array or series.
        dist: Name of a ``scipy.stats`` distribution (default ``"norm"``).

    Returns:
        Dictionary with:
        - ``statistic``: KS test statistic.
        - ``p_value``: p-value of the test.
        - ``dist``: distribution name tested.
        - ``params``: fitted distribution parameters.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]

    distribution = getattr(sp_stats, dist)
    params = distribution.fit(clean)
    ks_stat, ks_p = sp_stats.kstest(clean, dist, args=params)

    return {
        "statistic": float(ks_stat),
        "p_value": float(ks_p),
        "dist": dist,
        "params": params,
    }


# ---------------------------------------------------------------------------
# Anderson-Darling and best-fit distribution
# ---------------------------------------------------------------------------


def anderson_darling(
    data: pd.Series | np.ndarray,
    dist: str = "norm",
) -> dict:
    """Perform the Anderson-Darling goodness-of-fit test.

    The Anderson-Darling test is more sensitive to deviations in the
    tails than the Kolmogorov-Smirnov test, making it more appropriate
    for financial data where tail behaviour matters most (e.g., VaR
    and CVaR estimation).

    Parameters:
        data: Data array or series.
        dist: Distribution to test against.  Supported values depend on
            ``scipy.stats.anderson`` and include ``"norm"``, ``"expon"``,
            ``"logistic"``, ``"gumbel"``, ``"gumbel_l"``, ``"gumbel_r"``.

    Returns:
        Dictionary with:
        - ``statistic``: Anderson-Darling test statistic.
        - ``critical_values``: array of critical values for each
          significance level.
        - ``significance_levels``: corresponding significance levels (%).

    Example:
        >>> import numpy as np
        >>> data = np.random.default_rng(42).normal(0, 1, 1000)
        >>> anderson_darling(data)  # doctest: +SKIP
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    result = sp_stats.anderson(clean, dist=dist)

    return {
        "statistic": float(result.statistic),
        "critical_values": result.critical_values,
        "significance_levels": result.significance_level,
    }


def kernel_density_estimate(
    data: np.ndarray | pd.Series,
    n_points: int = 200,
    bandwidth: str | float = "scott",
) -> dict[str, np.ndarray]:
    """Kernel density estimation using Gaussian kernels.

    Non-parametric density estimation that makes no assumptions about
    the underlying distribution shape. Useful for visualizing return
    distributions, computing non-parametric VaR, and comparing
    regime-specific densities.

    Parameters:
        data: Sample data (1D array of returns or prices).
        n_points: Number of evaluation points for the density curve.
        bandwidth: Bandwidth method ("scott", "silverman") or float.

    Returns:
        Dictionary containing:
        - **x** -- Evaluation points (n_points,).
        - **density** -- Estimated density values (n_points,).
        - **bandwidth** -- Bandwidth used.
        - **mode** -- Location of peak density.
        - **cdf** -- Cumulative distribution values (n_points,).

    Example:
        >>> from wraquant.stats.distributions import kernel_density_estimate
        >>> kde = kernel_density_estimate(returns, n_points=500)
        >>> var_95 = kde['x'][np.searchsorted(kde['cdf'], 0.05)]
    """
    from scipy.stats import gaussian_kde

    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]

    kde = gaussian_kde(clean, bw_method=bandwidth)

    # Create evaluation grid spanning the data range with padding
    data_min, data_max = clean.min(), clean.max()
    data_range = data_max - data_min
    padding = data_range * 0.15
    x = np.linspace(data_min - padding, data_max + padding, n_points)

    density = kde.evaluate(x)

    # Compute CDF via cumulative trapezoidal integration
    cdf = np.cumsum(density)
    dx = x[1] - x[0]
    cdf = cdf * dx
    # Normalize to ensure CDF ends at 1
    cdf = cdf / cdf[-1]

    mode_idx = np.argmax(density)

    return {
        "x": x,
        "density": density,
        "bandwidth": float(kde.factor),
        "mode": float(x[mode_idx]),
        "cdf": cdf,
    }


def best_fit_distribution(
    data: pd.Series | np.ndarray,
    candidates: list[str] | None = None,
) -> pd.DataFrame:
    """Rank candidate distributions by goodness of fit.

    Fits multiple parametric distributions to the data and ranks them
    by AIC and KS/AD statistics.  Use this to choose the best model
    for return distributions when the assumption of normality fails
    (which it usually does in finance).

    Parameters:
        data: Data array or series.
        candidates: List of ``scipy.stats`` distribution names to test.
            Defaults to ``["norm", "t", "skewnorm", "gennorm", "nct",
            "johnsonsu"]`` -- a set well-suited for financial returns.

    Returns:
        DataFrame with columns: ``distribution``, ``params``,
        ``ks_statistic``, ``ad_statistic``, ``aic``, sorted by AIC
        (ascending).

    Example:
        >>> import numpy as np
        >>> data = np.random.default_rng(42).standard_t(df=5, size=1000)
        >>> best_fit_distribution(data)  # doctest: +SKIP
    """
    if candidates is None:
        candidates = ["norm", "t", "skewnorm", "gennorm", "nct", "johnsonsu"]

    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    n = len(clean)

    rows: list[dict] = []
    for name in candidates:
        try:
            dist_obj = getattr(sp_stats, name)
            params = dist_obj.fit(clean)

            # KS test
            ks_stat, _ = sp_stats.kstest(clean, name, args=params)

            # Anderson-Darling (only for 'norm'; otherwise use KS)
            try:
                ad_result = sp_stats.anderson(clean, dist=name)
                ad_stat = float(ad_result.statistic)
            except ValueError:
                # anderson() only supports a limited set of distributions
                ad_stat = float("nan")

            # AIC = 2k - 2ln(L)
            k = len(params)
            log_likelihood = float(np.sum(dist_obj.logpdf(clean, *params)))
            aic = 2 * k - 2 * log_likelihood

            rows.append(
                {
                    "distribution": name,
                    "params": params,
                    "ks_statistic": float(ks_stat),
                    "ad_statistic": ad_stat,
                    "aic": float(aic),
                }
            )
        except Exception:  # noqa: BLE001
            # Skip distributions that fail to fit
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("aic").reset_index(drop=True)
    return df
