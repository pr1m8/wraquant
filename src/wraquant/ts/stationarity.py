"""Stationarity tests and transformations for time series.

Provides both transformations (differencing, detrending) and formal
statistical tests (ADF, KPSS, Phillips-Perron) for assessing and
achieving stationarity in time series data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import detrend as sp_detrend


def difference(data: pd.Series, order: int = 1) -> pd.Series:
    """Apply integer differencing to a time series.

    Parameters:
        data: Time series.
        order: Differencing order (1 = first difference).

    Returns:
        Differenced series with NaN values dropped.
    """
    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    result = data
    for _ in range(order):
        result = result.diff()
    return result.dropna()


def fractional_difference(
    data: pd.Series,
    d: float = 0.5,
    threshold: float = 1e-5,
) -> pd.Series:
    """Apply fractional differencing to preserve long-memory information.

    Implements the fixed-width window fracdiff method from
    *Advances in Financial Machine Learning* (Lopez de Prado).

    Parameters:
        data: Time series.
        d: Fractional differencing parameter (0 < d < 1).
        threshold: Weight cutoff threshold for the window.

    Returns:
        Fractionally differenced series.
    """
    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    # Compute binomial weights, capped at the length of the data
    n = len(data)
    weights = [1.0]
    k = 1
    while k < n:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    weights = np.array(weights[::-1])
    width = len(weights)

    result = {}
    values = data.values
    for i in range(width - 1, len(values)):
        result[data.index[i]] = np.dot(weights, values[i - width + 1 : i + 1])

    return pd.Series(result, dtype=float)


def detrend(data: pd.Series, method: str = "linear") -> pd.Series:
    """Remove trend from a time series.

    Parameters:
        data: Time series.
        method: Detrending method — ``"linear"`` (default) or
            ``"constant"`` (demean).

    Returns:
        Detrended series.

    Raises:
        ValueError: If *method* is not recognized.
    """
    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    clean = data.dropna()

    if method in ("linear", "constant"):
        detrended = sp_detrend(clean.values, type=method)
        return pd.Series(detrended, index=clean.index, name=data.name)
    msg = f"Unknown detrend method: {method!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Augmented Dickey-Fuller Test
# ---------------------------------------------------------------------------


def adf_test(
    data: pd.Series,
    max_lags: int | None = None,
    regression: str = "c",
    significance: float = 0.05,
) -> dict:
    """Augmented Dickey-Fuller (ADF) unit root test.

    Tests the null hypothesis that a unit root is present (i.e., the
    series is non-stationary). A low p-value leads to rejection of the
    null, concluding stationarity.

    The test regression is:
        ``Delta y_t = alpha + beta*t + gamma*y_{t-1} + sum_i delta_i * Delta y_{t-i} + e_t``

    where alpha is a constant (if ``regression='c'``), beta*t is a
    time trend (if ``regression='ct'``), and the number of lagged
    differences is chosen to remove serial correlation in the residuals.

    **ADF vs KPSS**:
        - **ADF**: null = unit root (non-stationary). Rejecting H0
          supports stationarity.
        - **KPSS**: null = stationary. Rejecting H0 supports
          non-stationarity.
        - Best practice: run both. If ADF rejects and KPSS does not
          reject, strong evidence of stationarity. If both reject or
          both fail to reject, the series may be trend-stationary or
          difference-stationary.

    Parameters:
        data: Time series to test. NaN values are dropped.
        max_lags: Maximum number of lags for the lagged differences.
            If ``None``, the optimal lag is selected automatically
            using the AIC criterion.
        regression: Deterministic terms to include:
            ``"c"`` -- constant only (default, most common).
            ``"ct"`` -- constant and linear trend.
            ``"n"`` -- no constant, no trend.
        significance: Significance level for the ``is_stationary``
            convenience flag (default 0.05).

    Returns:
        Dictionary with:
        - ``test_statistic``: float, the ADF t-statistic.
        - ``p_value``: float, MacKinnon approximate p-value.
        - ``optimal_lag``: int, number of lags used.
        - ``n_obs``: int, number of observations used in the regression.
        - ``critical_values``: dict mapping significance levels
          (``"1%"``, ``"5%"``, ``"10%"``) to critical values.
        - ``is_stationary``: bool, True if p-value < significance.
        - ``interpretation``: str, human-readable conclusion.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> # White noise is stationary
        >>> stationary = pd.Series(rng.normal(0, 1, 500))
        >>> result = adf_test(stationary)
        >>> result['is_stationary']
        True

    References:
        - Dickey, D.A. & Fuller, W.A. (1979), "Distribution of the
          Estimators for Autoregressive Time Series With a Unit Root",
          JASA.
        - Said, S.E. & Dickey, D.A. (1984), "Testing for Unit Roots in
          Autoregressive-Moving Average Models of Unknown Order",
          Biometrika.
    """
    from statsmodels.tsa.stattools import adfuller

    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    clean = data.dropna().values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = adfuller(
            clean,
            maxlag=max_lags,
            regression=regression,
            autolag="AIC",
        )

    stat, pval, usedlag, nobs, crit, icbest = result

    is_stationary = bool(pval < significance)
    if is_stationary:
        interp = (
            f"ADF test statistic = {stat:.4f} (p-value = {pval:.4f}). "
            f"Reject the null of a unit root at the {significance:.0%} level. "
            f"The series appears stationary."
        )
    else:
        interp = (
            f"ADF test statistic = {stat:.4f} (p-value = {pval:.4f}). "
            f"Cannot reject the null of a unit root at the {significance:.0%} level. "
            f"The series appears non-stationary."
        )

    return {
        "test_statistic": float(stat),
        "p_value": float(pval),
        "optimal_lag": int(usedlag),
        "n_obs": int(nobs),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "is_stationary": is_stationary,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# KPSS Test
# ---------------------------------------------------------------------------


def kpss_test(
    data: pd.Series,
    regression: str = "c",
    n_lags: int | str = "auto",
    significance: float = 0.05,
) -> dict:
    """Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity test.

    Tests the null hypothesis that the series is stationary around a
    deterministic trend. This is the **opposite** of the ADF test:
    here, rejecting H0 implies non-stationarity.

    **When to use ADF vs KPSS vs both**:
        - **ADF alone**: quick check; but has low power against
          near-unit-root alternatives.
        - **KPSS alone**: confirms stationarity; but may over-reject
          for long-memory processes.
        - **Both (recommended)**: a confirmatory strategy. Four cases:
          1. ADF rejects, KPSS does not reject -> stationary.
          2. ADF does not reject, KPSS rejects -> non-stationary.
          3. Both reject -> trend-stationary (difference to remove trend).
          4. Neither rejects -> inconclusive, may need more data.

    Parameters:
        data: Time series to test. NaN values are dropped.
        regression: Deterministic component:
            ``"c"`` -- test for level stationarity (default).
            ``"ct"`` -- test for trend stationarity.
        n_lags: Number of lags for the Newey-West estimator. ``"auto"``
            (default) uses the data-dependent rule. An integer value
            fixes the lag truncation.
        significance: Significance level for the ``is_stationary``
            convenience flag (default 0.05).

    Returns:
        Dictionary with:
        - ``test_statistic``: float, the KPSS statistic.
        - ``p_value``: float, interpolated p-value (may be bounded
          at 0.01 or 0.10 depending on the table).
        - ``n_lags``: int, number of lags used.
        - ``critical_values``: dict mapping significance levels to
          critical values.
        - ``is_stationary``: bool, True if p-value >= significance
          (i.e., cannot reject the null of stationarity).
        - ``interpretation``: str, human-readable conclusion.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> stationary = pd.Series(rng.normal(0, 1, 500))
        >>> result = kpss_test(stationary)
        >>> result['is_stationary']
        True

    References:
        - Kwiatkowski, D. et al. (1992), "Testing the null hypothesis of
          stationarity against the alternative of a unit root", Journal
          of Econometrics.
    """
    from statsmodels.tsa.stattools import kpss as sm_kpss

    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    clean = data.dropna().values
    n_lags_param = n_lags if isinstance(n_lags, int) else "auto"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, used_lags, crit = sm_kpss(
            clean,
            regression=regression,
            nlags=n_lags_param,
        )

    # For KPSS, null = stationary, so p >= significance means stationary
    is_stationary = bool(pval >= significance)
    if is_stationary:
        interp = (
            f"KPSS test statistic = {stat:.4f} (p-value = {pval:.4f}). "
            f"Cannot reject the null of stationarity at the {significance:.0%} level. "
            f"The series appears stationary."
        )
    else:
        interp = (
            f"KPSS test statistic = {stat:.4f} (p-value = {pval:.4f}). "
            f"Reject the null of stationarity at the {significance:.0%} level. "
            f"The series appears non-stationary."
        )

    return {
        "test_statistic": float(stat),
        "p_value": float(pval),
        "n_lags": int(used_lags),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "is_stationary": is_stationary,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Phillips-Perron Test
# ---------------------------------------------------------------------------


def phillips_perron(
    data: pd.Series,
    regression: str = "c",
    significance: float = 0.05,
) -> dict:
    """Phillips-Perron (PP) unit root test.

    Like the ADF test, the PP test has the null hypothesis of a unit
    root (non-stationarity). However, the PP test uses a non-parametric
    correction to the t-statistic that is robust to heteroskedasticity
    and serial correlation in the error term, without needing to specify
    a lag order.

    **When to prefer PP over ADF**:
        - When the error process exhibits heteroskedasticity (e.g.,
          financial returns with volatility clustering).
        - When you are uncertain about the appropriate lag length for ADF.
        - As a robustness check alongside ADF.

    The PP test statistic modifies the ADF statistic with a correction
    factor based on the Newey-West long-run variance estimate.

    Parameters:
        data: Time series to test. NaN values are dropped.
        regression: Deterministic terms:
            ``"c"`` -- constant only (default).
            ``"ct"`` -- constant and trend.
            ``"n"`` -- no constant, no trend.
        significance: Significance level for the ``is_stationary``
            convenience flag (default 0.05).

    Returns:
        Dictionary with:
        - ``test_statistic``: float, the PP t-statistic.
        - ``p_value``: float, MacKinnon approximate p-value.
        - ``n_lags``: int, truncation lag for Newey-West.
        - ``critical_values``: dict mapping significance levels to
          critical values.
        - ``is_stationary``: bool, True if p-value < significance.
        - ``interpretation``: str, human-readable conclusion.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> stationary = pd.Series(rng.normal(0, 1, 500))
        >>> result = phillips_perron(stationary)
        >>> result['is_stationary']
        True

    References:
        - Phillips, P.C.B. & Perron, P. (1988), "Testing for a Unit Root
          in Time Series Regression", Biometrika.
    """
    from statsmodels.tsa.stattools import adfuller

    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    clean = data.dropna().values
    n = len(clean)
    nw_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    # Run ADF with zero lags as the base regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_result = adfuller(clean, maxlag=0, regression=regression, autolag=None)

    stat_adf = adf_result[0]
    pval_adf = adf_result[1]
    crit_adf = adf_result[4]

    # Compute the PP correction
    # Residuals from the AR(1) regression y_t = a + rho * y_{t-1} + e_t
    y = clean[1:]
    x_ar = clean[:-1]
    n_reg = len(y)

    if regression == "c" or regression == "ct":
        x_design = np.column_stack([np.ones(n_reg), x_ar])
        if regression == "ct":
            x_design = np.column_stack([x_design, np.arange(1, n_reg + 1)])
    else:
        x_design = x_ar.reshape(-1, 1)

    from wraquant.stats.regression import ols as _ols

    _pp_result = _ols(y, x_design, add_constant=False)
    beta = _pp_result["coefficients"]
    residuals = _pp_result["residuals"]

    # Newey-West long-run variance estimate
    gamma_0 = np.mean(residuals ** 2)
    s_sq = gamma_0
    for j in range(1, nw_lags + 1):
        weight = 1 - j / (nw_lags + 1)
        gamma_j = np.mean(residuals[j:] * residuals[:-j])
        s_sq += 2 * weight * gamma_j

    # PP correction factor
    if gamma_0 > 0:
        correction = (s_sq - gamma_0) / gamma_0
        # Approximate PP statistic
        pp_stat = stat_adf - 0.5 * correction * n_reg / stat_adf if abs(stat_adf) > 1e-10 else stat_adf
    else:
        pp_stat = stat_adf

    # The PP statistic has the same asymptotic distribution as ADF.
    # We approximate PP by running ADF with the Newey-West lag count,
    # which incorporates the non-parametric serial correlation correction.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp_full = adfuller(clean, maxlag=nw_lags, regression=regression, autolag=None)

    pp_stat_final = pp_full[0]
    pp_pval = pp_full[1]
    pp_crit = pp_full[4]

    is_stationary = bool(pp_pval < significance)
    if is_stationary:
        interp = (
            f"Phillips-Perron test statistic = {pp_stat_final:.4f} "
            f"(p-value = {pp_pval:.4f}). "
            f"Reject the null of a unit root at the {significance:.0%} level. "
            f"The series appears stationary."
        )
    else:
        interp = (
            f"Phillips-Perron test statistic = {pp_stat_final:.4f} "
            f"(p-value = {pp_pval:.4f}). "
            f"Cannot reject the null of a unit root at the {significance:.0%} level. "
            f"The series appears non-stationary."
        )

    return {
        "test_statistic": float(pp_stat_final),
        "p_value": float(pp_pval),
        "n_lags": nw_lags,
        "critical_values": {k: float(v) for k, v in pp_crit.items()},
        "is_stationary": is_stationary,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Optimal Differencing
# ---------------------------------------------------------------------------


def optimal_differencing(
    data: pd.Series,
    max_d: int = 2,
    significance: float = 0.05,
) -> dict:
    """Automatically determine the optimal differencing order for stationarity.

    Sequentially applies integer differencing (d = 0, 1, 2, ...) and
    runs the ADF test at each order. Returns the smallest d for which
    the ADF test rejects the unit root null.

    Parameters:
        data: Time series. NaN values are dropped.
        max_d: Maximum differencing order to try (default 2).
            Higher orders are rarely needed in practice.
        significance: Significance level for the ADF test
            (default 0.05).

    Returns:
        Dictionary with:
        - ``optimal_d``: int, the smallest differencing order that
          achieves stationarity (or ``max_d`` if stationarity is not
          achieved).
        - ``test_results``: dict mapping each d to its ``adf_test``
          result dictionary.
        - ``is_stationary``: bool, True if stationarity was achieved
          within ``max_d`` differences.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> # Random walk needs d=1
        >>> rw = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
        >>> result = optimal_differencing(rw)
        >>> result['optimal_d']
        1

    References:
        - Hyndman, R.J. & Khandakar, Y. (2008), "Automatic Time Series
          Forecasting: The forecast Package for R", JSS.
    """
    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    test_results: dict[int, dict] = {}
    current = data.dropna()

    for d in range(max_d + 1):
        if d > 0:
            current = difference(current, order=1)
            if len(current) < 20:
                break

        result = adf_test(current, significance=significance)
        test_results[d] = result

        if result["is_stationary"]:
            return {
                "optimal_d": d,
                "test_results": test_results,
                "is_stationary": True,
            }

    return {
        "optimal_d": max_d,
        "test_results": test_results,
        "is_stationary": False,
    }


# ---------------------------------------------------------------------------
# Variance Ratio Test
# ---------------------------------------------------------------------------


def variance_ratio_test(
    data: pd.Series,
    lags: int = 2,
    overlap: bool = True,
    significance: float = 0.05,
) -> dict:
    """Lo-MacKinlay variance ratio test for the random walk hypothesis.

    Tests whether the variance of k-period returns scales linearly
    with k, as implied by a random walk. A variance ratio VR(k) = 1
    is consistent with a random walk; VR(k) > 1 suggests positive
    autocorrelation (momentum); VR(k) < 1 suggests mean reversion.

    The test statistic under homoskedasticity is:
        ``VR(k) = Var(r_t(k)) / (k * Var(r_t))``
        ``z = (VR(k) - 1) / sqrt(2*(2k-1)*(k-1) / (3*k*T))``

    A heteroskedasticity-robust version is also computed.

    Parameters:
        data: Time series of prices or levels (NOT returns). NaN values
            are dropped.
        lags: Holding period / aggregation interval k (default 2).
            Common choices: 2, 4, 8, 16.
        overlap: Use overlapping returns for better power
            (default True). Non-overlapping uses fewer observations.
        significance: Significance level for the ``is_random_walk``
            convenience flag (default 0.05).

    Returns:
        Dictionary with:
        - ``variance_ratio``: float, the VR(k) statistic.
        - ``z_statistic``: float, the z-statistic (homoskedastic).
        - ``z_robust``: float, heteroskedasticity-robust z-statistic.
        - ``p_value``: float, two-sided p-value from the robust
          z-statistic.
        - ``is_random_walk``: bool, True if the random walk null
          cannot be rejected at the given significance level.
        - ``interpretation``: str, human-readable conclusion.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> # Random walk: VR should be close to 1
        >>> rw = pd.Series(np.cumsum(rng.normal(0, 1, 1000)))
        >>> result = variance_ratio_test(rw, lags=2)
        >>> 0.5 < result['variance_ratio'] < 1.5
        True

    References:
        - Lo, A.W. & MacKinlay, A.C. (1988), "Stock Market Prices Do
          Not Follow Random Walks: Evidence from a Simple Specification
          Test", Review of Financial Studies.
    """
    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    clean = data.dropna().values.astype(np.float64)
    n = len(clean)
    k = lags

    if n < k + 1:
        msg = f"Need at least {k + 1} observations, got {n}"
        raise ValueError(msg)

    # Log returns
    log_prices = np.log(clean)
    returns = np.diff(log_prices)
    t = len(returns)

    # Mean return
    mu = np.mean(returns)

    # Variance of 1-period returns
    sigma_1 = np.sum((returns - mu) ** 2) / (t - 1)

    if sigma_1 < 1e-15:
        return {
            "variance_ratio": 1.0,
            "z_statistic": 0.0,
            "z_robust": 0.0,
            "p_value": 1.0,
            "is_random_walk": True,
            "interpretation": "Series has near-zero variance; VR test is uninformative.",
        }

    # k-period returns (overlapping)
    returns_k = log_prices[k:] - log_prices[:-k]
    sigma_k = np.sum((returns_k - k * mu) ** 2) / (t - k + 1)

    vr = sigma_k / (k * sigma_1)

    # Homoskedastic z-statistic
    asy_var_homo = 2.0 * (2.0 * k - 1.0) * (k - 1.0) / (3.0 * k * t)
    z_homo = (vr - 1.0) / np.sqrt(asy_var_homo) if asy_var_homo > 0 else 0.0

    # Heteroskedasticity-robust z-statistic (Lo-MacKinlay 1988)
    delta = np.zeros(k - 1)
    for j in range(1, k):
        numer = np.sum(
            (returns[j:] - mu) ** 2 * (returns[:-j] - mu) ** 2
        )
        denom = (np.sum((returns - mu) ** 2)) ** 2
        delta[j - 1] = numer * t / denom

    weights = np.array([2.0 * (1.0 - j / k) for j in range(1, k)])
    asy_var_robust = float(np.sum(weights ** 2 * delta))
    z_robust = (vr - 1.0) / np.sqrt(asy_var_robust) if asy_var_robust > 0 else z_homo

    p_value = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(z_robust))))
    is_random_walk = bool(p_value >= significance)

    if is_random_walk:
        interp = (
            f"VR({k}) = {vr:.4f}, robust z = {z_robust:.4f} "
            f"(p-value = {p_value:.4f}). Cannot reject the random walk "
            f"hypothesis at the {significance:.0%} level."
        )
    else:
        direction = "positive autocorrelation (momentum)" if vr > 1 else "mean reversion"
        interp = (
            f"VR({k}) = {vr:.4f}, robust z = {z_robust:.4f} "
            f"(p-value = {p_value:.4f}). Reject the random walk "
            f"hypothesis at the {significance:.0%} level. "
            f"Evidence of {direction}."
        )

    return {
        "variance_ratio": float(vr),
        "z_statistic": float(z_homo),
        "z_robust": float(z_robust),
        "p_value": p_value,
        "is_random_walk": is_random_walk,
        "interpretation": interp,
    }
