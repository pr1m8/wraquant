"""Robust statistical methods for financial data.

Standard statistical measures (mean, std, covariance) are sensitive to
outliers, fat tails, and data contamination -- all common in financial
data.  This module provides robust alternatives that remain reliable
under such conditions.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from wraquant.core.decorators import requires_extra


def mad(data: pd.Series | np.ndarray, scale: str = "normal") -> float:
    """Compute the Median Absolute Deviation (MAD).

    MAD is a robust measure of dispersion.  Unlike standard deviation,
    it is not influenced by a few extreme values, making it ideal for
    financial return distributions with fat tails.

    Parameters:
        data: Data series or array.
        scale: Scaling factor.  Use ``"normal"`` (default) so that the
            result is consistent with standard deviation for normally
            distributed data.  Use ``None`` for the raw MAD.

    Returns:
        MAD as a float.

    Example:
        >>> import pandas as pd
        >>> returns = pd.Series([0.01, 0.02, -0.01, -0.05, 0.10])
        >>> mad(returns)  # doctest: +SKIP
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    return float(sp_stats.median_abs_deviation(clean, scale=scale))


def winsorize(
    data: pd.Series | np.ndarray,
    lower: float = 0.05,
    upper: float = 0.05,
) -> pd.Series | np.ndarray:
    """Cap extreme values at given percentiles (Winsorization).

    Winsorization limits extreme values to reduce the influence of
    outliers without removing observations.  This is preferable to
    trimming when you want to keep the same sample size.

    Parameters:
        data: Data series or array.
        lower: Fraction to clip on the lower tail (default 5%).
        upper: Fraction to clip on the upper tail (default 5%).

    Returns:
        Winsorized data, same type as input.

    Example:
        >>> import pandas as pd
        >>> returns = pd.Series([0.01, 0.02, -0.50, 0.03, 0.80])
        >>> winsorize(returns, lower=0.1, upper=0.1)  # doctest: +SKIP
    """
    from scipy.stats.mstats import winsorize as _winsorize

    is_series = isinstance(data, pd.Series)
    clean = np.asarray(data, dtype=float)
    result = np.asarray(_winsorize(clean, limits=(lower, upper)))

    if is_series:
        return pd.Series(result, index=data.index, name=getattr(data, "name", None))
    return result


def trimmed_mean(
    data: pd.Series | np.ndarray,
    proportiontocut: float = 0.05,
) -> float:
    """Compute the trimmed mean, excluding extreme observations.

    The trimmed mean removes a fraction of the highest and lowest values
    before computing the average.  Use it when the mean is distorted by
    outliers (e.g., flash-crash returns).

    Parameters:
        data: Data series or array.
        proportiontocut: Fraction to cut from each tail (default 5%).

    Returns:
        Trimmed mean as a float.

    Example:
        >>> import numpy as np
        >>> data = np.array([1, 2, 3, 4, 100])
        >>> trimmed_mean(data, proportiontocut=0.2)  # doctest: +SKIP
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    return float(sp_stats.trim_mean(clean, proportiontocut))


def trimmed_std(
    data: pd.Series | np.ndarray,
    proportiontocut: float = 0.05,
) -> float:
    """Compute the standard deviation after trimming extreme values.

    Combines trimming with standard deviation computation for a measure
    of dispersion that is less sensitive to outliers than standard std
    but retains more information than MAD.

    Parameters:
        data: Data series or array.
        proportiontocut: Fraction to cut from each tail (default 5%).

    Returns:
        Trimmed standard deviation as a float.

    Example:
        >>> import numpy as np
        >>> data = np.array([1, 2, 3, 4, 100])
        >>> trimmed_std(data, proportiontocut=0.2)  # doctest: +SKIP
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    n = len(clean)
    n_cut = int(n * proportiontocut)
    if n_cut > 0:
        sorted_data = np.sort(clean)
        trimmed = sorted_data[n_cut : n - n_cut]
    else:
        trimmed = clean
    return float(np.std(trimmed, ddof=1))


def robust_zscore(data: pd.Series | np.ndarray) -> pd.Series:
    """Compute robust z-scores using median and MAD.

    Standard z-scores ``(x - mean) / std`` are heavily influenced by
    outliers.  Robust z-scores replace mean with median and std with MAD,
    providing a more reliable outlier detection metric for financial data.

    Parameters:
        data: Data series or array.

    Returns:
        Robust z-scores as a ``pd.Series``.

    Example:
        >>> import pandas as pd
        >>> returns = pd.Series([0.01, 0.02, -0.01, -0.05, 0.50])
        >>> robust_zscore(returns)  # doctest: +SKIP
    """
    arr = np.asarray(data, dtype=float)
    median = np.nanmedian(arr)
    mad_val = float(sp_stats.median_abs_deviation(arr[~np.isnan(arr)], scale="normal"))
    if mad_val == 0:
        mad_val = 1e-10  # avoid division by zero

    scores = (arr - median) / mad_val
    idx = data.index if isinstance(data, pd.Series) else None
    name = getattr(data, "name", None)
    return pd.Series(scores, index=idx, name=name)


@requires_extra("cleaning")
def robust_covariance(
    data: pd.DataFrame,
    support_fraction: float | None = None,
) -> dict:
    """Estimate a robust covariance matrix via Minimum Covariance Determinant.

    The MCD estimator finds the subset of observations (of a given
    fraction) whose classical covariance has the smallest determinant.
    This makes it highly resistant to outliers -- essential when
    computing portfolio covariance from return data that may contain
    erroneous prints or fat-tailed events.

    Parameters:
        data: DataFrame of asset returns (columns = assets).
        support_fraction: Fraction of data to use in support
            (default ``None`` lets sklearn choose).

    Returns:
        Dictionary with:
        - ``covariance``: robust covariance matrix (``np.ndarray``).
        - ``location``: robust location estimate (``np.ndarray``).
        - ``support_fraction``: fraction of data used.

    Example:
        >>> import pandas as pd, numpy as np
        >>> returns = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
        >>> robust_covariance(returns)  # doctest: +SKIP
    """
    from sklearn.covariance import MinCovDet

    clean = data.dropna().values
    mcd = MinCovDet(support_fraction=support_fraction)
    mcd.fit(clean)

    return {
        "covariance": mcd.covariance_,
        "location": mcd.location_,
        "support_fraction": mcd.support_fraction_,
    }


def huber_mean(
    data: pd.Series | np.ndarray,
    delta: float = 1.5,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> float:
    """Compute the Huber M-estimator of location.

    The Huber estimator behaves like the mean for observations within
    *delta* MAD-scaled deviations of the center, but limits the
    influence of observations beyond that threshold via iteratively
    reweighted least squares.  It provides a smooth trade-off between
    efficiency (mean) and robustness (median).

    Parameters:
        data: Data series or array.
        delta: Threshold parameter controlling robustness.  Smaller
            values give more robustness (closer to median).  Default
            1.5 is a standard choice.
        max_iter: Maximum number of IRLS iterations.
        tol: Convergence tolerance for the location estimate.

    Returns:
        Huber location estimate as a float.

    Example:
        >>> import numpy as np
        >>> data = np.array([1, 2, 3, 4, 100])
        >>> huber_mean(data, delta=1.5)  # doctest: +SKIP
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]

    # Iteratively reweighted least-squares for Huber loss
    mu = float(np.median(clean))
    for _ in range(max_iter):
        s = float(sp_stats.median_abs_deviation(clean, scale="normal"))
        if s == 0:
            s = 1e-10
        residuals = (clean - mu) / s
        abs_res = np.abs(residuals)
        # Avoid division by zero for residuals exactly at the center
        safe_abs_res = np.where(abs_res == 0, 1.0, abs_res)
        weights = np.where(abs_res <= delta, 1.0, delta / safe_abs_res)
        mu_new = float(np.average(clean, weights=weights))
        if abs(mu_new - mu) < tol:
            mu = mu_new
            break
        mu = mu_new

    return mu


def outlier_detection(
    data: pd.Series | np.ndarray,
    method: Literal["mad", "iqr", "grubbs"] = "mad",
    threshold: float = 3.0,
) -> dict:
    """Flag outliers using a robust detection method.

    Outlier detection is critical in finance for identifying data errors
    (bad ticks), extreme events, or contaminated observations before
    computing risk metrics.

    Parameters:
        data: Data series or array.
        method: Detection method:
            - ``"mad"``: Median Absolute Deviation (default).  Flag
              points whose robust z-score exceeds *threshold*.  Best
              general-purpose choice for financial data.
            - ``"iqr"``: Interquartile Range.  Flag points outside
              ``[Q1 - threshold*IQR, Q3 + threshold*IQR]``.  Classic
              box-plot method.
            - ``"grubbs"``: Grubbs' test for a single outlier.  Tests
              whether the most extreme value is an outlier assuming
              approximate normality.
        threshold: Sensitivity parameter (default 3.0).  For MAD this
            is the z-score cutoff; for IQR it is the multiplier.

    Returns:
        Dictionary with:
        - ``outliers``: boolean array (``True`` = outlier).
        - ``n_outliers``: count of flagged outliers.
        - ``method``: method used.

    Raises:
        ValueError: If *method* is not recognized.

    Example:
        >>> import pandas as pd
        >>> returns = pd.Series([0.01, 0.02, -0.01, -0.50, 0.03])
        >>> result = outlier_detection(returns, method="mad")
        >>> result["n_outliers"]  # doctest: +SKIP
    """
    clean = np.asarray(data, dtype=float)

    if method == "mad":
        median = np.nanmedian(clean)
        mad_val = float(
            sp_stats.median_abs_deviation(clean[~np.isnan(clean)], scale="normal")
        )
        if mad_val == 0:
            mad_val = 1e-10
        z = np.abs((clean - median) / mad_val)
        outliers = z > threshold

    elif method == "iqr":
        q1, q3 = np.nanpercentile(clean, [25, 75])
        iqr_val = q3 - q1
        lower = q1 - threshold * iqr_val
        upper = q3 + threshold * iqr_val
        outliers = (clean < lower) | (clean > upper)

    elif method == "grubbs":
        # Grubbs' test for a single outlier (two-sided)
        n = len(clean[~np.isnan(clean)])
        mean = np.nanmean(clean)
        std = np.nanstd(clean, ddof=1)
        if std == 0:
            outliers = np.zeros(len(clean), dtype=bool)
        else:
            g = np.abs(clean - mean) / std
            # Critical value from t-distribution
            t_crit = sp_stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
            g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
            outliers = g > g_crit

    else:
        msg = f"Unknown outlier detection method: {method!r}. Use 'mad', 'iqr', or 'grubbs'."
        raise ValueError(msg)

    # Handle NaN positions: mark as not outlier
    nan_mask = np.isnan(clean)
    outliers = outliers & ~nan_mask

    return {
        "outliers": outliers,
        "n_outliers": int(np.sum(outliers)),
        "method": method,
    }
