"""Time series anomaly detection.

Provides methods for detecting anomalous observations in time series
data using statistical and machine learning approaches:

1. **Isolation Forest** on rolling features -- detects contextual
   anomalies by featurising the local window (mean, std, skew) and
   running an isolation forest on the feature space.

2. **Forecast-based anomaly detection** -- fits a time series model
   and flags observations that deviate from the forecast by more than
   k standard deviations. Does NOT require fbprophet.

3. **Rolling Grubbs test** -- applies the classical Grubbs outlier test
   in a rolling window to detect local outliers.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from wraquant.core.decorators import requires_extra


# ---------------------------------------------------------------------------
# Isolation Forest on Time Series Features
# ---------------------------------------------------------------------------


@requires_extra("ml")
def isolation_forest_ts(
    data: pd.Series,
    window: int = 20,
    contamination: float = 0.05,
    features: list[str] | None = None,
    random_state: int = 42,
) -> dict:
    """Anomaly detection using Isolation Forest on rolling features.

    Rather than applying Isolation Forest directly to the raw values,
    this function first computes rolling window features (mean, std,
    skew, min, max) to capture the *context* of each observation. This
    approach detects **contextual anomalies** -- values that are unusual
    given their local context, even if they are not globally extreme.

    Parameters:
        data: Time series. NaN values are dropped.
        window: Rolling window size for feature computation (default 20).
        contamination: Expected proportion of anomalies in the data
            (default 0.05). Controls the threshold for the Isolation
            Forest decision function.
        features: List of rolling features to compute. If ``None``,
            uses ``["mean", "std", "skew", "min", "max"]``. Supported
            values: ``"mean"``, ``"std"``, ``"skew"``, ``"min"``,
            ``"max"``, ``"median"``, ``"range"``.
        random_state: Random seed for reproducibility (default 42).

    Returns:
        Dictionary with:
        - ``anomaly_scores``: pd.Series of anomaly scores from the
          Isolation Forest decision function (lower = more anomalous).
        - ``anomaly_labels``: pd.Series of int labels (1 = normal,
          -1 = anomaly).
        - ``threshold``: float, the decision function threshold.
        - ``anomaly_indices``: list of index values flagged as
          anomalies.
        - ``n_anomalies``: int, number of detected anomalies.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> x = rng.normal(0, 1, 500)
        >>> x[100] = 20.0  # inject anomaly
        >>> x[300] = -15.0  # inject anomaly
        >>> data = pd.Series(x)
        >>> result = isolation_forest_ts(data, window=20)
        >>> result['n_anomalies'] > 0
        True

    References:
        - Liu, F.T. et al. (2008), "Isolation Forest", ICDM.
    """
    from sklearn.ensemble import IsolationForest

    clean = data.dropna()

    if features is None:
        features = ["mean", "std", "skew", "min", "max"]

    # Compute rolling features
    rolling = clean.rolling(window=window, min_periods=max(window // 2, 2))
    feature_cols: dict[str, pd.Series] = {}

    for feat in features:
        if feat == "mean":
            feature_cols["roll_mean"] = rolling.mean()
        elif feat == "std":
            feature_cols["roll_std"] = rolling.std()
        elif feat == "skew":
            feature_cols["roll_skew"] = rolling.skew()
        elif feat == "min":
            feature_cols["roll_min"] = rolling.min()
        elif feat == "max":
            feature_cols["roll_max"] = rolling.max()
        elif feat == "median":
            feature_cols["roll_median"] = rolling.median()
        elif feat == "range":
            feature_cols["roll_range"] = rolling.max() - rolling.min()

    # Also include the raw value and its deviation from rolling mean
    feature_cols["value"] = clean
    if "roll_mean" in feature_cols:
        feature_cols["deviation"] = clean - feature_cols["roll_mean"]

    feature_df = pd.DataFrame(feature_cols, index=clean.index).dropna()

    # Fit Isolation Forest
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    feature_matrix = feature_df.values
    labels = iso.fit_predict(feature_matrix)
    scores = iso.decision_function(feature_matrix)

    # Build result series aligned to the feature DataFrame index
    anomaly_scores = pd.Series(scores, index=feature_df.index, name="anomaly_score")
    anomaly_labels = pd.Series(labels, index=feature_df.index, name="anomaly_label")

    anomaly_mask = labels == -1
    anomaly_indices = list(feature_df.index[anomaly_mask])
    threshold = float(iso.offset_)

    return {
        "anomaly_scores": anomaly_scores,
        "anomaly_labels": anomaly_labels,
        "threshold": threshold,
        "anomaly_indices": anomaly_indices,
        "n_anomalies": int(anomaly_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Forecast-Based Anomaly Detection
# ---------------------------------------------------------------------------


def prophet_anomaly(
    data: pd.Series,
    k: float = 3.0,
    seasonal_period: int | None = None,
    trend: bool = True,
) -> dict:
    """Forecast-based anomaly detection using statsmodels.

    Fits an Unobserved Components Model (local level + optional seasonal)
    and flags observations where the actual value deviates from the
    smoothed state by more than ``k`` standard deviations of the
    residuals.

    This approach does NOT require fbprophet or Prophet. It uses
    pure statsmodels for a fully open-source, lightweight solution.

    Parameters:
        data: Time series. NaN values are dropped.
        k: Number of standard deviations for the anomaly threshold
            (default 3.0). Lower values flag more anomalies.
        seasonal_period: Seasonal period for the model. If ``None``,
            no seasonal component is included.
        trend: Include a trend component (default True).

    Returns:
        Dictionary with:
        - ``forecast``: pd.Series of smoothed / fitted values.
        - ``residuals``: pd.Series of (actual - forecast).
        - ``anomaly_mask``: pd.Series of bool, True for anomalies.
        - ``anomaly_indices``: list of index values flagged as
          anomalies.
        - ``upper_bound``: pd.Series, forecast + k * sigma.
        - ``lower_bound``: pd.Series, forecast - k * sigma.
        - ``sigma``: float, standard deviation of residuals.
        - ``n_anomalies``: int, number of detected anomalies.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> x = rng.normal(0, 1, 300)
        >>> x[50] = 15.0  # anomaly
        >>> x[200] = -12.0  # anomaly
        >>> data = pd.Series(x)
        >>> result = prophet_anomaly(data, k=3.0)
        >>> result['n_anomalies'] >= 2
        True

    References:
        - Harvey, A.C. (1989), *Forecasting, Structural Time Series
          Models and the Kalman Filter*. Cambridge University Press.
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    clean = data.dropna()

    level_type = "local linear trend" if trend else "local level"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = UnobservedComponents(
            clean,
            level=level_type,
            seasonal=seasonal_period,
            stochastic_seasonal=True if seasonal_period else False,
        )
        fit = model.fit(disp=False, maxiter=500)

    # Smoothed state = fitted values
    fitted = pd.Series(fit.fittedvalues, index=clean.index, name="fitted")
    residuals = clean - fitted

    sigma = float(np.std(residuals.dropna()))
    if sigma < 1e-10:
        sigma = 1e-10

    upper = fitted + k * sigma
    lower = fitted - k * sigma

    anomaly_mask = (clean > upper) | (clean < lower)
    anomaly_indices = list(clean.index[anomaly_mask])

    return {
        "forecast": fitted,
        "residuals": residuals,
        "anomaly_mask": anomaly_mask,
        "anomaly_indices": anomaly_indices,
        "upper_bound": pd.Series(upper, index=clean.index, name="upper"),
        "lower_bound": pd.Series(lower, index=clean.index, name="lower"),
        "sigma": sigma,
        "n_anomalies": int(anomaly_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Rolling Grubbs Test
# ---------------------------------------------------------------------------


def grubbs_test_ts(
    data: pd.Series,
    window: int = 50,
    significance: float = 0.05,
) -> dict:
    """Rolling Grubbs test for outlier detection in time series.

    Applies the Grubbs test (maximum normed residual test) within a
    rolling window to detect observations that are statistically
    unlikely given the local distribution.

    The Grubbs test statistic for a sample of size n is:
        ``G = max|x_i - mean| / std``

    and the critical value is derived from the t-distribution:
        ``G_crit = ((n-1) / sqrt(n)) * sqrt(t^2 / (n - 2 + t^2))``

    where ``t`` is the critical value of the t-distribution with
    ``n-2`` degrees of freedom at significance ``alpha / (2n)``.

    Parameters:
        data: Time series. NaN values are dropped.
        window: Rolling window size (default 50). Should be large
            enough for the Grubbs test to be meaningful (>= 7).
        significance: Significance level for the Grubbs test
            (default 0.05).

    Returns:
        Dictionary with:
        - ``outlier_mask``: pd.Series of bool, True for detected
          outliers.
        - ``test_statistics``: pd.Series of Grubbs G statistics at
          each position.
        - ``outlier_indices``: list of index values flagged as outliers.
        - ``n_outliers``: int, number of detected outliers.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> x = rng.normal(0, 1, 200)
        >>> x[75] = 20.0  # obvious outlier
        >>> data = pd.Series(x)
        >>> result = grubbs_test_ts(data, window=50)
        >>> 75 in result['outlier_indices']
        True

    References:
        - Grubbs, F.E. (1950), "Sample Criteria for Testing Outlying
          Observations", Annals of Mathematical Statistics.
    """
    clean = data.dropna()
    n_obs = len(clean)
    values = clean.values.astype(np.float64)

    outlier_flags = np.zeros(n_obs, dtype=bool)
    g_stats = np.full(n_obs, np.nan)

    half_w = window // 2

    for i in range(n_obs):
        start = max(0, i - half_w)
        end = min(n_obs, i + half_w + 1)
        w = values[start:end]
        n_w = len(w)

        if n_w < 7:
            continue

        mean_w = np.mean(w)
        std_w = np.std(w, ddof=1)

        if std_w < 1e-15:
            continue

        # Grubbs statistic for the center point
        g = abs(values[i] - mean_w) / std_w
        g_stats[i] = g

        # Critical value
        t_crit = sp_stats.t.ppf(1 - significance / (2 * n_w), n_w - 2)
        g_crit = ((n_w - 1) / np.sqrt(n_w)) * np.sqrt(
            t_crit ** 2 / (n_w - 2 + t_crit ** 2)
        )

        if g > g_crit:
            outlier_flags[i] = True

    outlier_mask = pd.Series(outlier_flags, index=clean.index, name="outlier")
    test_statistics = pd.Series(g_stats, index=clean.index, name="grubbs_G")
    outlier_indices = list(clean.index[outlier_flags])

    return {
        "outlier_mask": outlier_mask,
        "test_statistics": test_statistics,
        "outlier_indices": outlier_indices,
        "n_outliers": int(outlier_flags.sum()),
    }
