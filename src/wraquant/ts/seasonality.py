"""Seasonality detection and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram


def detect_seasonality(
    data: pd.Series,
    max_period: int = 365,
) -> list[int]:
    """Detect dominant seasonal periods via spectral analysis.

    Uses Welch's periodogram to identify significant frequency peaks.

    Parameters:
        data: Time series to analyse.
        max_period: Maximum period to consider.

    Returns:
        List of detected seasonal periods (in number of observations),
        sorted by spectral power (strongest first).
    """
    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    clean = data.dropna().values
    freqs, power = periodogram(clean)

    # Ignore the DC component (index 0) and frequencies below 1/max_period
    min_freq = 1.0 / max_period if max_period > 0 else 0.0
    mask = freqs > min_freq
    freqs = freqs[mask]
    power = power[mask]

    if len(power) == 0:
        return []

    peak_indices, _ = find_peaks(power, height=np.median(power) * 3)
    if len(peak_indices) == 0:
        return []

    # Sort peaks by power (descending)
    sorted_peaks = sorted(peak_indices, key=lambda i: power[i], reverse=True)
    periods = []
    for idx in sorted_peaks:
        if freqs[idx] > 0:
            period = int(round(1.0 / freqs[idx]))
            if 2 <= period <= max_period and period not in periods:
                periods.append(period)
    return periods


def fourier_features(
    index: pd.DatetimeIndex,
    period: int,
    n_harmonics: int,
) -> pd.DataFrame:
    """Generate Fourier sine/cosine features for a datetime index.

    Useful for encoding seasonality as regression features.

    Parameters:
        index: Datetime index.
        period: Seasonal period (in the same time unit as the index).
        n_harmonics: Number of Fourier harmonics to generate.

    Returns:
        DataFrame with ``sin_k`` and ``cos_k`` columns for each harmonic.
    """
    t = np.arange(len(index), dtype=np.float64)
    columns: dict[str, np.ndarray] = {}
    for k in range(1, n_harmonics + 1):
        angle = 2 * np.pi * k * t / period
        columns[f"sin_{k}"] = np.sin(angle)
        columns[f"cos_{k}"] = np.cos(angle)
    return pd.DataFrame(columns, index=index)


# ---------------------------------------------------------------------------
# Multi-period Fourier Features
# ---------------------------------------------------------------------------


def multi_fourier_features(
    index: pd.DatetimeIndex,
    periods: list[int],
    n_harmonics: int | list[int] = 3,
) -> pd.DataFrame:
    """Generate Fourier terms for multiple seasonal periods.

    Creates sin/cos pairs for each period-harmonic combination,
    producing a feature matrix suitable for regression-based seasonal
    modelling (e.g., linear regression, Prophet-style models, or as
    exogenous regressors for ARIMA).

    For a period P and harmonic k, the features are:
        ``sin(2 * pi * k * t / P)`` and ``cos(2 * pi * k * t / P)``

    This captures up to k-th order seasonal variation within each period.

    Parameters:
        index: Datetime index of the time series.
        periods: List of seasonal periods (in the same units as the
            index). For example, ``[7, 365]`` for daily data with
            weekly and yearly seasonality.
        n_harmonics: Number of harmonics per period. Can be a single int
            (applied to all periods) or a list of ints (one per period).
            Default 3.

    Returns:
        DataFrame with columns named ``sin_P{period}_H{harmonic}`` and
        ``cos_P{period}_H{harmonic}`` for each period-harmonic pair.
        Shape is ``(len(index), 2 * sum(harmonics))``.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range("2020-01-01", periods=365, freq="D")
        >>> df = multi_fourier_features(idx, periods=[7, 365], n_harmonics=3)
        >>> df.shape[1]
        12
    """
    if isinstance(n_harmonics, int):
        harmonics_per_period = [n_harmonics] * len(periods)
    else:
        if len(n_harmonics) != len(periods):
            msg = "n_harmonics list must have same length as periods"
            raise ValueError(msg)
        harmonics_per_period = n_harmonics

    t = np.arange(len(index), dtype=np.float64)
    columns: dict[str, np.ndarray] = {}

    for period, n_h in zip(periods, harmonics_per_period):
        for k in range(1, n_h + 1):
            angle = 2 * np.pi * k * t / period
            columns[f"sin_P{period}_H{k}"] = np.sin(angle)
            columns[f"cos_P{period}_H{k}"] = np.cos(angle)

    return pd.DataFrame(columns, index=index)


# ---------------------------------------------------------------------------
# Seasonal Strength
# ---------------------------------------------------------------------------


def seasonal_strength(
    data: pd.Series,
    period: int | None = None,
) -> float:
    """Quantify the strength of seasonality in a time series.

    Uses the Wang, Smith, and Hyndman (2006) measure:
        ``F_s = max(0, 1 - Var(R_t) / Var(S_t + R_t))``

    where S_t is the seasonal component and R_t is the remainder from
    an STL decomposition.

    Returns a float in [0, 1]:
        - 0.0: no detectable seasonality (remainder dominates).
        - 1.0: perfectly seasonal (no noise).
        - > 0.9 is typically considered "strong" seasonality.
        - < 0.4 is typically "weak".

    Parameters:
        data: Time series to evaluate. NaN values are dropped.
        period: Seasonal period for the STL decomposition. If ``None``,
            defaults to 7.

    Returns:
        Strength of seasonality as a float in [0, 1].

    Example:
        >>> import numpy as np, pandas as pd
        >>> t = np.arange(200, dtype=float)
        >>> pure_seasonal = pd.Series(10 * np.sin(2 * np.pi * t / 20))
        >>> strength = seasonal_strength(pure_seasonal, period=20)
        >>> strength > 0.9
        True

    References:
        - Wang, X., Smith, K. & Hyndman, R. (2006), "Characteristic-
          Based Clustering for Time Series Data", Data Mining and
          Knowledge Discovery.
    """
    from statsmodels.tsa.seasonal import STL as _STL

    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    clean = data.dropna()
    if period is None:
        period = 7
    if len(clean) < 2 * period:
        return 0.0

    result = _STL(clean, period=period).fit()
    seasonal_component = result.seasonal
    remainder = result.resid

    var_remainder = np.var(remainder)
    var_deseasonalised = np.var(seasonal_component + remainder)

    if var_deseasonalised < 1e-15:
        return 0.0

    strength = max(0.0, 1.0 - var_remainder / var_deseasonalised)
    return float(strength)


# ---------------------------------------------------------------------------
# Multi-Seasonal Decomposition
# ---------------------------------------------------------------------------


def multi_seasonal_decompose(
    data: pd.Series,
    periods: list[int],
) -> dict:
    """Decompose a time series with multiple seasonal periods.

    Handles complex seasonality (e.g., daily data with both weekly and
    yearly patterns) by using MSTL (Multiple Seasonal-Trend decomposition
    using Loess) from statsmodels when available, or an iterative STL
    approach as fallback.

    Parameters:
        data: Time series to decompose. NaN values are dropped.
        periods: List of seasonal periods, ordered from shortest to
            longest. For example, ``[7, 365]`` for weekly and yearly
            seasonality in daily data.

    Returns:
        Dictionary with:
        - ``trend``: pd.Series of the trend component.
        - ``seasonal``: dict mapping each period to its pd.Series
          seasonal component.
        - ``residual``: pd.Series of the residual after removing
          trend and all seasonal components.

    Example:
        >>> import numpy as np, pandas as pd
        >>> t = np.arange(730, dtype=float)
        >>> weekly = 3 * np.sin(2 * np.pi * t / 7)
        >>> yearly = 5 * np.sin(2 * np.pi * t / 365)
        >>> trend = 0.01 * t
        >>> data = pd.Series(trend + weekly + yearly)
        >>> result = multi_seasonal_decompose(data, periods=[7, 365])
        >>> sorted(result['seasonal'].keys())
        [7, 365]

    References:
        - Bandara, K. et al. (2021), "MSTL: A Seasonal-Trend
          Decomposition Algorithm for Time Series with Multiple Seasonal
          Patterns", arXiv:2107.13462.
    """
    from statsmodels.tsa.seasonal import STL as _STL

    from wraquant.core._coerce import coerce_series

    data = coerce_series(data, name="data")
    clean = data.dropna()

    # Try MSTL first (available in statsmodels >= 0.14)
    try:
        from statsmodels.tsa.seasonal import MSTL

        mstl = MSTL(clean, periods=periods)
        result = mstl.fit()

        seasonal_dict: dict[int, pd.Series] = {}
        # MSTL returns seasonal as a DataFrame with one column per period
        if hasattr(result.seasonal, "columns"):
            for i, period in enumerate(periods):
                col = result.seasonal.iloc[:, i]
                seasonal_dict[period] = pd.Series(
                    col.values, index=clean.index, name=f"seasonal_{period}",
                )
        else:
            # Single seasonal component
            seasonal_dict[periods[0]] = pd.Series(
                result.seasonal.values if hasattr(result.seasonal, "values") else result.seasonal,
                index=clean.index,
                name=f"seasonal_{periods[0]}",
            )

        return {
            "trend": pd.Series(result.trend, index=clean.index, name="trend"),
            "seasonal": seasonal_dict,
            "residual": pd.Series(result.resid, index=clean.index, name="residual"),
        }

    except (ImportError, TypeError):
        pass

    # Fallback: iterative STL
    # Process periods from shortest to longest
    sorted_periods = sorted(periods)
    remainder = clean.copy()
    seasonal_dict = {}

    for period in sorted_periods:
        if len(remainder) < 2 * period:
            seasonal_dict[period] = pd.Series(
                0.0, index=clean.index, name=f"seasonal_{period}",
            )
            continue

        stl_result = _STL(remainder, period=period).fit()
        seasonal_dict[period] = pd.Series(
            stl_result.seasonal.values,
            index=clean.index,
            name=f"seasonal_{period}",
        )
        remainder = pd.Series(
            stl_result.trend.values + stl_result.resid.values,
            index=clean.index,
        )

    # Final STL on remainder to get trend
    if len(sorted_periods) > 0:
        final_period = sorted_periods[0]
        if len(remainder) >= 2 * final_period:
            final_stl = _STL(remainder, period=final_period).fit()
            trend = pd.Series(
                final_stl.trend.values, index=clean.index, name="trend",
            )
            residual = pd.Series(
                final_stl.resid.values, index=clean.index, name="residual",
            )
        else:
            trend = remainder
            residual = pd.Series(0.0, index=clean.index, name="residual")
    else:
        trend = remainder
        residual = pd.Series(0.0, index=clean.index, name="residual")

    return {
        "trend": trend,
        "seasonal": seasonal_dict,
        "residual": residual,
    }
