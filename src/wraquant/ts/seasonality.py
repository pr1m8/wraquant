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
