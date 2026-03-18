"""Time series analysis and forecasting.

Covers decomposition, seasonality detection, change-point detection,
stationarity transformations, and forecasting.
"""

from wraquant.ts.advanced import (
    sktime_forecast,
    statsforecast_auto,
    stumpy_matrix_profile,
    tsfresh_features,
    tslearn_dtw,
    tslearn_kmeans,
    wavelet_denoise,
    wavelet_transform,
)
from wraquant.ts.changepoint import cusum, detect_changepoints
from wraquant.ts.decomposition import seasonal_decompose, stl_decompose, trend_filter
from wraquant.ts.forecasting import auto_arima, exponential_smoothing
from wraquant.ts.seasonality import detect_seasonality, fourier_features
from wraquant.ts.stationarity import detrend, difference, fractional_difference

__all__ = [
    # decomposition
    "seasonal_decompose",
    "stl_decompose",
    "trend_filter",
    # seasonality
    "detect_seasonality",
    "fourier_features",
    # changepoint
    "cusum",
    "detect_changepoints",
    # stationarity
    "difference",
    "fractional_difference",
    "detrend",
    # forecasting
    "exponential_smoothing",
    "auto_arima",
    # advanced integrations
    "tsfresh_features",
    "stumpy_matrix_profile",
    "wavelet_transform",
    "wavelet_denoise",
    "sktime_forecast",
    "statsforecast_auto",
    "tslearn_dtw",
    "tslearn_kmeans",
]
