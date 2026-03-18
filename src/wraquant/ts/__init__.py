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
from wraquant.ts.forecasting import (
    arima_diagnostics,
    arima_model_selection,
    auto_arima,
    auto_forecast,
    ensemble_forecast,
    exponential_smoothing,
    forecast_evaluation,
    holt_winters_forecast,
    rolling_forecast,
    ses_forecast,
    theta_forecast,
)
from wraquant.ts.seasonality import detect_seasonality, fourier_features
from wraquant.ts.stationarity import detrend, difference, fractional_difference
from wraquant.ts.stochastic import (
    jump_diffusion_forecast,
    ornstein_uhlenbeck_forecast,
    regime_switching_forecast,
    var_forecast,
)

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
    "arima_diagnostics",
    "arima_model_selection",
    "auto_forecast",
    "theta_forecast",
    "ses_forecast",
    "holt_winters_forecast",
    "ensemble_forecast",
    "forecast_evaluation",
    "rolling_forecast",
    # stochastic process forecasting
    "ornstein_uhlenbeck_forecast",
    "jump_diffusion_forecast",
    "regime_switching_forecast",
    "var_forecast",
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
