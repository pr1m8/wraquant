"""Time series analysis and forecasting.

Provides a complete pipeline for financial time series modeling: from
decomposing a series into trend, seasonal, and residual components,
through stationarity testing and transformation, to forecasting with
classical, stochastic-process, and machine-learning-backed methods.
Designed for the challenges of financial data -- non-stationarity,
regime changes, fat tails, and low signal-to-noise ratios.

Key sub-modules:

- **Decomposition** (``decomposition``) -- ``seasonal_decompose`` (additive/
  multiplicative), ``stl_decompose`` (STL -- robust to outliers),
  ``ssa_decompose`` (Singular Spectrum Analysis), ``emd_decompose``
  (Empirical Mode Decomposition), ``wavelet_decompose``, and
  ``unobserved_components`` (Harvey structural model).
- **Seasonality** (``seasonality``) -- ``detect_seasonality`` (auto-detect
  period), ``fourier_features`` (generate sin/cos regressors),
  ``multi_fourier_features``, ``seasonal_strength``, and
  ``multi_seasonal_decompose``.
- **Change-point detection** (``changepoint``) -- ``detect_changepoints``
  (Bayesian or PELT) and ``cusum`` for detecting structural breaks in
  mean, variance, or trend.
- **Stationarity** (``stationarity``) -- ``adf_test``, ``kpss_test``,
  ``phillips_perron``, ``variance_ratio_test``, ``difference``,
  ``fractional_difference`` (preserve memory while achieving
  stationarity), ``detrend``, and ``optimal_differencing``.
- **Features** (``features``) -- ``autocorrelation_features``,
  ``spectral_features``, and ``complexity_features`` for summarizing
  time series structure in ML pipelines.
- **Anomaly detection** (``anomaly``) -- ``isolation_forest_ts``,
  ``prophet_anomaly``, and ``grubbs_test_ts``.
- **Forecasting** (``forecasting``) -- ``auto_arima``, ``auto_forecast``
  (automatic model selection), ``exponential_smoothing``,
  ``theta_forecast``, ``ensemble_forecast``, ``rolling_forecast``,
  ``holt_winters_forecast``, and ``forecast_evaluation``.
- **Stochastic processes** (``stochastic``) -- ``ornstein_uhlenbeck_forecast``,
  ``jump_diffusion_forecast``, ``regime_switching_forecast``, and
  ``var_forecast``.
- **Advanced integrations** (``advanced``) -- Wrappers for tsfresh,
  stumpy (matrix profile), tslearn (DTW, k-means), darts, sktime,
  and statsforecast.

Example:
    >>> from wraquant.ts import stl_decompose, auto_forecast, adf_test
    >>> decomp = stl_decompose(prices, period=252)
    >>> stationarity = adf_test(returns)
    >>> forecast = auto_forecast(returns, h=20)

Use ``wraquant.ts`` for time series modeling and forecasting.  For
statistical tests on returns (normality, autocorrelation), see
``wraquant.stats``.  For volatility-specific forecasting (GARCH), see
``wraquant.vol``.  For regime-switching models, see ``wraquant.regimes``.
"""

from wraquant.ts.advanced import (
    darts_forecast,
    sktime_forecast,
    statsforecast_auto,
    stumpy_matrix_profile,
    tsfresh_features,
    tslearn_dtw,
    tslearn_kmeans,
    wavelet_denoise,
    wavelet_transform,
)
from wraquant.ts.anomaly import grubbs_test_ts, isolation_forest_ts, prophet_anomaly
from wraquant.ts.changepoint import cusum, detect_changepoints
from wraquant.ts.decomposition import (
    emd_decompose,
    seasonal_decompose,
    ssa_decompose,
    stl_decompose,
    trend_filter,
    unobserved_components,
    wavelet_decompose,
)
from wraquant.ts.features import (
    autocorrelation_features,
    complexity_features,
    spectral_features,
)
from wraquant.ts.forecasting import (
    arima_diagnostics,
    arima_model_selection,
    auto_arima,
    auto_forecast,
    ensemble_forecast,
    exponential_smoothing,
    forecast_evaluation,
    garch_residual_forecast,
    holt_winters_forecast,
    rolling_forecast,
    ses_forecast,
    theta_forecast,
)
from wraquant.ts.seasonality import (
    detect_seasonality,
    fourier_features,
    multi_fourier_features,
    multi_seasonal_decompose,
    seasonal_strength,
)
from wraquant.ts.stationarity import (
    adf_test,
    detrend,
    difference,
    fractional_difference,
    kpss_test,
    optimal_differencing,
    phillips_perron,
    variance_ratio_test,
)
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
    "ssa_decompose",
    "emd_decompose",
    "wavelet_decompose",
    "unobserved_components",
    # seasonality
    "detect_seasonality",
    "fourier_features",
    "multi_fourier_features",
    "seasonal_strength",
    "multi_seasonal_decompose",
    # changepoint
    "cusum",
    "detect_changepoints",
    # stationarity
    "difference",
    "fractional_difference",
    "detrend",
    "adf_test",
    "kpss_test",
    "phillips_perron",
    "optimal_differencing",
    "variance_ratio_test",
    # features
    "autocorrelation_features",
    "spectral_features",
    "complexity_features",
    # anomaly detection
    "isolation_forest_ts",
    "prophet_anomaly",
    "grubbs_test_ts",
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
    "garch_residual_forecast",
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
    "darts_forecast",
]
