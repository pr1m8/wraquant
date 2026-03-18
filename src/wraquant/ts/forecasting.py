"""Time series forecasting wrappers.

Provides classical univariate forecasting models (ARIMA, exponential
smoothing), automated model selection, and residual diagnostics.

The two primary approaches are:

1. **Exponential Smoothing (ETS / Holt-Winters)** -- weighted averages
   of past observations with exponentially decaying weights. Best for
   series with clear trend and/or seasonal patterns and relatively
   stable structure. Fast to fit, interpretable, and often competitive
   with more complex models for short-horizon forecasts.

2. **ARIMA / auto_arima** -- Box-Jenkins methodology that models the
   series as a linear combination of its own lags (AR), differences
   (I), and past forecast errors (MA). More flexible than ETS for
   capturing complex autocorrelation patterns, but requires careful
   order selection (handled automatically by ``auto_arima``).

When to use each:
    - **ETS**: strong seasonality, smooth trend, short horizon (1-12
      steps). Typical for retail sales, energy demand, macro indicators.
    - **ARIMA**: irregular patterns, no obvious seasonality, or when
      you need confidence intervals that account for parameter
      uncertainty. ``auto_arima`` handles order selection via AIC
      minimisation.
    - **Neither**: for high-frequency financial returns (near-random-
      walk), consider GARCH (``wraquant.vol``) for volatility
      forecasting or ML models (``wraquant.ml``).

References:
    - Hyndman & Athanasopoulos (2021), "Forecasting: Principles and
      Practice" (3rd ed.)
    - Box, Jenkins & Reinsel (2015), "Time Series Analysis"
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.stattools import acf, pacf

from wraquant.core.decorators import requires_extra


def exponential_smoothing(data: pd.Series, **kwargs: Any) -> Any:
    """Fit a Holt-Winters exponential smoothing model.

    Holt-Winters decomposes the series into level, trend, and seasonal
    components, updating each with exponentially decaying weights. It
    is one of the most widely used forecasting methods for business
    and macro time series.

    When to use:
        Use exponential smoothing when:
        - The series has a clear trend and/or seasonal pattern.
        - You need fast, interpretable forecasts.
        - The forecast horizon is short (1-2 seasonal cycles).
        Prefer ARIMA (``auto_arima``) when the autocorrelation structure
        is more complex or when you need differencing to achieve
        stationarity. Prefer ML models (``wraquant.ml``) for nonlinear
        patterns.

    Mathematical formulation:
        Simple:  l_t = alpha * y_t + (1 - alpha) * l_{t-1}
        Double:  adds trend b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
        Triple:  adds seasonal s_t (additive or multiplicative)

    How to interpret:
        Call ``.forecast(h)`` on the returned result for h-step ahead
        forecasts. Use ``.summary()`` to inspect smoothing parameters
        (alpha, beta, gamma). Low alpha => model relies on history;
        high alpha => model follows recent data closely.

    Parameters:
        data: Time series to fit. Should have a ``DatetimeIndex`` with
            a set frequency for seasonal models.
        **kwargs: Keyword arguments forwarded to
            ``statsmodels.tsa.holtwinters.ExponentialSmoothing``.
            Key arguments: ``trend`` (``"add"`` or ``"mul"``),
            ``seasonal`` (``"add"`` or ``"mul"``),
            ``seasonal_periods`` (int).

    Returns:
        Fitted ``HoltWintersResultsWrapper``. Call ``.forecast(h)``
        for predictions.

    Example:
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2020-01-01", periods=120, freq="MS")
        >>> data = pd.Series(np.arange(120.0) + np.random.randn(120) * 5, index=idx)
        >>> result = exponential_smoothing(data, trend="add")
        >>> forecast = result.forecast(12)
        >>> len(forecast)
        12

    See Also:
        auto_arima: Automatic ARIMA model selection.
        wraquant.ts.decomposition.stl_decompose: Decompose before
            forecasting.

    References:
        - Holt (1957), "Forecasting seasonals and trends by
          exponentially weighted moving averages"
        - Winters (1960), "Forecasting Sales by Exponentially Weighted
          Moving Averages"
    """
    model = ExponentialSmoothing(data, **kwargs)
    return model.fit()


@requires_extra("timeseries")
def auto_arima(data: pd.Series, **kwargs: Any) -> Any:
    """Automatically select and fit the best ARIMA model.

    Performs a stepwise or grid search over ARIMA(p,d,q) orders,
    selecting the model that minimises AIC (by default). This
    automates the Box-Jenkins methodology: differencing to achieve
    stationarity, then selecting AR and MA orders.

    When to use:
        Use ``auto_arima`` when:
        - You want ARIMA but are unsure of the correct order.
        - The series has complex autocorrelation (beyond what ETS
          handles).
        - You need forecast confidence intervals.
        Prefer ``exponential_smoothing`` for simple trend/seasonal
        series where speed matters. Prefer ``arima_model_selection``
        when you want to compare all candidate orders explicitly.

    How to interpret:
        The returned model object has:
        - ``.predict(n_periods)`` for point forecasts.
        - ``.predict_in_sample()`` for fitted values.
        - ``.summary()`` for model diagnostics (coefficients,
          AIC/BIC, residual tests).
        - ``.order()`` for the selected (p,d,q) order.
        Use ``arima_diagnostics`` to validate the residuals.

    Parameters:
        data: Time series to fit. Should have >50 observations for
            reliable order selection.
        **kwargs: Keyword arguments forwarded to
            ``pmdarima.auto_arima``. Key arguments include
            ``seasonal`` (bool), ``m`` (seasonal period),
            ``stepwise`` (bool, default True for speed),
            ``information_criterion`` (``"aic"`` or ``"bic"``).

    Returns:
        Fitted ARIMA model from pmdarima with ``.predict()``,
        ``.summary()``, and ``.order()`` methods.

    Raises:
        ImportError: If the ``timeseries`` optional dependency group
            (``pmdarima``) is not installed.

    Example:
        >>> import pandas as pd, numpy as np
        >>> data = pd.Series(np.cumsum(np.random.randn(200)))
        >>> model = auto_arima(data)  # doctest: +SKIP
        >>> model.order()  # doctest: +SKIP
        (1, 1, 0)

    See Also:
        exponential_smoothing: Holt-Winters ETS models.
        arima_diagnostics: Residual checks for fitted ARIMA models.
        arima_model_selection: Manual grid search over ARIMA orders.

    References:
        - Hyndman & Khandakar (2008), "Automatic Time Series
          Forecasting: the forecast Package for R"
    """
    import pmdarima as pm

    return pm.auto_arima(data, **kwargs)


def arima_diagnostics(
    model_result: Any,
    nlags: int = 10,
    alpha: float = 0.05,
) -> dict:
    """Run comprehensive residual diagnostics on a fitted ARIMA model.

    After fitting an ARIMA model, it is critical to check that the
    residuals behave like white noise.  This function runs:

    - **Ljung-Box test**: no remaining autocorrelation in residuals.
    - **Jarque-Bera test**: normality of residuals.
    - **ARCH-LM test**: no remaining heteroskedasticity (ARCH effects).
    - **Durbin-Watson statistic**: first-order autocorrelation check.
    - **ACF/PACF values**: for visual inspection of residual structure.

    Parameters:
        model_result: Fitted model result from ``statsmodels`` ARIMA/SARIMAX
            (must have a ``.resid`` attribute) or ``pmdarima`` ARIMA
            (must have a ``.resid()`` method).
        nlags: Number of lags for autocorrelation tests (default 10).
        alpha: Significance level for pass/fail decisions (default 0.05).

    Returns:
        Dictionary with:
        - ``ljung_box``: dict with ``statistic``, ``p_value``, ``pass``.
        - ``jarque_bera``: dict with ``statistic``, ``p_value``, ``pass``.
        - ``arch_lm``: dict with ``statistic``, ``p_value``, ``pass``.
        - ``durbin_watson``: Durbin-Watson statistic (near 2 = no autocorrelation).
        - ``acf_values``: autocorrelation function values.
        - ``pacf_values``: partial autocorrelation function values.
        - ``model_adequate``: ``True`` if all diagnostic tests pass.

    Example:
        >>> from statsmodels.tsa.arima.model import ARIMA
        >>> import pandas as pd, numpy as np
        >>> data = pd.Series(np.cumsum(np.random.randn(200)))
        >>> fit = ARIMA(data, order=(1, 1, 1)).fit()
        >>> arima_diagnostics(fit)  # doctest: +SKIP
    """
    # Extract residuals (handle both statsmodels and pmdarima)
    if hasattr(model_result, "resid") and callable(model_result.resid):
        resid = model_result.resid()
    elif hasattr(model_result, "resid"):
        resid = model_result.resid
    else:
        msg = "Model result must have a 'resid' attribute or method."
        raise AttributeError(msg)

    resid = np.asarray(resid, dtype=float)
    resid = resid[~np.isnan(resid)]

    # Ljung-Box test for autocorrelation
    lb_result = acorr_ljungbox(resid, lags=nlags, return_df=True)
    lb_last = lb_result.iloc[-1]
    lb_stat = float(lb_last["lb_stat"])
    lb_pval = float(lb_last["lb_pvalue"])
    lb_pass = bool(lb_pval > alpha)

    # Jarque-Bera normality test
    jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(resid)
    jb_pass = bool(jb_pval > alpha)

    # ARCH-LM test for heteroskedasticity
    arch_stat, arch_pval, _, _ = het_arch(resid, nlags=nlags)
    arch_pass = bool(arch_pval > alpha)

    # Durbin-Watson statistic
    dw = float(durbin_watson(resid))

    # ACF / PACF values
    max_acf_lags = min(nlags, len(resid) // 2 - 1)
    if max_acf_lags < 1:
        max_acf_lags = 1
    acf_vals = acf(resid, nlags=max_acf_lags, fft=True)
    pacf_vals = pacf(resid, nlags=max_acf_lags)

    model_adequate = lb_pass and jb_pass and arch_pass

    return {
        "ljung_box": {
            "statistic": lb_stat,
            "p_value": lb_pval,
            "pass": lb_pass,
        },
        "jarque_bera": {
            "statistic": float(jb_stat),
            "p_value": float(jb_pval),
            "pass": jb_pass,
        },
        "arch_lm": {
            "statistic": float(arch_stat),
            "p_value": float(arch_pval),
            "pass": arch_pass,
        },
        "durbin_watson": dw,
        "acf_values": acf_vals,
        "pacf_values": pacf_vals,
        "model_adequate": model_adequate,
    }


def arima_model_selection(
    data: pd.Series,
    p_range: range | list[int] = range(0, 4),
    d_range: range | list[int] = range(0, 3),
    q_range: range | list[int] = range(0, 4),
    criterion: str = "aic",
) -> pd.DataFrame:
    """Compare ARIMA(p,d,q) combinations and rank by information criterion.

    Performs a grid search over combinations of ARIMA orders and ranks
    models by AIC or BIC.  Use this to systematically find the best
    ARIMA specification for a time series.

    Parameters:
        data: Time series to model.
        p_range: Range of AR orders to test (default 0-3).
        d_range: Range of differencing orders to test (default 0-2).
        q_range: Range of MA orders to test (default 0-3).
        criterion: Ranking criterion, ``"aic"`` (default) or ``"bic"``.

    Returns:
        DataFrame with columns: ``order`` (tuple), ``aic``, ``bic``,
        ``converged``, sorted by the chosen criterion (ascending).

    Example:
        >>> import pandas as pd, numpy as np
        >>> data = pd.Series(np.cumsum(np.random.randn(200)))
        >>> arima_model_selection(data, p_range=range(0, 3), d_range=range(0, 2), q_range=range(0, 3))  # doctest: +SKIP
    """
    from statsmodels.tsa.arima.model import ARIMA

    rows: list[dict] = []

    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(data, order=(p, d, q))
                fit = model.fit()
            rows.append(
                {
                    "order": (p, d, q),
                    "aic": float(fit.aic),
                    "bic": float(fit.bic),
                    "converged": bool(fit.mle_retvals.get("converged", True)),
                }
            )
        except Exception:  # noqa: BLE001
            rows.append(
                {
                    "order": (p, d, q),
                    "aic": float("inf"),
                    "bic": float("inf"),
                    "converged": False,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(criterion).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _detect_seasonal_period(data: pd.Series) -> int:
    """Auto-detect seasonal period from a time series index or ACF.

    Uses the index frequency if available, otherwise picks the first
    significant ACF peak beyond lag 1.

    Returns:
        Detected seasonal period (minimum 2).
    """
    freq_map: dict[str, int] = {
        "B": 5,
        "D": 7,
        "W": 52,
        "M": 12,
        "MS": 12,
        "Q": 4,
        "QS": 4,
        "Y": 1,
        "YS": 1,
        "h": 24,
        "H": 24,
        "min": 60,
        "T": 60,
    }
    if hasattr(data.index, "freqstr") and data.index.freqstr is not None:
        base = data.index.freqstr.split("-")[0]
        if base in freq_map:
            return freq_map[base]

    # Fall back to ACF peak detection
    n = len(data)
    max_lag = min(n // 2 - 1, 120)
    if max_lag < 3:
        return 2
    acf_vals = acf(data.values, nlags=max_lag, fft=True)
    # Skip lag 0 and 1, find first local max
    for i in range(2, len(acf_vals) - 1):
        if acf_vals[i] > acf_vals[i - 1] and acf_vals[i] > acf_vals[i + 1]:
            return max(i, 2)
    return 2


def _naive_seasonal_forecast(
    data: pd.Series, h: int, seasonal_period: int
) -> np.ndarray:
    """Seasonal naive forecast: repeat last full season."""
    vals = data.values
    forecasts = np.empty(h)
    for i in range(h):
        idx = len(vals) - seasonal_period + (i % seasonal_period)
        if idx < 0:
            idx = len(vals) - 1
        forecasts[i] = vals[idx]
    return forecasts


def _drift_forecast(data: pd.Series, h: int) -> np.ndarray:
    """Random walk with drift forecast."""
    vals = data.values
    n = len(vals)
    drift = (vals[-1] - vals[0]) / (n - 1) if n > 1 else 0.0
    return np.array([vals[-1] + drift * (i + 1) for i in range(h)])


def _make_forecast_index(data: pd.Series, h: int) -> pd.Index:
    """Create a forecast index extending the data's index by *h* periods."""
    if isinstance(data.index, pd.DatetimeIndex) and data.index.freq is not None:
        return pd.date_range(
            start=data.index[-1] + data.index.freq,
            periods=h,
            freq=data.index.freq,
        )
    if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
        freq = pd.infer_freq(data.index)
        if freq is not None:
            return pd.date_range(
                start=data.index[-1] + pd.tseries.frequencies.to_offset(freq),
                periods=h,
                freq=freq,
            )
    # fallback: integer index
    last = data.index[-1]
    if isinstance(last, (int, np.integer)):
        return pd.RangeIndex(start=last + 1, stop=last + 1 + h)
    return pd.RangeIndex(h)


def _safe_hw(data: pd.Series, h: int, seasonal_period: int) -> np.ndarray:
    """Holt-Winters wrapper that falls back to NaN on failure."""
    try:
        res = holt_winters_forecast(data, h=h, seasonal_periods=seasonal_period)
        return res["forecast"].values
    except Exception:  # noqa: BLE001
        return np.full(h, np.nan)


# ---------------------------------------------------------------------------
# theta_forecast
# ---------------------------------------------------------------------------


def theta_forecast(
    data: pd.Series,
    h: int = 10,
    theta: float = 2.0,
) -> dict:
    """Theta method forecast (Assimakopoulos & Nikolopoulos, 2000).

    The Theta method decomposes the series into two "theta lines":

    - **Theta-0**: the linear regression line (long-run trend).
    - **Theta-2**: an amplified version of the original curvature,
      forecast with Simple Exponential Smoothing (SES).

    The final forecast is the average of these two components.  Despite
    its simplicity, Theta consistently ranks among the best methods in
    forecasting competitions (M3, M4).

    When to use:
        - Quick, competitive baseline for non-seasonal or deseasonalised
          series.
        - When you need a robust method that does not require order
          selection (unlike ARIMA).

    Math:
        Given a series *y_t*, define the second differences
        ``D^2(y_t) = y_t - 2*y_{t-1} + y_{t-2}``.  The theta line
        ``Z_theta(t)`` satisfies ``D^2(Z) = theta * D^2(y)``.  For
        ``theta=0`` the solution is a straight line; for ``theta=2``
        the curvature is doubled.

    Parameters:
        data: Time series (at least 6 observations).
        h: Forecast horizon (default 10).
        theta: Theta parameter controlling curvature amplification
            (default 2.0).

    Returns:
        Dictionary with:
        - ``forecast``: pd.Series of point forecasts.
        - ``fitted_values``: pd.Series of in-sample fitted values.
        - ``theta_params``: dict with ``theta``, ``ses_alpha``,
          ``drift``.

    References:
        Assimakopoulos, V., & Nikolopoulos, K. (2000). The Theta
        model: a decomposition approach to forecasting.
        *International Journal of Forecasting*, 16(4), 521-530.

    Example:
        >>> import pandas as pd, numpy as np
        >>> data = pd.Series(np.cumsum(np.random.randn(100)) + 50)
        >>> result = theta_forecast(data, h=12)
        >>> result['forecast'].shape
        (12,)
    """
    y = data.values.astype(float)
    n = len(y)
    t = np.arange(1, n + 1, dtype=float)

    # Theta-0: linear regression line
    slope, intercept = np.polyfit(t, y, 1)
    theta0_in = intercept + slope * t
    theta0_fcast = np.array(
        [intercept + slope * (n + i) for i in range(1, h + 1)]
    )

    # Theta-2: amplified curvature via SES
    theta2_in = theta * y - (theta - 1) * theta0_in
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ses_model = SimpleExpSmoothing(theta2_in).fit(optimized=True)
    ses_alpha = float(ses_model.params.get("smoothing_level", 0.5))
    theta2_fcast = ses_model.forecast(h)

    # Average the two theta lines
    fcast_vals = 0.5 * (theta0_fcast + theta2_fcast)
    fitted_vals = 0.5 * (theta0_in + ses_model.fittedvalues)

    idx = _make_forecast_index(data, h)
    return {
        "forecast": pd.Series(fcast_vals, index=idx, name="theta_forecast"),
        "fitted_values": pd.Series(
            fitted_vals, index=data.index, name="theta_fitted"
        ),
        "theta_params": {
            "theta": theta,
            "ses_alpha": ses_alpha,
            "drift": float(slope),
        },
    }


# ---------------------------------------------------------------------------
# ses_forecast
# ---------------------------------------------------------------------------


def ses_forecast(
    data: pd.Series,
    h: int = 10,
    alpha: float | None = None,
) -> dict:
    """Simple Exponential Smoothing (SES) with optimal alpha.

    SES is the simplest exponential smoothing method.  It assigns
    exponentially decreasing weights to past observations.  The single
    smoothing parameter ``alpha`` controls how fast old observations
    are discounted:

    - ``alpha`` near 1 => recent data dominates (series is volatile).
    - ``alpha`` near 0 => old data persists (series is stable).

    When to use:
        - Stationary series with no clear trend or seasonality.
        - Quick baseline forecast.
        - As a building block inside ensemble or Theta methods.

    Math:
        ``l_t = alpha * y_t + (1 - alpha) * l_{t-1}``

        Forecast: ``y_{t+h|t} = l_t`` (flat forecast for all horizons).

    Parameters:
        data: Time series to forecast.
        h: Forecast horizon (default 10).
        alpha: Smoothing parameter in (0, 1).  If ``None`` (default),
            the optimal alpha is estimated by minimising SSE.

    Returns:
        Dictionary with:
        - ``forecast``: pd.Series of point forecasts.
        - ``alpha``: optimised smoothing parameter.
        - ``fitted_values``: pd.Series of in-sample fitted values.
        - ``residuals``: pd.Series of in-sample residuals.

    Example:
        >>> import pandas as pd, numpy as np
        >>> data = pd.Series(np.random.randn(100).cumsum() + 50)
        >>> result = ses_forecast(data, h=5)
        >>> len(result['forecast'])
        5
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if alpha is not None:
            fit = SimpleExpSmoothing(data).fit(
                smoothing_level=alpha, optimized=False
            )
        else:
            fit = SimpleExpSmoothing(data).fit(optimized=True)

    opt_alpha = float(fit.params.get("smoothing_level", alpha or 0.5))
    fcast = fit.forecast(h)
    idx = _make_forecast_index(data, h)

    return {
        "forecast": pd.Series(
            fcast.values, index=idx, name="ses_forecast"
        ),
        "alpha": opt_alpha,
        "fitted_values": pd.Series(
            fit.fittedvalues.values, index=data.index, name="ses_fitted"
        ),
        "residuals": pd.Series(
            (data.values - fit.fittedvalues.values),
            index=data.index,
            name="ses_residuals",
        ),
    }


# ---------------------------------------------------------------------------
# holt_winters_forecast
# ---------------------------------------------------------------------------


def holt_winters_forecast(
    data: pd.Series,
    h: int = 10,
    seasonal: str = "add",
    seasonal_periods: int | None = None,
    trend: str | None = "add",
) -> dict:
    """Holt-Winters exponential smoothing with trend and seasonality.

    This is the full triple exponential smoothing method.  It extends
    SES by adding a trend component (Holt) and a seasonal component
    (Winters), supporting both additive and multiplicative seasonality.

    When to use:
        - Series with clear trend **and** seasonality (e.g., monthly
          sales, quarterly GDP).
        - Choose ``seasonal='mul'`` when seasonal fluctuations grow
          with the level of the series.

    Math (additive):
        ``l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})``
        ``b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}``
        ``s_t = gamma * (y_t - l_t) + (1 - gamma) * s_{t-m}``

    Parameters:
        data: Time series (must have at least 2 full seasonal cycles).
        h: Forecast horizon (default 10).
        seasonal: ``"add"`` (default) or ``"mul"`` for additive or
            multiplicative seasonality.
        seasonal_periods: Number of periods in a season. Auto-detected
            from the index frequency if ``None``.
        trend: Trend component type: ``"add"`` (default), ``"mul"``,
            or ``None`` for no trend.

    Returns:
        Dictionary with:
        - ``forecast``: pd.Series of point forecasts.
        - ``params``: dict with ``alpha``, ``beta``, ``gamma``.
        - ``fitted_values``: pd.Series of in-sample fitted values.
        - ``seasonal_components``: pd.Series of estimated seasonal
          factors.

    Example:
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range('2020-01-01', periods=120, freq='MS')
        >>> seasonal = 10 * np.sin(2 * np.pi * np.arange(120) / 12)
        >>> data = pd.Series(
        ...     np.arange(120) * 0.5 + seasonal + np.random.randn(120),
        ...     index=idx,
        ... )
        >>> result = holt_winters_forecast(data, h=12, seasonal_periods=12)
        >>> len(result['forecast'])
        12
    """
    if seasonal_periods is None:
        seasonal_periods = _detect_seasonal_period(data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        fit = model.fit(optimized=True)

    fcast = fit.forecast(h)
    idx = _make_forecast_index(data, h)

    params = {
        "alpha": float(fit.params.get("smoothing_level", 0.0)),
        "beta": float(fit.params.get("smoothing_trend", 0.0)),
        "gamma": float(fit.params.get("smoothing_seasonal", 0.0)),
    }

    # Extract seasonal component
    season_vals = fit.season if hasattr(fit, "season") else np.zeros(len(data))
    if season_vals is None:
        season_vals = np.zeros(len(data))

    return {
        "forecast": pd.Series(
            fcast.values, index=idx, name="hw_forecast"
        ),
        "params": params,
        "fitted_values": pd.Series(
            fit.fittedvalues.values, index=data.index, name="hw_fitted"
        ),
        "seasonal_components": pd.Series(
            np.asarray(season_vals).ravel()[: len(data)],
            index=data.index,
            name="hw_seasonal",
        ),
    }


# ---------------------------------------------------------------------------
# forecast_evaluation
# ---------------------------------------------------------------------------


def forecast_evaluation(
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
    naive_forecast: pd.Series | np.ndarray | None = None,
    benchmark_forecast: pd.Series | np.ndarray | None = None,
) -> dict:
    """Comprehensive forecast accuracy metrics.

    Computes a wide set of point-forecast accuracy measures used in
    the forecasting literature.  When a *benchmark_forecast* is
    provided the Diebold-Mariano test is run to determine whether the
    two forecasts have significantly different accuracy.

    Metrics included:
        - **RMSE** (Root Mean Squared Error): penalises large errors.
        - **MAE** (Mean Absolute Error): robust to outliers.
        - **MAPE** (Mean Absolute Percentage Error): scale-independent
          but undefined when actuals are zero.
        - **SMAPE** (Symmetric MAPE): bounded version of MAPE.
        - **MdAPE** (Median Absolute Percentage Error): robust MAPE.
        - **MASE** (Mean Absolute Scaled Error): scaled by in-sample
          naive error; needs *naive_forecast* or falls back to
          ``MAE / mean(|diff(actual)|)``.
        - **Directional accuracy**: fraction of periods where the
          forecast correctly predicts the sign of the change.
        - **Diebold-Mariano test**: compares this forecast against
          *benchmark_forecast* (two-sided, squared-error loss).

    Parameters:
        actual: Observed values.
        forecast: Predicted values (same length as *actual*).
        naive_forecast: Optional naive forecast for MASE scaling.
            If ``None``, MASE is computed using the mean absolute
            first difference of *actual*.
        benchmark_forecast: Optional second forecast for
            Diebold-Mariano comparison.

    Returns:
        Dictionary with:
        - ``rmse``, ``mae``, ``mape``, ``smape``, ``mdape``, ``mase``,
          ``directional_accuracy``.
        - ``diebold_mariano``: dict with ``statistic`` and ``p_value``
          (only present when *benchmark_forecast* is given).

    References:
        Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive
        Accuracy. *Journal of Business & Economic Statistics*, 13(3).

    Example:
        >>> import numpy as np
        >>> actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> pred   = np.array([1.1, 2.2, 2.8, 4.3, 4.7])
        >>> metrics = forecast_evaluation(actual, pred)
        >>> 'rmse' in metrics
        True
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    e = a - f

    # RMSE
    rmse = float(np.sqrt(np.mean(e**2)))

    # MAE
    mae = float(np.mean(np.abs(e)))

    # MAPE (exclude zeros in actual)
    nonzero = a != 0
    if np.any(nonzero):
        mape = float(np.mean(np.abs(e[nonzero] / a[nonzero])) * 100)
    else:
        mape = float("inf")

    # SMAPE
    denom = np.abs(a) + np.abs(f)
    nonzero_denom = denom != 0
    if np.any(nonzero_denom):
        smape = float(
            np.mean(
                2.0 * np.abs(e[nonzero_denom]) / denom[nonzero_denom]
            )
            * 100
        )
    else:
        smape = 0.0

    # MdAPE
    if np.any(nonzero):
        mdape = float(
            np.median(np.abs(e[nonzero] / a[nonzero])) * 100
        )
    else:
        mdape = float("inf")

    # MASE
    if naive_forecast is not None:
        naive_err = np.mean(
            np.abs(a - np.asarray(naive_forecast, dtype=float))
        )
    else:
        naive_err = (
            np.mean(np.abs(np.diff(a))) if len(a) > 1 else 1.0
        )
    mase = float(mae / naive_err) if naive_err > 0 else float("inf")

    # Directional accuracy
    if len(a) > 1:
        actual_dir = np.sign(np.diff(a))
        fcast_dir = np.sign(np.diff(f))
        dir_acc = float(np.mean(actual_dir == fcast_dir) * 100)
    else:
        dir_acc = float("nan")

    result: dict[str, Any] = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "mdape": mdape,
        "mase": mase,
        "directional_accuracy": dir_acc,
    }

    # Diebold-Mariano test
    if benchmark_forecast is not None:
        b = np.asarray(benchmark_forecast, dtype=float)
        e_bench = a - b
        d = e**2 - e_bench**2
        d_mean = np.mean(d)
        n_dm = len(d)
        # Newey-West style variance (lag 0 only for simplicity)
        d_var = np.var(d, ddof=1) / n_dm if n_dm > 1 else 1.0
        dm_stat = (
            float(d_mean / np.sqrt(d_var)) if d_var > 0 else 0.0
        )
        dm_pval = float(2 * (1 - sp_stats.norm.cdf(abs(dm_stat))))
        result["diebold_mariano"] = {
            "statistic": dm_stat,
            "p_value": dm_pval,
        }

    return result


# ---------------------------------------------------------------------------
# auto_forecast
# ---------------------------------------------------------------------------


def auto_forecast(
    data: pd.Series,
    h: int = 10,
    holdout_pct: float = 0.2,
    seasonal_period: int | None = None,
) -> dict:
    """Unified auto-forecasting: try multiple models, pick the best.

    Fits several forecasting methods on a training subset, evaluates
    them on a held-out test set, and returns the forecast from the
    best model re-fitted on the full data.  This is the recommended
    starting point when you have no prior knowledge about the series
    dynamics.

    Models tried:
        - **SES**: Simple Exponential Smoothing.
        - **Holt-Winters (additive)**: trend + additive seasonality.
        - **Theta**: Theta method.
        - **Seasonal naive**: repeat the last seasonal cycle.
        - **Drift**: random walk with drift.

    When to use auto_forecast vs manual model selection:
        Use ``auto_forecast`` for rapid prototyping, exploratory work,
        or when you need a robust default.  Switch to manual model
        selection (``arima_model_selection``, ``auto_arima``) when you
        need fine-grained control, interpretability, or domain-specific
        constraints.

    Parameters:
        data: Time series (at least 30 observations recommended).
        h: Forecast horizon (default 10).
        holdout_pct: Fraction of data reserved for model evaluation
            (default 0.20).
        seasonal_period: Seasonal period for Holt-Winters and seasonal
            naive.  Auto-detected if ``None``.

    Returns:
        Dictionary with:
        - ``best_model``: name of the winning model.
        - ``forecast``: pd.Series of point forecasts from the best
          model (re-fitted on full data).
        - ``confidence_intervals``: dict with ``lower`` and ``upper``
          pd.Series (approximate, based on residual std).
        - ``model_comparison``: pd.DataFrame with RMSE, MAE, MAPE per
          model.
        - ``residual_diagnostics``: dict with ``mean``, ``std``,
          ``ljung_box_pvalue`` of residuals from the best model.

    Example:
        >>> import pandas as pd, numpy as np
        >>> data = pd.Series(np.cumsum(np.random.randn(200)) + 100)
        >>> result = auto_forecast(data, h=10)
        >>> result['best_model'] in [
        ...     'ses', 'holt_winters', 'theta', 'seasonal_naive', 'drift',
        ... ]
        True
    """
    n = len(data)
    split = int(n * (1 - holdout_pct))
    train, test = data.iloc[:split], data.iloc[split:]
    test_h = len(test)

    if seasonal_period is None:
        seasonal_period = _detect_seasonal_period(data)

    # ------------------------------------------------------------------
    # Candidate model functions
    # ------------------------------------------------------------------
    def _try_ses(tr: pd.Series, fh: int) -> np.ndarray:
        return ses_forecast(tr, h=fh)["forecast"].values

    def _try_hw(tr: pd.Series, fh: int) -> np.ndarray:
        try:
            return holt_winters_forecast(
                tr, h=fh, seasonal_periods=seasonal_period
            )["forecast"].values
        except Exception:  # noqa: BLE001
            return np.full(fh, np.nan)

    def _try_theta(tr: pd.Series, fh: int) -> np.ndarray:
        return theta_forecast(tr, h=fh)["forecast"].values

    def _try_snaive(tr: pd.Series, fh: int) -> np.ndarray:
        return _naive_seasonal_forecast(tr, fh, seasonal_period)

    candidates: dict[str, Callable] = {
        "ses": _try_ses,
        "holt_winters": _try_hw,
        "theta": _try_theta,
        "seasonal_naive": _try_snaive,
        "drift": _drift_forecast,
    }

    # Evaluate on hold-out
    rows: list[dict] = []
    for name, fn in candidates.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = fn(train, test_h)
        if np.any(np.isnan(pred)):
            rows.append(
                {
                    "model": name,
                    "rmse": float("inf"),
                    "mae": float("inf"),
                    "mape": float("inf"),
                }
            )
            continue
        metrics = forecast_evaluation(test.values, pred)
        rows.append(
            {
                "model": name,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
            }
        )

    comparison = pd.DataFrame(rows).sort_values("rmse").reset_index(
        drop=True
    )
    best_name = str(comparison.iloc[0]["model"])

    # Re-fit best model on full data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_fcast = candidates[best_name](data, h)
    idx = _make_forecast_index(data, h)
    fcast_series = pd.Series(
        best_fcast, index=idx, name="auto_forecast"
    )

    # Approximate confidence intervals from first-difference std
    resid_std = float(np.std(data.values[1:] - data.values[:-1]))
    z = 1.96
    ci_lower = pd.Series(
        best_fcast - z * resid_std * np.sqrt(np.arange(1, h + 1)),
        index=idx,
        name="lower",
    )
    ci_upper = pd.Series(
        best_fcast + z * resid_std * np.sqrt(np.arange(1, h + 1)),
        index=idx,
        name="upper",
    )

    # Residual diagnostics for best model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        in_sample_pred = candidates[best_name](data, len(data))
    residuals = data.values - in_sample_pred[: len(data)]
    residuals = residuals[~np.isnan(residuals)]

    lb_pval = float("nan")
    if len(residuals) > 10:
        try:
            lb = acorr_ljungbox(residuals, lags=10, return_df=True)
            lb_pval = float(lb.iloc[-1]["lb_pvalue"])
        except Exception:  # noqa: BLE001
            pass

    return {
        "best_model": best_name,
        "forecast": fcast_series,
        "confidence_intervals": {"lower": ci_lower, "upper": ci_upper},
        "model_comparison": comparison,
        "residual_diagnostics": {
            "mean": (
                float(np.mean(residuals))
                if len(residuals) > 0
                else float("nan")
            ),
            "std": (
                float(np.std(residuals))
                if len(residuals) > 0
                else float("nan")
            ),
            "ljung_box_pvalue": lb_pval,
        },
    }


# ---------------------------------------------------------------------------
# ensemble_forecast
# ---------------------------------------------------------------------------


def ensemble_forecast(
    data: pd.Series,
    h: int = 10,
    methods: Sequence[str] | None = None,
    holdout_pct: float = 0.2,
    seasonal_period: int | None = None,
) -> dict:
    """Combine multiple forecasting models via inverse-RMSE weighting.

    Ensemble forecasting often outperforms any single model because it
    diversifies model risk.  Weights are proportional to the inverse
    RMSE on a held-out validation set, so better-performing models
    contribute more.

    When to use:
        - When no single model clearly dominates.
        - For production pipelines where robustness matters more than
          interpretability.

    Parameters:
        data: Time series to forecast.
        h: Forecast horizon (default 10).
        methods: List of method names to include.  Supported:
            ``"ses"``, ``"theta"``, ``"holt_winters"``,
            ``"drift"``.  If ``None``, uses all four.
        holdout_pct: Fraction of data for validation (default 0.20).
        seasonal_period: Seasonal period for Holt-Winters.
            Auto-detected if ``None``.

    Returns:
        Dictionary with:
        - ``forecast``: pd.Series of weighted-average forecasts.
        - ``weights``: dict mapping method name to its weight.
        - ``individual_forecasts``: dict mapping method name to
          pd.Series forecast.
        - ``rmse_per_model``: dict mapping method name to its
          hold-out RMSE.

    Example:
        >>> import pandas as pd, numpy as np
        >>> data = pd.Series(np.cumsum(np.random.randn(200)) + 100)
        >>> result = ensemble_forecast(data, h=10)
        >>> abs(sum(result['weights'].values()) - 1.0) < 1e-6
        True
    """
    if methods is None:
        methods = ["ses", "theta", "holt_winters", "drift"]
    if seasonal_period is None:
        seasonal_period = _detect_seasonal_period(data)

    n = len(data)
    split = int(n * (1 - holdout_pct))
    train, test = data.iloc[:split], data.iloc[split:]
    test_h = len(test)

    method_fns: dict[str, Callable] = {
        "ses": lambda tr, fh: ses_forecast(tr, h=fh)["forecast"].values,
        "theta": lambda tr, fh: theta_forecast(tr, h=fh)[
            "forecast"
        ].values,
        "holt_winters": lambda tr, fh: _safe_hw(
            tr, fh, seasonal_period
        ),
        "drift": _drift_forecast,
    }

    rmses: dict[str, float] = {}
    for name in methods:
        if name not in method_fns:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pred = method_fns[name](train, test_h)
                if np.any(np.isnan(pred)):
                    rmses[name] = float("inf")
                else:
                    rmses[name] = float(
                        np.sqrt(np.mean((test.values - pred) ** 2))
                    )
            except Exception:  # noqa: BLE001
                rmses[name] = float("inf")

    # Inverse-RMSE weights (exclude inf)
    finite_rmses = {
        k: v for k, v in rmses.items() if np.isfinite(v) and v > 0
    }
    if not finite_rmses:
        finite_rmses = {k: 1.0 for k in rmses}
    inv = {k: 1.0 / v for k, v in finite_rmses.items()}
    total_inv = sum(inv.values())
    weights = {k: v / total_inv for k, v in inv.items()}

    # Generate full-data forecasts
    idx = _make_forecast_index(data, h)
    individual: dict[str, pd.Series] = {}
    combined = np.zeros(h)
    for name, w in weights.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = method_fns[name](data, h)
        s = pd.Series(pred, index=idx, name=name)
        individual[name] = s
        combined += w * pred

    return {
        "forecast": pd.Series(
            combined, index=idx, name="ensemble_forecast"
        ),
        "weights": weights,
        "individual_forecasts": individual,
        "rmse_per_model": rmses,
    }


# ---------------------------------------------------------------------------
# rolling_forecast
# ---------------------------------------------------------------------------


def rolling_forecast(
    data: pd.Series,
    forecast_fn: Callable[[pd.Series, int], np.ndarray | pd.Series],
    h: int = 1,
    initial_window: int | None = None,
    step: int = 1,
) -> dict:
    """Walk-forward (expanding window) out-of-sample forecasting.

    At each step the model is re-fitted on all data up to time *t*,
    then forecasts *h* steps ahead.  The result is a time series of
    true out-of-sample predictions that can be compared against
    actuals.

    This is the gold standard for evaluating forecasting models
    because it mimics how the model would be used in production.

    Parameters:
        data: Full time series.
        forecast_fn: Callable ``(train_series, h) -> array-like``
            that returns *h* point forecasts.
        h: Forecast horizon at each step (default 1).
        initial_window: Minimum training window size.  Defaults to
            ``max(30, len(data) // 3)``.
        step: Number of observations to advance between re-fits
            (default 1).

    Returns:
        Dictionary with:
        - ``forecasts``: pd.Series of out-of-sample predictions.
        - ``actuals``: pd.Series of corresponding actual values.
        - ``errors``: pd.Series of forecast errors.
        - ``cumulative_metrics``: dict with cumulative RMSE, MAE.

    Example:
        >>> import pandas as pd, numpy as np
        >>> data = pd.Series(np.cumsum(np.random.randn(100)) + 50)
        >>> def my_model(train, h):
        ...     return np.full(h, train.iloc[-1])
        >>> result = rolling_forecast(
        ...     data, my_model, h=1, initial_window=50,
        ... )
        >>> len(result['forecasts']) > 0
        True
    """
    n = len(data)
    if initial_window is None:
        initial_window = max(30, n // 3)

    fcast_vals: list[float] = []
    actual_vals: list[float] = []
    fcast_idx: list = []

    t = initial_window
    while t + h <= n:
        train = data.iloc[:t]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pred = forecast_fn(train, h)
                pred = np.asarray(pred).ravel()
                fcast_vals.append(float(pred[-1]))  # h-step-ahead
            except Exception:  # noqa: BLE001
                fcast_vals.append(float("nan"))
        actual_vals.append(float(data.iloc[t + h - 1]))
        fcast_idx.append(data.index[t + h - 1])
        t += step

    forecasts = pd.Series(
        fcast_vals, index=fcast_idx, name="rolling_forecast"
    )
    actuals = pd.Series(
        actual_vals, index=fcast_idx, name="actuals"
    )
    errors = actuals - forecasts

    valid = ~np.isnan(errors.values)
    cum_rmse = (
        float(np.sqrt(np.mean(errors.values[valid] ** 2)))
        if np.any(valid)
        else float("nan")
    )
    cum_mae = (
        float(np.mean(np.abs(errors.values[valid])))
        if np.any(valid)
        else float("nan")
    )

    return {
        "forecasts": forecasts,
        "actuals": actuals,
        "errors": errors,
        "cumulative_metrics": {"rmse": cum_rmse, "mae": cum_mae},
    }
