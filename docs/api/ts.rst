Time Series (``wraquant.ts``)
=============================

Time series analysis and forecasting for financial data: decomposition,
seasonality detection, changepoint detection, stationarity tests and
transformations, feature extraction, anomaly detection, and forecasting.

**Submodules:**

- **Decomposition** -- STL, seasonal, SSA, EMD, wavelets, unobserved components
- **Forecasting** -- auto ARIMA, exponential smoothing, Holt-Winters, theta, ensemble, GARCH residual
- **Stationarity** -- ADF, KPSS, Phillips-Perron, fractional differencing, detrending
- **Seasonality** -- detection, Fourier features, multi-seasonal decomposition
- **Changepoint** -- CUSUM, Bayesian changepoint detection
- **Anomaly** -- isolation forest, Prophet-based, Grubbs test
- **Features** -- autocorrelation, spectral, complexity features
- **Stochastic** -- OU, jump-diffusion, regime-switching, and VAR forecasts
- **Advanced** -- tsfresh, stumpy, wavelets, sktime, statsforecast, tslearn, darts

Quick Example
-------------

.. code-block:: python

   from wraquant.ts import auto_arima, stl_decompose, adf_test

   # Test stationarity
   adf = adf_test(returns)
   print(f"ADF statistic: {adf['statistic']:.4f}")
   print(f"p-value: {adf['p_value']:.4f}")
   print(f"Stationary: {adf['is_stationary']}")

   # STL decomposition (trend + seasonal + residual)
   decomp = stl_decompose(prices, period=252)
   trend = decomp['trend']
   seasonal = decomp['seasonal']
   residual = decomp['residual']

   # Auto ARIMA forecasting
   forecast = auto_arima(returns, h=21)
   print(f"21-day forecast mean: {forecast['mean'].mean():.6f}")
   print(f"AIC: {forecast['aic']:.2f}")
   print(f"Order: {forecast['order']}")

Ensemble Forecasting
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.ts import ensemble_forecast, forecast_evaluation

   # Combine multiple forecasting methods
   ensemble = ensemble_forecast(returns, h=21, methods=["arima", "ets", "theta"])
   print(f"Ensemble forecast: {ensemble['mean']}")

   # Evaluate forecast accuracy with proper metrics
   eval_result = forecast_evaluation(actual, predicted)
   print(f"MAE: {eval_result['mae']:.6f}")
   print(f"RMSE: {eval_result['rmse']:.6f}")
   print(f"MAPE: {eval_result['mape']:.2%}")

Stochastic Process Forecasting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.ts import ornstein_uhlenbeck_forecast, regime_switching_forecast

   # OU process forecast (mean-reverting -- good for spreads)
   ou = ornstein_uhlenbeck_forecast(spread, h=21)
   print(f"OU mean-reversion speed: {ou['theta']:.4f}")
   print(f"Long-run mean: {ou['mu']:.4f}")

   # Regime-switching forecast
   rs = regime_switching_forecast(returns, n_regimes=2, h=21)
   print(f"Forecast (accounting for regime probabilities): {rs['mean']}")

.. seealso::

   - :doc:`stats` -- Statistical tests and distribution fitting
   - :doc:`vol` -- GARCH volatility forecasting
   - :doc:`regimes` -- Changepoint and regime detection

API Reference
-------------

.. automodule:: wraquant.ts
   :members:
   :undoc-members:
   :show-inheritance:

Decomposition
^^^^^^^^^^^^^

.. automodule:: wraquant.ts.decomposition
   :members:

Forecasting
^^^^^^^^^^^

.. automodule:: wraquant.ts.forecasting
   :members:

Changepoint Detection
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.ts.changepoint
   :members:

Stationarity Tests
^^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.ts.stationarity
   :members:

Seasonality
^^^^^^^^^^^

.. automodule:: wraquant.ts.seasonality
   :members:

Feature Extraction
^^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.ts.features
   :members:

Anomaly Detection
^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.ts.anomaly
   :members:

Stochastic Processes
^^^^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.ts.stochastic
   :members:

Advanced Methods
^^^^^^^^^^^^^^^^

.. automodule:: wraquant.ts.advanced
   :members:
