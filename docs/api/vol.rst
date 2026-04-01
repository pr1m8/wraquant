Volatility Modeling (``wraquant.vol``)
======================================

The volatility module provides 28+ functions for measuring, modeling, and
forecasting financial volatility -- the most critical input to risk
management, option pricing, and portfolio construction.

**Four families of volatility tools:**

1. **Realized volatility estimators** -- non-parametric measures from OHLC
   data (close-to-close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)
2. **GARCH family** -- parametric conditional volatility models for forecasting
   (GARCH, EGARCH, GJR, FIGARCH, HARCH, APARCH, DCC)
3. **Stochastic and self-exciting vol** -- latent process models (stochastic vol,
   Hawkes processes, Gaussian mixture vol)
4. **Implied vol and VRP** -- market-derived measures (SVI surface, variance
   risk premium)

Quick Example
-------------

.. code-block:: python

   from wraquant.vol import yang_zhang, garch_fit, egarch_fit, garch_forecast

   # Best general-purpose realized vol from OHLC data
   rv = yang_zhang(open, high, low, close, window=21)
   print(f"Current realized vol: {rv.iloc[-1]:.4f}")

   # Fit GARCH(1,1) for conditional volatility
   result = garch_fit(returns)
   print(f"alpha={result['alpha']:.4f}, beta={result['beta']:.4f}")
   print(f"Persistence: {result['alpha'] + result['beta']:.4f}")

   # EGARCH for leverage effect
   egarch = egarch_fit(returns)
   print(f"Leverage parameter: {egarch['gamma']:.4f}")

   # 10-day ahead forecast
   forecast = garch_forecast(result, horizon=10)
   print(f"1-day vol: {forecast['forecasts'][0]:.4f}")

Model Selection
^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.vol import garch_model_selection

   # Automatic selection across GARCH variants
   selection = garch_model_selection(returns)
   print(f"Best model: {selection['best_model']}")
   for m in selection['rankings']:
       print(f"  {m['name']:<12} BIC={m['bic']:.2f}")

Multivariate Volatility (DCC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.vol import dcc_fit

   dcc = dcc_fit(multi_returns)
   print(f"Time-varying correlation:\n{dcc['correlations'].iloc[-1]}")

.. seealso::

   - :doc:`/tutorials/volatility_modeling` -- Full volatility modeling tutorial
   - :doc:`risk` -- VaR functions that use GARCH vol forecasts
   - :doc:`price` -- Options pricing that requires vol inputs

API Reference
-------------

.. automodule:: wraquant.vol
   :members:
   :undoc-members:
   :show-inheritance:

GARCH Models
~~~~~~~~~~~~

.. automodule:: wraquant.vol.models
   :members:

Realized Volatility
~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.vol.realized
   :members:
