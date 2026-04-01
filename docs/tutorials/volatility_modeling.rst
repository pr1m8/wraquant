Volatility Modeling
===================

This tutorial covers the full volatility modeling workflow: estimating
realized volatility, fitting GARCH models, diagnosing leverage effects,
comparing model variants, forecasting, and running rolling GARCH.

Volatility is the most critical input to risk management, option pricing,
and portfolio construction -- yet it is not directly observable. The methods
here let you measure, model, and forecast it.


Step 1: Realized Volatility Estimators
----------------------------------------

Start with non-parametric realized volatility from observed prices. These
do not assume any model -- they simply estimate vol from the data.

.. code-block:: python

   import wraquant as wq
   import pandas as pd
   from wraquant.vol import (
       realized_volatility, parkinson, garman_klass,
       rogers_satchell, yang_zhang,
   )

   # Load OHLCV data
   ohlcv = pd.read_csv("ohlcv.csv", index_col=0, parse_dates=True)

   # Close-to-close (simplest, but ignores intraday info)
   rv_cc = realized_volatility(ohlcv["Close"], window=21)

   # Parkinson (uses High/Low) -- ~5x more efficient
   rv_park = parkinson(ohlcv["High"], ohlcv["Low"], window=21)

   # Garman-Klass (uses OHLC) -- ~8x more efficient
   rv_gk = garman_klass(
       ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"], window=21
   )

   # Yang-Zhang (best general-purpose for daily OHLC)
   rv_yz = yang_zhang(
       ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"], window=21
   )

   # Compare the estimators
   print(f"Close-to-close: {rv_cc.iloc[-1]:.4f}")
   print(f"Parkinson:      {rv_park.iloc[-1]:.4f}")
   print(f"Garman-Klass:   {rv_gk.iloc[-1]:.4f}")
   print(f"Yang-Zhang:     {rv_yz.iloc[-1]:.4f}")

   # Yang-Zhang is generally preferred because it handles drift
   # (trending markets) and uses all OHLC information.


Step 2: Fit a GARCH(1,1) Model
--------------------------------

GARCH captures volatility clustering -- the empirical fact that large moves
tend to be followed by large moves. The conditional variance evolves as:

.. math::

   \sigma^2_t = \omega + \alpha \cdot \varepsilon^2_{t-1} + \beta \cdot \sigma^2_{t-1}

.. code-block:: python

   from wraquant.vol import garch_fit, volatility_persistence

   daily_returns = ohlcv["Close"].pct_change().dropna()

   result = garch_fit(daily_returns)
   print(f"omega: {result['omega']:.6f}")
   print(f"alpha: {result['alpha']:.4f}")
   print(f"beta:  {result['beta']:.4f}")

   # Persistence = alpha + beta
   pers = volatility_persistence(result)
   print(f"Persistence: {pers:.4f}")
   # Close to 1.0 means volatility shocks decay very slowly.
   # >0.99 suggests an IGARCH (integrated GARCH) process.

   # Log-likelihood and information criteria for model comparison
   print(f"Log-likelihood: {result['log_likelihood']:.2f}")
   print(f"AIC: {result['aic']:.2f}")
   print(f"BIC: {result['bic']:.2f}")


Step 3: Diagnose the Leverage Effect
--------------------------------------

In equity markets, negative returns increase volatility more than positive
returns of the same magnitude. This is called the "leverage effect" (Black,
1976). The news impact curve visualizes this asymmetry.

.. code-block:: python

   from wraquant.vol import egarch_fit, gjr_garch_fit, news_impact_curve

   # EGARCH captures leverage via a log-linear specification
   egarch = egarch_fit(daily_returns)
   print(f"EGARCH leverage (gamma): {egarch['gamma']:.4f}")
   # Negative gamma means negative shocks increase vol more.

   # GJR-GARCH captures leverage via an asymmetric threshold
   gjr = gjr_garch_fit(daily_returns)
   print(f"GJR gamma: {gjr['gamma']:.4f}")
   # Positive gamma means negative shocks get extra weight.

   # News impact curve: plot variance response to return shocks
   nic_garch = news_impact_curve(result)     # symmetric for standard GARCH
   nic_egarch = news_impact_curve(egarch)    # asymmetric for EGARCH
   nic_gjr = news_impact_curve(gjr)          # asymmetric for GJR

   # nic_egarch['shocks'] contains a grid of return shocks
   # nic_egarch['variance'] contains the resulting conditional variance
   # Plotting these shows the characteristic "smirk" for leverage models.


Step 4: Compare Models
------------------------

Use information criteria (AIC, BIC) to select the best model specification
for your data.

.. code-block:: python

   from wraquant.vol import figarch_fit, harch_fit, garch_model_selection

   # Automatic model selection across GARCH variants
   selection = garch_model_selection(daily_returns)
   print(f"Best model: {selection['best_model']}")
   print(f"\nModel rankings (by BIC):")
   for model in selection['rankings']:
       print(f"  {model['name']:<12} AIC={model['aic']:.2f}  BIC={model['bic']:.2f}")

   # FIGARCH for long-memory volatility (slow decay)
   figarch = figarch_fit(daily_returns)
   print(f"\nFIGARCH d (fractional integration): {figarch['d']:.4f}")
   # d close to 0 -> short memory (like standard GARCH)
   # d close to 1 -> integrated (like IGARCH)
   # d between 0.3-0.5 is typical for equity indices


Step 5: Forecast Volatility
-----------------------------

Generate multi-step ahead volatility forecasts from a fitted GARCH model.
These feed into VaR calculations, option pricing, and position sizing.

.. code-block:: python

   from wraquant.vol import garch_forecast

   # Forecast 10 trading days ahead using the GJR model (best for equities)
   forecast = garch_forecast(gjr, horizon=10)

   print("Volatility forecasts (annualized):")
   for i, vol in enumerate(forecast['forecasts'], 1):
       ann_vol = vol * (252 ** 0.5)
       print(f"  Day {i:2d}: {vol:.4f} (ann: {ann_vol:.2%})")

   # GARCH forecasts mean-revert to unconditional variance.
   # Short-horizon forecasts reflect current conditions.
   # Long-horizon forecasts converge to the long-run average.

   unc_var = forecast.get('unconditional_variance')
   if unc_var is not None:
       print(f"\nUnconditional vol: {unc_var**0.5:.4f}")


Step 6: Rolling GARCH
-----------------------

In production, you re-estimate the model periodically on a rolling window
to adapt to structural changes.

.. code-block:: python

   from wraquant.vol import garch_rolling_forecast

   # Re-estimate every 63 days (quarterly) on a 504-day window (2 years)
   rolling = garch_rolling_forecast(
       daily_returns,
       model="GJR",
       window=504,
       refit_every=63,
       forecast_horizon=1,
   )

   # rolling['forecasts'] contains the 1-day-ahead vol forecast at each date
   # rolling['realized'] contains the actual realized vol for comparison
   print(f"Rolling forecast series length: {len(rolling['forecasts'])}")
   print(f"Mean forecast error (RMSE): {rolling['rmse']:.6f}")

   # Compare forecast vs realized to assess model quality.
   # A good model's forecast errors should have no serial correlation
   # and no relationship with the volatility level.


Step 7: Multivariate Volatility with DCC
------------------------------------------

For multi-asset portfolios, use DCC-GARCH to model time-varying correlations
and covariances.

.. code-block:: python

   from wraquant.vol import dcc_fit

   # Fit DCC-GARCH to multi-asset returns
   multi_returns = pd.read_csv("multi_returns.csv", index_col=0, parse_dates=True)

   dcc = dcc_fit(multi_returns)
   print(f"DCC alpha: {dcc['dcc_alpha']:.4f}")
   print(f"DCC beta:  {dcc['dcc_beta']:.4f}")

   # Time-varying correlation matrix at the last observation
   print(f"Current correlation matrix:\n{dcc['correlations'].iloc[-1]}")

   # Time-varying covariance matrix for portfolio optimization
   cov_t = dcc['covariances'].iloc[-1]


Next Steps
----------

- :doc:`/tutorials/risk_analysis` -- Use GARCH volatility forecasts for
  time-varying VaR.
- :doc:`/tutorials/portfolio_construction` -- Feed DCC covariance forecasts
  into portfolio optimization.
- :doc:`/api/vol` -- Full API reference for all volatility functions.
