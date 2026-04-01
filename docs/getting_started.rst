Getting Started
===============

Installation
------------

Install wraquant with pip:

.. code-block:: bash

   pip install wraquant

wraquant has a modular dependency system. Install only what you need:

.. code-block:: bash

   # Market data fetching (yfinance, fredapi, nasdaq-data-link)
   pip install wraquant[market-data]

   # Visualization (plotly, dash)
   pip install wraquant[viz]

   # Machine learning (scikit-learn, torch)
   pip install wraquant[ml]

   # Multiple extras at once
   pip install wraquant[market-data,viz,risk,regimes]

For development with PDM:

.. code-block:: bash

   pdm install -G dev -G market-data -G viz

.. admonition:: Available extras

   ``market-data``, ``timeseries``, ``cleaning``, ``validation``, ``etl``,
   ``warehouse``, ``workflow``, ``optimization``, ``regimes``, ``backtesting``,
   ``risk``, ``pricing``, ``stochastic``, ``causal``, ``quant-math``,
   ``bayes``, ``viz``, ``scale``, ``dev``


First Analysis
--------------

The ``analyze`` function runs a full diagnostic pipeline on a return series
in a single call -- descriptive stats, risk metrics, distribution fitting,
stationarity tests, and optionally regime detection and GARCH volatility.

.. code-block:: python

   import wraquant as wq
   import pandas as pd

   # Load price data and compute returns
   prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)["Close"]
   daily_returns = wq.returns(prices)

   # One-liner comprehensive analysis
   result = wq.analyze(daily_returns)

   # Descriptive statistics
   print(f"Mean return:  {result['descriptive']['mean']:.4f}")
   print(f"Volatility:   {result['descriptive']['std']:.4f}")
   print(f"Skewness:     {result['descriptive']['skewness']:.4f}")

   # Risk metrics
   print(f"Sharpe ratio: {result['risk']['sharpe']:.4f}")
   print(f"Max drawdown: {result['risk']['max_drawdown']:.4f}")

   # Stationarity
   print(f"ADF p-value:  {result['stationarity']['p_value']:.4f}")
   print(f"Stationary:   {result['stationarity']['is_stationary']}")

The output dictionary is organized into sections: ``descriptive``, ``risk``,
``distribution``, ``stationarity``, ``regime`` (if detected), ``garch`` (if
fitted), and ``benchmark`` (if a benchmark series is provided).


Composable Workflows
--------------------

For more control, use the workflow system to chain specific analysis steps:

.. code-block:: python

   from wraquant.compose import Workflow, steps

   wf = (
       Workflow("my_analysis")
       .add(steps.returns())
       .add(steps.regime_detect(n_regimes=2))
       .add(steps.garch_vol())
       .add(steps.risk_metrics())
   )

   result = wf.run(prices)
   print(result.risk)      # risk metrics dict
   print(result.regimes)   # regime detection output
   print(result.garch)     # GARCH model results

Pre-built workflows are available for common patterns:

.. code-block:: python

   import wraquant as wq

   # Quick analysis (stats + risk + regimes + GARCH)
   result = wq.quick_analysis_workflow().run(prices)

   # Risk-focused analysis
   result = wq.risk_workflow().run(prices)

   # Portfolio optimization pipeline
   result = wq.portfolio_workflow().run(returns_df)


GARCH Volatility Modeling
-------------------------

Fit a GARCH model to capture volatility clustering and forecast future
volatility. wraquant supports the full GARCH family: standard GARCH,
EGARCH (leverage effect), GJR-GARCH (asymmetric threshold), FIGARCH
(long memory), and HARCH (heterogeneous horizons).

.. code-block:: python

   from wraquant.vol import garch_fit, egarch_fit, garch_forecast, news_impact_curve

   # Fit standard GARCH(1,1)
   result = garch_fit(daily_returns)
   print(f"omega: {result['omega']:.6f}")
   print(f"alpha: {result['alpha']:.4f}")
   print(f"beta:  {result['beta']:.4f}")
   print(f"Persistence: {result['alpha'] + result['beta']:.4f}")

   # EGARCH captures asymmetric (leverage) effects
   egarch = egarch_fit(daily_returns)
   print(f"Leverage parameter: {egarch['gamma']:.4f}")
   # Negative gamma means negative returns increase vol more

   # Forecast 10 days ahead
   forecast = garch_forecast(result, horizon=10)
   print(f"1-day vol forecast:  {forecast['forecasts'][0]:.4f}")
   print(f"10-day vol forecast: {forecast['forecasts'][-1]:.4f}")

   # News impact curve: how do shocks map to volatility?
   nic = news_impact_curve(result)
   # nic['shocks'] and nic['variance'] show the asymmetric response

See the :doc:`tutorials/volatility_modeling` tutorial for model comparison,
rolling GARCH, and diagnostic interpretation.


Regime Detection
----------------

Financial markets alternate between distinct regimes (bull/bear, high/low
volatility). wraquant provides Hidden Markov Models, Markov-switching
regression, Kalman filtering, and changepoint detection.

.. code-block:: python

   from wraquant.regimes import fit_gaussian_hmm, regime_statistics

   # Fit a 2-state Gaussian HMM
   hmm = fit_gaussian_hmm(daily_returns, n_states=2)

   # Each state has its own mean and variance
   for i in range(2):
       print(f"State {i}: mean={hmm['means'][i]:.4f}, "
             f"vol={hmm['variances'][i]**0.5:.4f}")

   # Transition matrix: how likely is each state to persist?
   print(f"Transition matrix:\n{hmm['transition_matrix']}")

   # State sequence: which regime was active at each time?
   print(f"Current regime: {hmm['states'][-1]}")

   # Per-regime statistics
   stats = regime_statistics(daily_returns, hmm['states'])
   print(stats)  # mean, vol, Sharpe, max DD per regime

See the :doc:`tutorials/regime_investing` tutorial for regime-conditional
portfolio construction and backtesting.


Portfolio Optimization
----------------------

Build optimal portfolios using mean-variance optimization, risk parity,
Black-Litterman, or Hierarchical Risk Parity.

.. code-block:: python

   from wraquant.opt import max_sharpe, risk_parity, black_litterman
   from wraquant.risk import risk_contribution

   # Max Sharpe portfolio
   result = max_sharpe(returns_df)
   print(f"Weights: {result['weights']}")
   print(f"Expected return: {result['expected_return']:.4f}")
   print(f"Expected vol:    {result['expected_volatility']:.4f}")
   print(f"Sharpe ratio:    {result['sharpe_ratio']:.4f}")

   # Risk parity: equal risk contribution from each asset
   rp = risk_parity(returns_df)
   contributions = risk_contribution(returns_df, rp['weights'])
   print(f"Risk contributions: {contributions}")
   # All contributions should be approximately equal

   # Black-Litterman: blend market equilibrium with your views
   views = {"AAPL": 0.10, "MSFT": 0.08}  # expected returns
   bl = black_litterman(returns_df, market_caps, views)
   print(f"BL weights: {bl['weights']}")

See the :doc:`tutorials/portfolio_construction` tutorial for full examples
with risk decomposition and regime-adjusted allocation.


Backtesting
-----------

Backtest trading strategies with wraquant's event-driven engine, then
analyze performance with tearsheets and 15+ advanced metrics.

.. code-block:: python

   from wraquant.backtest import Backtest, Strategy, performance_summary
   from wraquant.backtest import generate_tearsheet
   from wraquant.ta import ema, crossover

   # Define a simple moving average crossover strategy
   class MACrossover(Strategy):
       def generate_signals(self, prices):
           fast = ema(prices, period=10)
           slow = ema(prices, period=50)
           return crossover(fast, slow).astype(float)

   # Run backtest
   bt = Backtest(MACrossover())
   result = bt.run(prices)

   # Performance analysis
   perf = performance_summary(result['returns'])
   print(f"Total return: {perf['total_return']:.2%}")
   print(f"Sharpe ratio: {perf['sharpe_ratio']:.4f}")
   print(f"Max drawdown: {perf['max_drawdown']:.2%}")
   print(f"Win rate:     {perf['win_rate']:.2%}")

   # Generate full tearsheet
   tearsheet = generate_tearsheet(result['returns'])

See the :doc:`tutorials/backtesting_strategies` tutorial for walk-forward
validation, position sizing, and regime-conditional strategies.


Technical Analysis
------------------

wraquant provides 263 technical indicators across 19 modules. Every indicator
accepts ``pd.Series`` and returns ``pd.Series`` or ``dict[str, pd.Series]``.

.. code-block:: python

   from wraquant.ta import rsi, macd, bollinger_bands, adx, atr

   # Momentum
   rsi_values = rsi(close, period=14)
   macd_result = macd(close)  # dict with 'macd', 'signal', 'histogram'

   # Volatility bands
   bb = bollinger_bands(close, period=20, std_dev=2.0)
   # bb['upper'], bb['middle'], bb['lower']

   # Trend strength
   adx_values = adx(high, low, close, period=14)
   # ADX > 25 indicates a strong trend

   # Position sizing with ATR
   atr_values = atr(high, low, close, period=14)
   stop_distance = 2.0 * atr_values.iloc[-1]
   position_size = risk_per_trade / stop_distance

See the full :doc:`api/ta` reference for all 263 indicators.


What's Next
-----------

.. grid:: 2

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Step-by-step guides for risk analysis, regime investing,
      volatility modeling, portfolio construction, backtesting,
      and ML alpha research.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Complete API documentation for all 25+ modules with
      examples, parameter guidance, and cross-references.
