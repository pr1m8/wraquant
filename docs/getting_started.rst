Getting Started
===============

Installation
------------

Install wraquant with pip or pdm:

.. code-block:: bash

   pip install wraquant

For optional dependency groups (e.g. market data, visualization, Bayesian inference),
install the extras you need:

.. code-block:: bash

   pip install wraquant[market-data,viz,bayes]

Or with PDM during development:

.. code-block:: bash

   pdm install -G dev -G market-data -G viz

Quick Example
-------------

The ``analyze`` convenience function runs a full diagnostic pipeline on a
returns series in a single call:

.. code-block:: python

   import wraquant as wq
   import pandas as pd

   # Load your daily returns
   prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)["Close"]
   daily_returns = wq.returns(prices)

   # One-liner: descriptive stats, risk metrics, regime detection
   result = wq.analyze(daily_returns)

   print(f"Sharpe ratio : {result['risk']['sharpe']:.4f}")
   print(f"Max drawdown : {result['risk']['max_drawdown']:.4f}")

Module Overview
---------------

wraquant is organized into focused modules. Click through to the API reference
for full documentation.

* :doc:`api/risk` -- VaR, CVaR, copulas, EVT, stress testing, factor risk
* :doc:`api/vol` -- GARCH family, realized volatility, DCC, stochastic vol
* :doc:`api/regimes` -- HMM, Markov-switching, Kalman filtering, changepoints
* :doc:`api/ta` -- 263 technical indicators across 19 sub-modules
* :doc:`api/ml` -- sklearn + PyTorch pipelines, walk-forward, ensembles, deep learning
* :doc:`api/price` -- Options pricing, fixed income, SDEs, Levy processes
* :doc:`api/ts` -- Decomposition, forecasting, changepoints, wavelets
* :doc:`api/stats` -- Regression, correlation, distributions, cointegration
* :doc:`api/backtest` -- Event-driven engine, strategies, position sizing, tearsheets
* :doc:`api/opt` -- MVO, risk parity, Black-Litterman, HRP, convex/nonlinear solvers
* :doc:`api/microstructure` -- Liquidity, toxicity, market quality
* :doc:`api/execution` -- TWAP, VWAP, Almgren-Chriss optimal execution
* :doc:`api/causal` -- DID, synthetic control, IPW, treatment effects
* :doc:`api/forex` -- Pairs, sessions, carry trade, FX risk
* :doc:`api/bayes` -- PyMC, emcee, BlackJAX, NumPyro, Bayesian models
* :doc:`api/viz` -- Interactive Plotly dashboards, tearsheets, candlestick charts
* :doc:`api/data` -- Data fetching (yfinance, FRED, NASDAQ), cleaning, validation
* :doc:`api/econometrics` -- Panel data, IV/2SLS, event studies, structural breaks
* :doc:`api/io` -- Database, cloud storage, file I/O, streaming
* :doc:`api/math` -- Levy processes, networks, optimal stopping, Hawkes, PDEs
* :doc:`api/core` -- Configuration, types, exceptions, logging, decorators
* :doc:`api/recipes` -- Pre-built quantitative finance workflows
