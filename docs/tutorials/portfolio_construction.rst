Portfolio Construction
=====================

This tutorial demonstrates end-to-end portfolio construction: computing
expected returns and covariances, optimizing with MVO, risk parity, and
Black-Litterman, decomposing risk contributions, and adjusting for
market regimes.


Step 1: Prepare Data
---------------------

.. code-block:: python

   import wraquant as wq
   import pandas as pd
   import numpy as np

   # Multi-asset return data
   prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
   returns = prices.pct_change().dropna()
   assets = returns.columns.tolist()

   print(f"Assets: {assets}")
   print(f"Observations: {len(returns)}")
   print(f"\nAnnualized returns:")
   for col in assets:
       ann_ret = returns[col].mean() * 252
       ann_vol = returns[col].std() * (252 ** 0.5)
       print(f"  {col}: return={ann_ret:.2%}, vol={ann_vol:.2%}")


Step 2: Mean-Variance Optimization
------------------------------------

The classic Markowitz approach: maximize the Sharpe ratio subject to
constraints.

.. code-block:: python

   from wraquant.opt import max_sharpe, min_volatility, mean_variance

   # Maximum Sharpe ratio portfolio
   ms = max_sharpe(returns)
   print("Max Sharpe Portfolio:")
   for asset, weight in zip(assets, ms['weights']):
       print(f"  {asset}: {weight:.2%}")
   print(f"  Expected return: {ms['expected_return']:.4f}")
   print(f"  Expected vol:    {ms['expected_volatility']:.4f}")
   print(f"  Sharpe ratio:    {ms['sharpe_ratio']:.4f}")

   # Minimum volatility portfolio
   mv = min_volatility(returns)
   print(f"\nMin Vol Portfolio: vol={mv['expected_volatility']:.4f}")
   for asset, weight in zip(assets, mv['weights']):
       print(f"  {asset}: {weight:.2%}")

   # MVO is sensitive to expected return estimates (estimation error).
   # Min vol is more robust because it does not require return forecasts.


Step 3: Risk Parity
---------------------

Risk parity allocates so that each asset contributes equally to portfolio
risk. It avoids the concentration problems of MVO.

.. code-block:: python

   from wraquant.opt import risk_parity
   from wraquant.risk import risk_contribution

   rp = risk_parity(returns)
   print("Risk Parity Weights:")
   for asset, weight in zip(assets, rp['weights']):
       print(f"  {asset}: {weight:.2%}")

   # Verify equal risk contributions
   rc = risk_contribution(returns, rp['weights'])
   total_risk = sum(rc)
   print(f"\nRisk contributions (should be equal):")
   for asset, contrib in zip(assets, rc):
       print(f"  {asset}: {contrib:.4f} ({contrib/total_risk:.1%})")


Step 4: Black-Litterman
-------------------------

Black-Litterman combines the market equilibrium (implied by market caps)
with your subjective views to produce stable expected returns for
optimization.

.. code-block:: python

   from wraquant.opt import black_litterman

   # Market capitalizations (relative is fine)
   market_caps = {"AAPL": 3.0e12, "MSFT": 2.8e12, "AMZN": 1.5e12, "GOOGL": 1.8e12}

   # Your views: AAPL will return 12% annualized, MSFT will outperform AMZN by 5%
   views = {"AAPL": 0.12, "MSFT": 0.08}

   bl = black_litterman(returns, market_caps, views)
   print("Black-Litterman Weights:")
   for asset, weight in zip(assets, bl['weights']):
       print(f"  {asset}: {weight:.2%}")

   print(f"\nBL expected returns (annualized):")
   for asset, ret in zip(assets, bl['expected_returns']):
       print(f"  {asset}: {ret:.2%}")

   # BL returns are a blend of equilibrium and your views,
   # weighted by the confidence in each view.


Step 5: Hierarchical Risk Parity
----------------------------------

HRP (Lopez de Prado, 2016) uses hierarchical clustering on the correlation
matrix to build a diversified portfolio without matrix inversion -- making
it more stable than MVO for large asset universes.

.. code-block:: python

   from wraquant.opt import hierarchical_risk_parity

   hrp = hierarchical_risk_parity(returns)
   print("HRP Weights:")
   for asset, weight in zip(assets, hrp['weights']):
       print(f"  {asset}: {weight:.2%}")

   # HRP produces well-diversified portfolios without requiring
   # expected return estimates or covariance matrix inversion.
   # Particularly useful when N (assets) is close to T (observations).


Step 6: Risk Decomposition
----------------------------

For any chosen portfolio, decompose risk to understand where it comes from.

.. code-block:: python

   from wraquant.risk import (
       portfolio_volatility, risk_contribution,
       diversification_ratio, component_var, marginal_var,
   )

   # Use the max Sharpe weights
   w = ms['weights']

   vol = portfolio_volatility(returns, w)
   print(f"Portfolio volatility: {vol:.4f}")

   # Marginal VaR: how much does VaR change per unit weight increase?
   mvar = marginal_var(returns, w, confidence=0.95)
   for asset, mv in zip(assets, mvar):
       print(f"  {asset} marginal VaR: {mv:.4f}")

   # Diversification ratio
   dr = diversification_ratio(returns, w)
   print(f"\nDiversification ratio: {dr:.4f}")
   # A ratio of 1.0 means no diversification benefit.
   # Higher values mean the portfolio benefits from low correlations.


Step 7: Regime-Adjusted Allocation
------------------------------------

Combine regime detection with portfolio optimization for dynamic allocation.

.. code-block:: python

   from wraquant.regimes import fit_gaussian_hmm, regime_statistics

   # Detect regimes on a broad market index
   market_returns = returns.mean(axis=1)
   hmm = fit_gaussian_hmm(market_returns, n_states=2)
   current_regime = hmm['states'][-1]

   # Compute regime-specific covariance matrices
   bull_mask = hmm['states'] == 0
   bear_mask = hmm['states'] == 1

   bull_cov = returns[bull_mask].cov()
   bear_cov = returns[bear_mask].cov()

   # Optimize separately for each regime
   from wraquant.opt import max_sharpe

   bull_portfolio = max_sharpe(returns[bull_mask])
   bear_portfolio = min_volatility(returns[bear_mask])

   print(f"Current regime: {'Bull' if current_regime == 0 else 'Bear'}")
   print(f"\nBull weights: {dict(zip(assets, bull_portfolio['weights']))}")
   print(f"Bear weights: {dict(zip(assets, bear_portfolio['weights']))}")

   # In practice, blend the two portfolios using the regime probability
   # rather than switching abruptly.


Step 8: Backtest the Constructed Portfolio
-------------------------------------------

Evaluate the out-of-sample performance of your chosen portfolio strategy.

.. code-block:: python

   from wraquant.backtest import performance_summary

   # Simple: compute portfolio returns using the chosen weights
   portfolio_returns = (returns * ms['weights']).sum(axis=1)
   equal_weight_returns = returns.mean(axis=1)

   perf_opt = performance_summary(portfolio_returns)
   perf_ew = performance_summary(equal_weight_returns)

   print(f"{'Metric':<25} {'Optimized':>12} {'Equal Wt':>12}")
   print("-" * 50)
   for metric in ['sharpe_ratio', 'max_drawdown', 'annual_return', 'annual_volatility']:
       print(f"{metric:<25} {perf_opt[metric]:>12.4f} {perf_ew[metric]:>12.4f}")


Next Steps
----------

- :doc:`/tutorials/backtesting_strategies` -- Build a complete strategy
  around the optimized portfolio with rebalancing logic.
- :doc:`/tutorials/risk_analysis` -- Deep-dive into the risk properties
  of the constructed portfolio.
- :doc:`/api/opt` -- Full API reference for optimization functions.
- :doc:`/api/risk` -- Portfolio risk decomposition and analytics.
