Regime Detection (``wraquant.regimes``)
========================================

The regimes module provides 38+ functions for detecting, classifying, and
exploiting market regime shifts. Markets alternate between distinct states
(bull/bear, high/low volatility, risk-on/risk-off), and strategies that
ignore these shifts suffer from unstable parameters and tail-risk blow-ups.

**Model families:**

- **Hidden Markov Models** -- Gaussian HMM, multivariate HMM, Markov-switching regression
- **Kalman filtering** -- linear filter/smoother, time-varying regression, UKF
- **Changepoint detection** -- Bayesian online, PELT, binary segmentation, CUSUM
- **Regime labeling** -- rule-based classification for backtesting
- **Regime scoring** -- stability, separation, predictability metrics

Quick Example
-------------

.. code-block:: python

   from wraquant.regimes import fit_gaussian_hmm, regime_statistics

   # Fit 2-state HMM to daily returns
   hmm = fit_gaussian_hmm(returns, n_states=2)

   # Each state has its own distribution
   for i in range(2):
       print(f"State {i}: mean={hmm['means'][i]:.5f}, "
             f"vol={hmm['variances'][i]**0.5:.4f}")

   # Transition matrix
   print(f"Transition matrix:\n{hmm['transition_matrix']}")

   # Current regime
   print(f"Current regime: {hmm['states'][-1]}")

   # Per-regime performance
   stats = regime_statistics(returns, hmm['states'])
   for regime, s in stats.items():
       print(f"Regime {regime}: Sharpe={s['sharpe']:.3f}, vol={s['vol']:.4f}")

Regime-Aware Portfolio
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.regimes import regime_aware_portfolio

   allocations = {
       0: {"equity": 1.0, "cash": 0.0},   # Bull
       1: {"equity": 0.3, "cash": 0.7},   # Bear
   }
   portfolio = regime_aware_portfolio(returns, hmm['states'], allocations)
   print(f"Regime Sharpe: {portfolio['sharpe']:.4f}")

Kalman Filter
^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.regimes import kalman_regression

   # Time-varying beta estimation
   kf = kalman_regression(asset_returns, market_returns)
   print(f"Current beta: {kf['beta'].iloc[-1]:.4f}")
   # kf['beta'] is a time series of the evolving coefficient

.. seealso::

   - :doc:`/tutorials/regime_investing` -- Full regime-based investing tutorial
   - :doc:`vol` -- Regime-conditional volatility models
   - :doc:`backtest` -- Regime-conditional position sizing

API Reference
-------------

.. automodule:: wraquant.regimes
   :members:
   :undoc-members:
   :show-inheritance:

Base Detection
~~~~~~~~~~~~~~

.. automodule:: wraquant.regimes.base
   :members:

Hidden Markov Models
~~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.regimes.hmm
   :members:

Kalman Filtering
~~~~~~~~~~~~~~~~

.. automodule:: wraquant.regimes.kalman
   :members:

Changepoint Detection
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.regimes.changepoint
   :members:

Regime Scoring
~~~~~~~~~~~~~~

.. automodule:: wraquant.regimes.scoring
   :members:

Regime Labels
~~~~~~~~~~~~~

.. automodule:: wraquant.regimes.labels
   :members:

Integrations
~~~~~~~~~~~~

.. automodule:: wraquant.regimes.integrations
   :members:
