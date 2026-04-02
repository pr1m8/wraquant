Statistics (``wraquant.stats``)
===============================

Statistical analysis for financial data: descriptive statistics, hypothesis
testing, correlation and covariance estimation, distribution fitting,
cointegration, regression, factor analysis, and robust statistics.

**Submodules:**

- **Descriptive** -- summary stats, rolling Sharpe, return attribution
- **Regression** -- OLS, WLS, rolling OLS, Fama-MacBeth, Newey-West
- **Correlation** -- shrunk covariance, distance correlation, mutual information, MST
- **Distributions** -- fit distributions, tail index, KDE, Q-Q plot data
- **Cointegration** -- Engle-Granger, Johansen, spread, hedge ratio, pairs signals
- **Tests** -- normality, stationarity, autocorrelation, heteroskedasticity, structural breaks
- **Factor analysis** -- PCA factors, Fama-French, factor loadings, varimax rotation
- **Robust** -- MAD, trimmed mean, winsorize, Huber mean, outlier detection

Quick Example
-------------

.. code-block:: python

   from wraquant.stats import summary_stats, test_normality, test_stationarity

   # Comprehensive summary statistics
   stats = summary_stats(returns)
   print(f"Mean:     {stats['mean']:.6f}")
   print(f"Std:      {stats['std']:.4f}")
   print(f"Skewness: {stats['skewness']:.4f}")
   print(f"Kurtosis: {stats['kurtosis']:.4f}")

   # Normality test (JB + Shapiro-Wilk)
   norm = test_normality(returns)
   print(f"JB p-value: {norm['jarque_bera_pvalue']:.4f}")
   # p < 0.05 rejects normality (expected for financial returns)

   # Stationarity test (ADF)
   stat = test_stationarity(returns)
   print(f"ADF p-value: {stat['p_value']:.4f}")
   print(f"Stationary: {stat['is_stationary']}")

Cointegration & Pairs Trading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.stats import (
       engle_granger, half_life, spread, zscore_signal,
       find_cointegrated_pairs,
   )

   # Test cointegration between two assets
   eg = engle_granger(prices_a, prices_b)
   print(f"Cointegrated: {eg['cointegrated']} (p={eg['p_value']:.4f})")
   print(f"Hedge ratio: {eg['hedge_ratio']:.4f}")

   # Compute the mean-reverting spread
   s = spread(prices_a, prices_b, hedge_ratio=eg['hedge_ratio'])
   hl = half_life(s)
   print(f"Half-life: {hl:.1f} days")

   # Generate trading signals from z-score of the spread
   signals = zscore_signal(s, entry_z=2.0, exit_z=0.5)

   # Scan for cointegrated pairs in a universe
   pairs = find_cointegrated_pairs(prices_df, significance=0.05)
   for pair in pairs:
       print(f"{pair['asset_a']}/{pair['asset_b']}: p={pair['p_value']:.4f}")

Factor Analysis
^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.stats import pca_factors, fama_french_regression

   # PCA factor decomposition
   factors = pca_factors(returns_df, n_factors=3)
   print(f"Explained variance: {factors['explained_variance_ratio']}")

   # Fama-French regression
   ff = fama_french_regression(returns, model="3factor")
   print(f"Alpha: {ff['alpha']:.4f} (p={ff['alpha_pvalue']:.3f})")

.. seealso::

   - :doc:`risk` -- Risk metrics built on statistical foundations
   - :doc:`ts` -- Time series analysis and forecasting
   - :doc:`/tutorials/risk_analysis` -- Uses statistical tests in risk workflows

API Reference
-------------

.. automodule:: wraquant.stats
   :members:
   :undoc-members:
   :show-inheritance:

Descriptive Statistics
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.stats.descriptive
   :members:

Regression
^^^^^^^^^^

.. automodule:: wraquant.stats.regression
   :members:

Correlation
^^^^^^^^^^^

.. automodule:: wraquant.stats.correlation
   :members:

Distributions
^^^^^^^^^^^^^

.. automodule:: wraquant.stats.distributions
   :members:

Cointegration
^^^^^^^^^^^^^

.. automodule:: wraquant.stats.cointegration
   :members:

Statistical Tests
^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.stats.tests
   :members:

Factor Analysis
^^^^^^^^^^^^^^^

.. automodule:: wraquant.stats.factor_analysis
   :members:

Factor Models
^^^^^^^^^^^^^

.. automodule:: wraquant.stats.factor
   :members:

Dependence
^^^^^^^^^^

.. automodule:: wraquant.stats.dependence
   :members:

Robust Statistics
^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.stats.robust
   :members:
