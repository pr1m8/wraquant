Bayesian Inference (``wraquant.bayes``)
========================================

Bayesian inference for quantitative finance, with support for PyMC, emcee,
BlackJAX, and NumPyro backends. Build probabilistic models for parameter
estimation, uncertainty quantification, and posterior predictive analysis.

Quick Example
-------------

.. code-block:: python

   from wraquant.bayes import models, mcmc

   # Bayesian linear regression with uncertainty quantification
   result = models.bayesian_regression(X, y, n_samples=2000)
   print(f"Beta mean: {result['beta_mean']}")
   print(f"Beta 95% CI: {result['beta_ci']}")

   # MCMC diagnostics
   diag = mcmc.diagnostics(result['trace'])
   print(f"R-hat: {diag['r_hat']}")  # should be < 1.01
   print(f"ESS: {diag['ess']}")       # effective sample size

.. seealso::

   - :doc:`regimes` -- Bayesian changepoint detection
   - :doc:`vol` -- Stochastic volatility via MCMC

API Reference
-------------

.. automodule:: wraquant.bayes
   :members:
   :undoc-members:
   :show-inheritance:

Models
^^^^^^

.. automodule:: wraquant.bayes.models
   :members:

MCMC
^^^^

.. automodule:: wraquant.bayes.mcmc
   :members:

Integrations
^^^^^^^^^^^^

.. automodule:: wraquant.bayes.integrations
   :members:
