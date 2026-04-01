Econometrics (``wraquant.econometrics``)
=========================================

Econometric methods for financial research: panel data models, IV/2SLS,
event studies, structural breaks, and cross-sectional analysis.

Quick Example
-------------

.. code-block:: python

   from wraquant.econometrics import panel, event_study

   # Fixed effects panel regression
   result = panel.fixed_effects(y, X, entity_id="ticker", time_id="date")
   print(f"Coefficients: {result['coefficients']}")
   print(f"R-squared: {result['r_squared']:.4f}")

   # Event study: abnormal returns around an event
   es = event_study.event_study(
       returns=returns,
       event_date="2023-03-10",
       estimation_window=(-252, -21),
       event_window=(-5, 10),
   )
   print(f"CAR: {es['car']:.4f}")
   print(f"CAR t-stat: {es['t_stat']:.4f}")

.. seealso::

   - :doc:`stats` -- Regression and statistical testing
   - :doc:`causal` -- Causal inference methods (DID, synthetic control)

API Reference
-------------

.. automodule:: wraquant.econometrics
   :members:
   :undoc-members:
   :show-inheritance:

Panel Data
~~~~~~~~~~

.. automodule:: wraquant.econometrics.panel
   :members:

Cross Section
~~~~~~~~~~~~~

.. automodule:: wraquant.econometrics.cross_section
   :members:

Time Series Econometrics
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.econometrics.timeseries
   :members:

Event Studies
~~~~~~~~~~~~~

.. automodule:: wraquant.econometrics.event_study
   :members:

Diagnostics
~~~~~~~~~~~

.. automodule:: wraquant.econometrics.diagnostics
   :members:

Volatility
~~~~~~~~~~

.. automodule:: wraquant.econometrics.volatility
   :members:
