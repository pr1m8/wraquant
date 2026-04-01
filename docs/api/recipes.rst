Recipes (``wraquant.recipes``)
==============================

Pre-built quantitative finance workflows that chain wraquant modules into
complete analysis pipelines. Recipes are thin orchestration layers -- the
real logic lives in individual modules; recipes sequence the calls, align
data, and assemble outputs.

Quick Example
-------------

.. code-block:: python

   import wraquant as wq

   # The "just give me everything" function
   result = wq.analyze(daily_returns)

   # Descriptive statistics
   print(result['descriptive'])

   # Risk metrics (Sharpe, Sortino, max drawdown)
   print(result['risk'])

   # Distribution fit (normal params + KS test)
   print(result['distribution'])

   # Stationarity test (ADF)
   print(result['stationarity'])

   # With a benchmark for relative metrics
   result = wq.analyze(daily_returns, benchmark=spy_returns)
   print(f"Information ratio: {result['benchmark']['information_ratio']:.4f}")
   print(f"Beta: {result['benchmark']['beta']:.4f}")

.. seealso::

   - The :doc:`/getting_started` guide for an introduction to ``analyze``
   - ``wraquant.compose`` for the newer composable workflow system

API Reference
-------------

.. automodule:: wraquant.recipes
   :members:
   :undoc-members:
   :show-inheritance:
