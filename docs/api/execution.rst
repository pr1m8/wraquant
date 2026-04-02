Execution Algorithms (``wraquant.execution``)
==============================================

Execution algorithms for minimizing market impact and transaction costs:
TWAP, VWAP, Almgren-Chriss optimal execution, and transaction cost analysis.

Quick Example
-------------

.. code-block:: python

   from wraquant.execution import algorithms, cost, optimal

   # TWAP schedule: split order evenly across time
   schedule = algorithms.twap(total_shares=10000, n_intervals=20)

   # VWAP schedule: split order proportional to volume profile
   schedule = algorithms.vwap(total_shares=10000, volume_profile=hist_volume)

   # Almgren-Chriss optimal execution
   ac = optimal.almgren_chriss(
       total_shares=10000,
       volatility=0.02,
       impact_coeff=0.001,
       risk_aversion=1e-6,
       T=1.0,
       n_intervals=20,
   )
   print(f"Optimal trajectory: {ac['trajectory']}")
   print(f"Expected cost: {ac['expected_cost']:.4f}")

   # Transaction cost estimation
   tc = cost.estimate_cost(shares=10000, price=50.0, spread=0.02, impact=0.001)
   print(f"Estimated cost: ${tc:.2f}")

.. seealso::

   - :doc:`microstructure` -- Liquidity and market quality for execution analysis
   - :doc:`backtest` -- Incorporate execution costs into backtests

API Reference
-------------

.. automodule:: wraquant.execution
   :members:
   :undoc-members:
   :show-inheritance:

Algorithms
^^^^^^^^^^

.. automodule:: wraquant.execution.algorithms
   :members:

Transaction Costs
^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.execution.cost
   :members:

Optimal Execution
^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.execution.optimal
   :members:
