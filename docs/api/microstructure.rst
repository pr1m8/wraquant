Market Microstructure (``wraquant.microstructure``)
====================================================

Tools for analyzing market microstructure: liquidity measurement, order
flow toxicity, and market quality metrics. These are essential for
execution analysis, market-making strategy design, and understanding
the microscopic structure of price formation.

Quick Example
-------------

.. code-block:: python

   from wraquant.microstructure import liquidity, toxicity

   # Bid-ask spread estimation from trade data
   spread = liquidity.effective_spread(trades)
   print(f"Effective spread: {spread:.4f}")

   # VPIN (Volume-Synchronized Probability of Informed Trading)
   vpin = toxicity.vpin(volume_bars)
   print(f"Current VPIN: {vpin.iloc[-1]:.4f}")
   # High VPIN signals toxic order flow (informed trading)

.. seealso::

   - :doc:`execution` -- Execution algorithms that account for microstructure
   - :doc:`ml` -- Microstructure features for ML models

API Reference
-------------

.. automodule:: wraquant.microstructure
   :members:
   :undoc-members:
   :show-inheritance:

Liquidity
^^^^^^^^^

.. automodule:: wraquant.microstructure.liquidity
   :members:

Toxicity
^^^^^^^^

.. automodule:: wraquant.microstructure.toxicity
   :members:

Market Quality
^^^^^^^^^^^^^^

.. automodule:: wraquant.microstructure.market_quality
   :members:
