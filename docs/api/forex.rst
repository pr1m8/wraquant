Forex Analysis (``wraquant.forex``)
====================================

Foreign exchange analysis tools: currency pair analytics, trading session
detection, carry trade analysis, and FX-specific risk measures.

Quick Example
-------------

.. code-block:: python

   from wraquant.forex import pairs, session, carry

   # Identify the active trading session
   current_session = session.current_session()
   print(f"Active session: {current_session}")

   # Carry trade analysis: interest rate differential
   carry_result = carry.carry_trade_return(
       spot_rate=1.10,
       domestic_rate=0.05,
       foreign_rate=0.03,
       holding_period=90,
   )
   print(f"Carry return: {carry_result['return']:.4f}")

   # Currency pair correlation analysis
   corr = pairs.cross_correlation(fx_returns_df)
   print(corr)

.. seealso::

   - :doc:`stats` -- Cointegration for FX pairs trading
   - :doc:`risk` -- VaR for FX portfolios

API Reference
-------------

.. automodule:: wraquant.forex
   :members:
   :undoc-members:
   :show-inheritance:

Pairs
^^^^^

.. automodule:: wraquant.forex.pairs
   :members:

Sessions
^^^^^^^^

.. automodule:: wraquant.forex.session
   :members:

Analysis
^^^^^^^^

.. automodule:: wraquant.forex.analysis
   :members:

Carry Trade
^^^^^^^^^^^

.. automodule:: wraquant.forex.carry
   :members:
