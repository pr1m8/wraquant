Backtesting (``wraquant.backtest``)
====================================

The backtesting module provides event-driven and vectorized backtesting
engines, strategy abstractions, position sizing, performance metrics, event
tracking, and tearsheet generation.

**Key components:**

- ``Backtest`` -- event-driven engine with fill simulation
- ``VectorizedBacktest`` -- fast vectorized engine for signal-based strategies
- ``Strategy`` -- base class for defining trading strategies
- ``PositionSizer`` -- ATR-based, risk parity, and regime-conditional sizing
- 15+ performance metrics beyond Sharpe: omega, Kelly, profit factor, SQN, recovery factor
- Tearsheet generation: equity curves, drawdown tables, monthly heatmaps

Quick Example
-------------

.. code-block:: python

   from wraquant.backtest import Backtest, Strategy, performance_summary
   from wraquant.ta import ema, crossover

   class MACrossover(Strategy):
       def generate_signals(self, prices):
           fast = ema(prices, period=10)
           slow = ema(prices, period=50)
           return crossover(fast, slow).astype(float)

   bt = Backtest(MACrossover())
   result = bt.run(prices)

   perf = performance_summary(result['returns'])
   print(f"Sharpe: {perf['sharpe_ratio']:.4f}")
   print(f"Max DD: {perf['max_drawdown']:.2%}")
   print(f"Win rate: {perf['win_rate']:.2%}")

Tearsheet
^^^^^^^^^^

.. code-block:: python

   from wraquant.backtest import generate_tearsheet, monthly_returns_table

   tearsheet = generate_tearsheet(result['returns'])
   monthly = monthly_returns_table(result['returns'])
   print(monthly)   # year x month heatmap of returns

Walk-Forward Validation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.backtest import walk_forward_backtest

   wf = walk_forward_backtest(
       strategy=MACrossover(),
       prices=prices,
       train_period=504,
       test_period=126,
   )
   print(f"Walk-forward Sharpe: {wf['sharpe_ratio']:.4f}")

Regime-Conditional Sizing
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.backtest import regime_conditional_sizing

   sizing = regime_conditional_sizing(
       signal=result['positions'],
       states=hmm['states'],
       sizing_map={0: 1.0, 1: 0.3},   # full in bull, 30% in bear
   )

.. seealso::

   - :doc:`/tutorials/backtesting_strategies` -- Full backtesting tutorial
   - :doc:`risk` -- Metrics module (single source of truth for risk calculations)
   - :doc:`regimes` -- Regime detection for conditional strategies

API Reference
-------------

.. automodule:: wraquant.backtest
   :members:
   :undoc-members:
   :show-inheritance:

Engine
~~~~~~

.. automodule:: wraquant.backtest.engine
   :members:

Strategy
~~~~~~~~

.. automodule:: wraquant.backtest.strategy
   :members:

Position Sizing
~~~~~~~~~~~~~~~

.. automodule:: wraquant.backtest.position
   :members:

Events
~~~~~~

.. automodule:: wraquant.backtest.events
   :members:

Metrics
~~~~~~~

.. automodule:: wraquant.backtest.metrics
   :members:

Tearsheet
~~~~~~~~~

.. automodule:: wraquant.backtest.tearsheet
   :members:

Integrations
~~~~~~~~~~~~

.. automodule:: wraquant.backtest.integrations
   :members:
