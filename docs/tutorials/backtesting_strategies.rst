Backtesting Strategies
======================

This tutorial walks through the full backtesting lifecycle: defining a
strategy, running a backtest, analyzing performance metrics, generating
tearsheets, and running walk-forward validation.


Step 1: Define a Strategy
--------------------------

wraquant strategies inherit from ``Strategy`` and implement
``generate_signals``, which receives price data and returns a signal
series (1 for long, -1 for short, 0 for flat).

.. code-block:: python

   import wraquant as wq
   import pandas as pd
   from wraquant.backtest import Strategy
   from wraquant.ta import ema, rsi, crossover, crossunder

   class MomentumStrategy(Strategy):
       """EMA crossover with RSI filter.

       Go long when fast EMA crosses above slow EMA AND RSI < 70.
       Exit when fast EMA crosses below slow EMA OR RSI > 80.
       """

       def generate_signals(self, prices):
           fast = ema(prices, period=10)
           slow = ema(prices, period=50)
           rsi_values = rsi(prices, period=14)

           # Entry: fast crosses above slow, RSI not overbought
           long_entry = crossover(fast, slow) & (rsi_values < 70)

           # Exit: fast crosses below slow or RSI extremely overbought
           long_exit = crossunder(fast, slow) | (rsi_values > 80)

           # Build position signal: 1 = long, 0 = flat
           signal = pd.Series(0.0, index=prices.index)
           position = 0.0
           for i in range(len(signal)):
               if long_entry.iloc[i]:
                   position = 1.0
               elif long_exit.iloc[i]:
                   position = 0.0
               signal.iloc[i] = position

           return signal


Step 2: Run the Backtest
-------------------------

The ``Backtest`` engine applies your signals to historical data and
computes the resulting returns.

.. code-block:: python

   from wraquant.backtest import Backtest

   prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)["Close"]

   bt = Backtest(MomentumStrategy())
   result = bt.run(prices)

   print(f"Strategy returns: {len(result['returns'])} observations")
   print(f"Total return: {(1 + result['returns']).prod() - 1:.2%}")

   # result contains:
   # - 'returns': daily strategy returns
   # - 'positions': position series over time
   # - 'equity_curve': cumulative equity


Step 3: Performance Analysis
------------------------------

Analyze the strategy with 15+ performance metrics beyond the basics.

.. code-block:: python

   from wraquant.backtest import (
       performance_summary, omega_ratio, kelly_fraction,
       profit_factor, system_quality_number, recovery_factor,
   )

   perf = performance_summary(result['returns'])
   print(f"Annual return:     {perf['annual_return']:.2%}")
   print(f"Annual volatility: {perf['annual_volatility']:.2%}")
   print(f"Sharpe ratio:      {perf['sharpe_ratio']:.4f}")
   print(f"Sortino ratio:     {perf['sortino_ratio']:.4f}")
   print(f"Max drawdown:      {perf['max_drawdown']:.2%}")
   print(f"Win rate:          {perf['win_rate']:.2%}")
   print(f"Total trades:      {perf['total_trades']}")

   # Advanced metrics
   omega = omega_ratio(result['returns'], threshold=0.0)
   print(f"\nOmega ratio: {omega:.4f}")
   # >1.0 means more probability-weighted gains than losses.

   kelly = kelly_fraction(result['returns'])
   print(f"Kelly fraction: {kelly:.4f}")
   # Optimal fraction of capital to bet. In practice, use half-Kelly.

   pf = profit_factor(result['returns'])
   print(f"Profit factor: {pf:.4f}")
   # Gross profit / gross loss. >1.0 is profitable. >2.0 is strong.

   sqn = system_quality_number(result['returns'])
   print(f"SQN: {sqn:.4f}")
   # Van Tharp's quality score. >2.0 is tradable. >3.0 is excellent.


Step 4: Generate a Tearsheet
------------------------------

A tearsheet is a one-page visual summary of strategy performance:
equity curve, drawdown chart, monthly returns heatmap, and key metrics.

.. code-block:: python

   from wraquant.backtest import (
       generate_tearsheet, monthly_returns_table,
       drawdown_table, rolling_metrics_table,
   )

   # Full tearsheet
   tearsheet = generate_tearsheet(result['returns'])

   # Monthly returns table (like a hedge fund factsheet)
   monthly = monthly_returns_table(result['returns'])
   print(monthly)
   # Rows are years, columns are months, values are monthly returns.

   # Worst drawdown episodes
   dd_table = drawdown_table(result['returns'], top_n=5)
   for dd in dd_table:
       print(f"Drawdown: {dd['max_drawdown']:.2%}, "
             f"Start: {dd['start']}, End: {dd['end']}, "
             f"Duration: {dd['duration']} days")

   # Rolling Sharpe and volatility
   rolling = rolling_metrics_table(result['returns'], window=252)
   print(f"\nRolling Sharpe (latest): {rolling['sharpe'].iloc[-1]:.4f}")


Step 5: Compare Strategies
----------------------------

Test multiple strategy variants and compare them side by side.

.. code-block:: python

   from wraquant.backtest import strategy_comparison

   # Define alternative strategies
   class BuyAndHold(Strategy):
       def generate_signals(self, prices):
           return pd.Series(1.0, index=prices.index)

   class FastMomentum(Strategy):
       def generate_signals(self, prices):
           fast = ema(prices, period=5)
           slow = ema(prices, period=20)
           return crossover(fast, slow).astype(float)

   # Run all backtests
   strategies = {
       "Momentum (10/50)": MomentumStrategy(),
       "Fast Momentum (5/20)": FastMomentum(),
       "Buy & Hold": BuyAndHold(),
   }

   results = {}
   for name, strat in strategies.items():
       bt = Backtest(strat)
       results[name] = bt.run(prices)['returns']

   # Side-by-side comparison
   comparison = strategy_comparison(results)
   print(comparison)
   # Tabular comparison of Sharpe, return, drawdown, etc.


Step 6: Walk-Forward Validation
---------------------------------

Walk-forward is the gold standard for strategy validation. It re-optimizes
strategy parameters on a rolling window and tests out-of-sample at each step.

.. code-block:: python

   from wraquant.backtest import walk_forward_backtest

   # Walk-forward with 2-year train, 6-month test windows
   wf_result = walk_forward_backtest(
       strategy=MomentumStrategy(),
       prices=prices,
       train_period=504,     # ~2 years of trading days
       test_period=126,      # ~6 months
       step_size=126,        # advance by one test period
   )

   print(f"Walk-forward Sharpe: {wf_result['sharpe_ratio']:.4f}")
   print(f"Walk-forward MaxDD:  {wf_result['max_drawdown']:.2%}")
   print(f"Number of windows:   {wf_result['n_windows']}")

   # Compare in-sample vs out-of-sample Sharpe to detect overfitting.
   # If IS Sharpe >> OOS Sharpe, the strategy is overfit.


Step 7: Position Sizing
--------------------------

Control risk at the position level with ATR-based sizing, risk parity,
or regime-conditional sizing.

.. code-block:: python

   from wraquant.backtest import (
       PositionSizer, risk_parity_position,
       regime_conditional_sizing,
   )
   from wraquant.ta import atr

   # ATR-based position sizing: risk a fixed amount per trade
   atr_values = atr(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], period=14)
   risk_per_trade = 0.02   # 2% of equity
   equity = 100_000

   position_size = (equity * risk_per_trade) / (2 * atr_values.iloc[-1])
   print(f"Position size (shares): {position_size:.0f}")

   # Regime-conditional sizing: reduce size in bear regimes
   from wraquant.regimes import fit_gaussian_hmm
   daily_returns = wq.returns(prices)
   hmm = fit_gaussian_hmm(daily_returns, n_states=2)

   sizing = regime_conditional_sizing(
       signal=result['positions'],
       states=hmm['states'],
       sizing_map={0: 1.0, 1: 0.3},   # full size in bull, 30% in bear
   )
   print(f"Current position scale: {sizing.iloc[-1]:.2f}")


Next Steps
----------

- :doc:`/tutorials/ml_alpha_research` -- Use ML to generate strategy signals.
- :doc:`/tutorials/risk_analysis` -- Analyze the risk profile of your
  backtest results.
- :doc:`/api/backtest` -- Full API reference for backtesting.
