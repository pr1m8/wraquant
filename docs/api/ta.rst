Technical Analysis (``wraquant.ta``)
=====================================

263 indicators across 19 sub-modules, covering every category of technical
analysis: moving averages, momentum oscillators, volume studies, trend
detection, volatility measurement, candlestick patterns, cycle analysis,
Fibonacci tools, support/resistance, market breadth, and exotic indicators.

Every indicator accepts ``pd.Series`` (or OHLCV components) and returns
either a ``pd.Series`` or a ``dict[str, pd.Series]`` for multi-output
indicators.

Quick Example
-------------

.. code-block:: python

   from wraquant.ta import rsi, macd, bollinger_bands, adx, atr, crossover

   # Momentum: RSI and MACD
   rsi_values = rsi(close, period=14)
   macd_result = macd(close)   # {'macd', 'signal', 'histogram'}

   # Volatility bands
   bb = bollinger_bands(close, period=20, std_dev=2.0)
   # bb['upper'], bb['middle'], bb['lower']

   # Trend strength (ADX > 25 = strong trend)
   adx_values = adx(high, low, close, period=14)

   # Signal generation: EMA crossover
   from wraquant.ta import ema
   fast = ema(close, period=10)
   slow = ema(close, period=50)
   buy_signal = crossover(fast, slow)

Candlestick Patterns
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.ta import doji, engulfing, hammer, morning_star

   # Each pattern returns a boolean Series
   dojis = doji(open, high, low, close)
   engulfings = engulfing(open, high, low, close)
   hammers = hammer(open, high, low, close)

   print(f"Doji days: {dojis.sum()}")
   print(f"Engulfing days: {engulfings.sum()}")

Advanced Smoothing
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.ta import alma, jma, supersmoother

   # ALMA: Arnaud Legoux Moving Average (low lag, low noise)
   alma_values = alma(close, period=21)

   # JMA: Jurik Moving Average (adaptive smoothing)
   jma_values = jma(close, period=14)

   # Super Smoother (Ehlers 2-pole filter)
   ss = supersmoother(close, period=10)

Support & Resistance Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.ta import find_support_resistance, supply_demand_zones

   levels = find_support_resistance(close, n_levels=5)
   print(f"Support levels: {levels['support']}")
   print(f"Resistance levels: {levels['resistance']}")

   zones = supply_demand_zones(open, high, low, close)
   print(f"Supply zones: {zones['supply']}")
   print(f"Demand zones: {zones['demand']}")

.. seealso::

   - :doc:`/tutorials/ml_alpha_research` -- Use TA indicators as ML features
   - :doc:`/tutorials/backtesting_strategies` -- Build strategies from TA signals
   - :doc:`ml` -- ``technical_features()`` wraps TA into feature DataFrames

API Reference
-------------

.. automodule:: wraquant.ta
   :members:
   :undoc-members:
   :show-inheritance:

Overlap Studies
^^^^^^^^^^^^^^^

Moving averages, bands, and channel studies drawn on the price chart.

.. automodule:: wraquant.ta.overlap
   :members:

Momentum Indicators
^^^^^^^^^^^^^^^^^^^

Oscillators measuring speed and magnitude of price changes.

.. automodule:: wraquant.ta.momentum
   :members:

Volume Indicators
^^^^^^^^^^^^^^^^^

Volume-confirmed signals and accumulation/distribution studies.

.. automodule:: wraquant.ta.volume
   :members:

Trend Indicators
^^^^^^^^^^^^^^^^

Trend direction, strength, and adaptive moving averages.

.. automodule:: wraquant.ta.trend
   :members:

Volatility Indicators
^^^^^^^^^^^^^^^^^^^^^

ATR, Bollinger Width, and OHLC-based volatility estimators.

.. automodule:: wraquant.ta.volatility
   :members:

Pattern Recognition
^^^^^^^^^^^^^^^^^^^

38 candlestick patterns returning boolean match Series.

.. automodule:: wraquant.ta.patterns
   :members:

Signal Generation
^^^^^^^^^^^^^^^^^

Utility functions for combining indicators: crossover, crossunder,
above, below, rising, falling.

.. automodule:: wraquant.ta.signals
   :members:

Statistical Functions
^^^^^^^^^^^^^^^^^^^^^

Z-score, percentile rank, skewness, kurtosis, Hurst exponent,
rolling beta, R-squared.

.. automodule:: wraquant.ta.statistics
   :members:

Cycles
^^^^^^

Hilbert Transform, sine wave indicators, bandpass and roofing filters.

.. automodule:: wraquant.ta.cycles
   :members:

Custom Indicators
^^^^^^^^^^^^^^^^^

Squeeze Momentum, Anchored VWAP, Adaptive RSI, Linear Regression Channel,
Market Structure, Volume-Weighted MACD.

.. automodule:: wraquant.ta.custom
   :members:

Fibonacci
^^^^^^^^^

Retracements, extensions, fans, time zones, pivot points, auto-Fibonacci.

.. automodule:: wraquant.ta.fibonacci
   :members:

Support & Resistance
^^^^^^^^^^^^^^^^^^^^

Algorithmic detection of support/resistance levels, fractal levels,
supply/demand zones, trendlines.

.. automodule:: wraquant.ta.support_resistance
   :members:

Market Breadth
^^^^^^^^^^^^^^

Advance/Decline, McClellan, Arms Index, percent above MA -- for indices
and baskets.

.. automodule:: wraquant.ta.breadth
   :members:

Performance
^^^^^^^^^^^

Relative performance, alpha, tracking error, up/down capture, drawdown
analytics.

.. automodule:: wraquant.ta.performance
   :members:

Smoothing
^^^^^^^^^

ALMA, JMA, Butterworth, Super Smoother, Gaussian, windowed MAs.

.. automodule:: wraquant.ta.smoothing
   :members:

Exotic Indicators
^^^^^^^^^^^^^^^^^

Choppiness Index, Random Walk Index, Polarized Fractal Efficiency,
Elder Thermometer, Connors TPS, and more.

.. automodule:: wraquant.ta.exotic
   :members:

Candlestick Analytics
^^^^^^^^^^^^^^^^^^^^^

Structural candlestick measures: body size, shadow ratios, inside/outside
bars, pin bars.

.. automodule:: wraquant.ta.candles
   :members:

Price Action
^^^^^^^^^^^^

Higher highs/lows, swing points, gap analysis, narrow range, key reversals.

.. automodule:: wraquant.ta.price_action
   :members:
