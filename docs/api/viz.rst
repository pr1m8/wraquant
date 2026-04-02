Visualization (``wraquant.viz``)
================================

Interactive Plotly-based visualizations for portfolio analysis, risk
monitoring, regime dashboards, candlestick charts, tearsheets, and
correlation networks.

**Submodules:**

- **Charts** -- line, scatter, histogram, heatmap utilities
- **Candlestick** -- OHLC candlestick charts with indicator overlays
- **Dashboard** -- comprehensive multi-panel dashboards
- **Portfolio** -- allocation, efficient frontier, rebalancing visualization
- **Risk** -- VaR, drawdown, risk decomposition charts
- **Returns** -- equity curves, monthly heatmaps, distribution plots
- **Time Series** -- decomposition, forecast, and ACF/PACF plots
- **Advanced** -- correlation networks, vol surfaces, regime maps

Quick Example
-------------

.. code-block:: python

   from wraquant.viz import charts, candlestick, portfolio, risk

   # Candlestick chart with volume
   fig = candlestick.plot(ohlcv)

   # Equity curve with drawdown overlay
   fig = risk.equity_drawdown_chart(strategy_returns)

   # Portfolio allocation pie chart
   fig = portfolio.allocation_chart(weights, asset_names)

   # Monthly returns heatmap
   fig = charts.monthly_heatmap(returns)

.. seealso::

   - :doc:`backtest` -- Tearsheet generation includes visualization
   - :doc:`risk` -- Risk analysis outputs that can be visualized

API Reference
-------------

.. automodule:: wraquant.viz
   :members:
   :undoc-members:
   :show-inheritance:

Charts
^^^^^^

.. automodule:: wraquant.viz.charts
   :members:

Candlestick
^^^^^^^^^^^

.. automodule:: wraquant.viz.candlestick
   :members:

Dashboard
^^^^^^^^^

.. automodule:: wraquant.viz.dashboard
   :members:

Interactive
^^^^^^^^^^^

.. automodule:: wraquant.viz.interactive
   :members:

Portfolio
^^^^^^^^^

.. automodule:: wraquant.viz.portfolio
   :members:

Returns
^^^^^^^

.. automodule:: wraquant.viz.returns
   :members:

Risk
^^^^

.. automodule:: wraquant.viz.risk
   :members:

Time Series
^^^^^^^^^^^

.. automodule:: wraquant.viz.timeseries
   :members:

Advanced
^^^^^^^^

.. automodule:: wraquant.viz.advanced
   :members:

Themes
^^^^^^

.. automodule:: wraquant.viz.themes
   :members:
