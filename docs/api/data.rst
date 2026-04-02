Data (``wraquant.data``)
========================

Data fetching, cleaning, validation, transformation, and caching. Supports
yfinance, FRED, and NASDAQ Data Link as data sources, with comprehensive
cleaning pipelines for handling missing values, outliers, corporate actions,
and calendar alignment.

Quick Example
-------------

.. code-block:: python

   from wraquant.data import fetch_prices

   # Fetch daily OHLCV from yfinance
   prices = fetch_prices(["AAPL", "MSFT", "GOOGL"], start="2020-01-01")
   print(prices.head())

   # Data is automatically cleaned: forward-filled, split-adjusted,
   # with trading calendar alignment.

.. seealso::

   - :doc:`/getting_started` -- First analysis with fetched data
   - :doc:`stats` -- Statistical analysis on fetched data

API Reference
-------------

.. automodule:: wraquant.data
   :members:
   :undoc-members:
   :show-inheritance:

Loaders
^^^^^^^

.. automodule:: wraquant.data.loaders
   :members:

Cleaning
^^^^^^^^

.. automodule:: wraquant.data.cleaning
   :members:

Advanced Cleaning
^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.data.cleaning_advanced
   :members:

Validation
^^^^^^^^^^

.. automodule:: wraquant.data.validation
   :members:

Advanced Validation
^^^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.data.validation_advanced
   :members:

Transforms
^^^^^^^^^^

.. automodule:: wraquant.data.transforms
   :members:

Calendar
^^^^^^^^

.. automodule:: wraquant.data.calendar
   :members:

Cache
^^^^^

.. automodule:: wraquant.data.cache
   :members:

Base
^^^^

.. automodule:: wraquant.data.base
   :members:

Utilities
^^^^^^^^^

.. automodule:: wraquant.data.utils
   :members:
