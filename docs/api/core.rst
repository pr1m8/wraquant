Core (``wraquant.core``)
========================

Foundation infrastructure: configuration management, type definitions,
custom exceptions, structured logging, decorators, and result dataclasses.

Quick Example
-------------

.. code-block:: python

   import wraquant as wq

   # Configuration
   cfg = wq.get_config()
   print(f"Backend: {cfg.backend}")

   # Type aliases for consistent function signatures
   from wraquant.core.types import PriceSeries, ReturnSeries, OHLCVFrame

   # Exceptions
   from wraquant.core.exceptions import WQError, ValidationError

   # Result dataclasses
   from wraquant.core.results import GARCHResult, BacktestResult

API Reference
-------------

.. automodule:: wraquant.core
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
^^^^^^^^^^^^^

.. automodule:: wraquant.core.config
   :members:

Types
^^^^^

.. automodule:: wraquant.core.types
   :members:

Exceptions
^^^^^^^^^^

.. automodule:: wraquant.core.exceptions
   :members:

Logging
^^^^^^^

.. automodule:: wraquant.core.logging
   :members:

Decorators
^^^^^^^^^^

.. automodule:: wraquant.core.decorators
   :members:

Results
^^^^^^^

.. automodule:: wraquant.core.results
   :members:
