Mathematics (``wraquant.math``)
================================

Advanced mathematical tools for quantitative finance: Levy processes,
network analysis, optimal stopping, Hawkes processes, numerical methods,
spectral analysis, signal processing, information theory, and ergodicity
economics.

Quick Example
-------------

.. code-block:: python

   from wraquant.math import levy, network, optimal_stopping

   # Simulate a Variance Gamma process
   vg_paths = levy.variance_gamma(S0=100, sigma=0.2, theta=-0.1,
                                  nu=0.5, T=1.0, n_paths=1000)

   # Minimum spanning tree of asset correlations
   mst = network.correlation_mst(returns_df)
   print(f"MST edges: {mst['edges']}")

   # American option optimal exercise boundary
   boundary = optimal_stopping.exercise_boundary(
       S_range=(80, 120), K=100, T=1.0, r=0.05, sigma=0.2
   )

.. seealso::

   - :doc:`price` -- Levy process pricing (FFT, COS method)
   - :doc:`vol` -- Hawkes processes for volatility clustering

API Reference
-------------

.. automodule:: wraquant.math
   :members:
   :undoc-members:
   :show-inheritance:

Levy Processes
^^^^^^^^^^^^^^

.. automodule:: wraquant.math.levy
   :members:

Network Analysis
^^^^^^^^^^^^^^^^

.. automodule:: wraquant.math.network
   :members:

Optimal Stopping
^^^^^^^^^^^^^^^^

.. automodule:: wraquant.math.optimal_stopping
   :members:

Hawkes Processes
^^^^^^^^^^^^^^^^

.. automodule:: wraquant.math.hawkes
   :members:

Numerical Methods
^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.math.numerical
   :members:

Spectral Methods
^^^^^^^^^^^^^^^^

.. automodule:: wraquant.math.spectral
   :members:

Signal Processing
^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.math.signals
   :members:

Information Theory
^^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.math.information
   :members:

Ergodicity
^^^^^^^^^^

.. automodule:: wraquant.math.ergodicity
   :members:
