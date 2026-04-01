I/O (``wraquant.io``)
=====================

ETL, database connectivity, cloud storage, file I/O, and streaming
data utilities for loading and persisting financial data.

Quick Example
-------------

.. code-block:: python

   from wraquant.io import database, files, cloud

   # Read from a SQL database
   df = database.read_sql("SELECT * FROM prices WHERE date > '2023-01-01'",
                          connection_string="sqlite:///market.db")

   # Export to Parquet (columnar, compressed)
   files.to_parquet(df, "prices.parquet")

   # Read from cloud storage
   df = cloud.read_s3("s3://bucket/prices.parquet")

.. seealso::

   - :doc:`data` -- Data fetching and cleaning pipelines

API Reference
-------------

.. automodule:: wraquant.io
   :members:
   :undoc-members:
   :show-inheritance:

Database
~~~~~~~~

.. automodule:: wraquant.io.database
   :members:

Files
~~~~~

.. automodule:: wraquant.io.files
   :members:

Export
~~~~~~

.. automodule:: wraquant.io.export
   :members:

Cloud Storage
~~~~~~~~~~~~~

.. automodule:: wraquant.io.cloud
   :members:

Streaming
~~~~~~~~~

.. automodule:: wraquant.io.streaming
   :members:
