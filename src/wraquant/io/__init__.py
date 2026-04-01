"""I/O connectors for file formats, streaming, and report export.

This module consolidates all data persistence and export functionality
for wraquant.  It handles reading and writing financial data to local
files in multiple formats, real-time tick streaming via WebSocket
connections, and exporting analysis results to JSON, dictionaries, and
formatted tearsheets.

Key components:

- **File I/O** -- Read and write CSV, Parquet, HDF5, and Excel files with
  ``read_csv`` / ``write_csv``, ``read_parquet`` / ``write_parquet``,
  ``read_hdf`` / ``write_hdf``, ``read_excel`` / ``write_excel``.
  Parquet is recommended for large datasets (columnar, compressed,
  preserves dtypes).
- **Streaming** -- ``WebSocketClient`` connects to real-time market data
  feeds, and ``TickBuffer`` accumulates ticks into OHLCV bars for
  downstream consumption.
- **Export** -- ``to_tearsheet`` renders analysis results as formatted
  HTML/PDF reports, ``to_json`` / ``to_dict`` serialize results for
  APIs, and ``format_table`` produces publication-ready text tables.

Example:
    >>> from wraquant.io import read_parquet, write_parquet, to_json
    >>> prices = read_parquet("prices.parquet")
    >>> # ... run analysis ...
    >>> write_parquet(prices, "prices_clean.parquet")
    >>> json_str = to_json({"sharpe": 1.5, "max_dd": -0.12})

Use ``wraquant.io`` for persisting data to disk and exporting results.
For fetching data from external market data providers (Yahoo Finance,
FRED), use ``wraquant.data`` instead.  For database and cloud storage
connectors (SQL, S3, GCS), install the ``etl`` or ``warehouse`` extra
groups.
"""

from __future__ import annotations

from wraquant.io.export import format_table, to_dict, to_json, to_tearsheet
from wraquant.io.files import (
    read_csv,
    read_excel,
    read_hdf,
    read_parquet,
    write_csv,
    write_excel,
    write_hdf,
    write_parquet,
)
from wraquant.io.streaming import TickBuffer, WebSocketClient

__all__ = [
    # File I/O
    "read_csv",
    "write_csv",
    "read_parquet",
    "write_parquet",
    "read_hdf",
    "write_hdf",
    "read_excel",
    "write_excel",
    # Streaming
    "WebSocketClient",
    "TickBuffer",
    # Export
    "to_tearsheet",
    "to_json",
    "to_dict",
    "format_table",
]
