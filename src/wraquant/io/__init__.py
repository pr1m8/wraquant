"""I/O connectors for databases, file formats, cloud storage, and streaming.

This module consolidates all data ingestion and export functionality
including local file I/O, SQL database connectors, cloud storage access,
real-time streaming utilities, and reporting exports.
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
