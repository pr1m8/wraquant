"""File format I/O with financial defaults.

Provides convenience wrappers around pandas I/O functions with sensible
defaults for financial time-series data (date parsing, index handling, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "read_csv",
    "write_csv",
    "read_parquet",
    "write_parquet",
    "read_hdf",
    "write_hdf",
    "read_excel",
    "write_excel",
]


def read_csv(
    path: str | Path,
    date_column: str = "Date",
    parse_dates: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file with financial defaults.

    By default, parses a ``Date`` column and sets it as the index, which
    matches the most common layout for financial time-series CSVs.

    Parameters:
        path: Path to the CSV file.
        date_column: Name of the date column to parse and set as index.
            Ignored when *parse_dates* is ``False``.
        parse_dates: If ``True``, parse *date_column* as datetime and use
            it as the DataFrame index.
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_csv`.

    Returns:
        DataFrame with the CSV contents, optionally date-indexed.
    """
    path = Path(path)

    if parse_dates:
        kwargs.setdefault("parse_dates", [date_column])
        kwargs.setdefault("index_col", date_column)

    return pd.read_csv(path, **kwargs)


def write_csv(
    data: pd.DataFrame | pd.Series,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """Write a DataFrame or Series to CSV.

    Parameters:
        data: Data to write.
        path: Destination file path.
        **kwargs: Additional keyword arguments forwarded to
            :meth:`pandas.DataFrame.to_csv`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, **kwargs)


def read_parquet(
    path: str | Path,
    columns: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a Parquet file.

    Parameters:
        path: Path to the Parquet file.
        columns: Subset of columns to read. ``None`` reads all columns.
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_parquet`.

    Returns:
        DataFrame with the Parquet contents.
    """
    path = Path(path)
    return pd.read_parquet(path, columns=columns, **kwargs)


def write_parquet(
    data: pd.DataFrame,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """Write a DataFrame to Parquet format.

    Parameters:
        data: DataFrame to write.
        path: Destination file path.
        **kwargs: Additional keyword arguments forwarded to
            :meth:`pandas.DataFrame.to_parquet`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, **kwargs)


def read_hdf(
    path: str | Path,
    key: str = "data",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an HDF5 file.

    Requires the ``tables`` (PyTables) package to be installed.

    Parameters:
        path: Path to the HDF5 file.
        key: The group identifier in the HDF5 store.
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_hdf`.

    Returns:
        DataFrame with the HDF5 contents.
    """
    path = Path(path)
    return pd.read_hdf(path, key=key, **kwargs)


def write_hdf(
    data: pd.DataFrame | pd.Series,
    path: str | Path,
    key: str = "data",
    **kwargs: Any,
) -> None:
    """Write a DataFrame or Series to HDF5 format.

    Requires the ``tables`` (PyTables) package to be installed.

    Parameters:
        data: Data to write.
        path: Destination file path.
        key: The group identifier in the HDF5 store.
        **kwargs: Additional keyword arguments forwarded to
            :meth:`pandas.DataFrame.to_hdf`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_hdf(path, key=key, **kwargs)


def read_excel(
    path: str | Path,
    sheet_name: str | int = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an Excel file.

    Parameters:
        path: Path to the Excel file (``.xlsx`` or ``.xls``).
        sheet_name: Name or index of the sheet to read. Defaults to the
            first sheet.
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_excel`.

    Returns:
        DataFrame with the Excel sheet contents.
    """
    path = Path(path)
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


def write_excel(
    data: pd.DataFrame,
    path: str | Path,
    sheet_name: str = "Sheet1",
    **kwargs: Any,
) -> None:
    """Write a DataFrame to Excel format.

    Parameters:
        data: DataFrame to write.
        path: Destination file path.
        sheet_name: Name of the worksheet.
        **kwargs: Additional keyword arguments forwarded to
            :meth:`pandas.DataFrame.to_excel`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_excel(path, sheet_name=sheet_name, **kwargs)
