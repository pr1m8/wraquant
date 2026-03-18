"""Database connectors for reading and writing financial data.

All functions in this module require the ``etl`` optional dependency group
which provides SQLAlchemy and connectorx.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "read_sql",
    "write_sql",
    "create_engine",
    "read_sql_fast",
]


@requires_extra("etl")
def read_sql(
    query: str,
    connection_string: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read data from a SQL database using SQLAlchemy.

    Parameters:
        query: SQL query string or table name.
        connection_string: SQLAlchemy-compatible connection URI
            (e.g., ``"postgresql://user:pass@host/db"``).
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_sql`.

    Returns:
        DataFrame with the query results.
    """
    import sqlalchemy

    engine = sqlalchemy.create_engine(connection_string)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, **kwargs)


@requires_extra("etl")
def write_sql(
    data: pd.DataFrame,
    table_name: str,
    connection_string: str,
    if_exists: str = "append",
    **kwargs: Any,
) -> None:
    """Write a DataFrame to a SQL database table.

    Parameters:
        data: DataFrame to write.
        table_name: Destination table name.
        connection_string: SQLAlchemy-compatible connection URI.
        if_exists: Behavior when the table already exists. One of
            ``'fail'``, ``'replace'``, or ``'append'`` (default).
        **kwargs: Additional keyword arguments forwarded to
            :meth:`pandas.DataFrame.to_sql`.
    """
    import sqlalchemy

    engine = sqlalchemy.create_engine(connection_string)
    with engine.connect() as conn:
        data.to_sql(table_name, conn, if_exists=if_exists, **kwargs)
        conn.commit()


@requires_extra("etl")
def create_engine(
    connection_string: str,
    **kwargs: Any,
) -> Any:
    """Create a SQLAlchemy engine.

    A thin wrapper that provides a consistent interface and keeps the
    SQLAlchemy import gated behind the ``etl`` extra.

    Parameters:
        connection_string: SQLAlchemy-compatible connection URI.
        **kwargs: Additional keyword arguments forwarded to
            :func:`sqlalchemy.create_engine`.

    Returns:
        A SQLAlchemy ``Engine`` instance.
    """
    import sqlalchemy

    return sqlalchemy.create_engine(connection_string, **kwargs)


@requires_extra("etl")
def read_sql_fast(
    query: str,
    connection_string: str,
) -> pd.DataFrame:
    """Read from a SQL database using connectorx for speed, with pandas fallback.

    Attempts to use `connectorx <https://github.com/sfu-db/connector-x>`_
    for significantly faster reads.  Falls back to the standard
    :func:`pandas.read_sql` path via SQLAlchemy if connectorx is not
    available or raises an error.

    Parameters:
        query: SQL query string.
        connection_string: Database connection URI.

    Returns:
        DataFrame with the query results.
    """
    try:
        import connectorx as cx

        return cx.read_sql(connection_string, query)
    except (ImportError, Exception):
        # Fall back to pandas + SQLAlchemy when connectorx is unavailable
        # or encounters an unsupported database driver.
        import sqlalchemy

        engine = sqlalchemy.create_engine(connection_string)
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
