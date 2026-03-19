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

    Connects to any SQLAlchemy-supported database (PostgreSQL, MySQL,
    SQLite, etc.), executes the query, and returns the result as a
    DataFrame.  The connection is automatically closed after the read.

    Parameters:
        query (str): SQL query string or table name.
        connection_string (str): SQLAlchemy-compatible connection URI
            (e.g., ``"postgresql://user:pass@host/db"``).
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_sql`.

    Returns:
        pd.DataFrame: DataFrame with the query results.

    Example:
        >>> df = read_sql("SELECT * FROM prices", "sqlite:///data.db")  # doctest: +SKIP

    See Also:
        write_sql: Write a DataFrame to a SQL table.
        read_sql_fast: Faster alternative using connectorx.
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

    Inserts the DataFrame rows into the specified table, with
    configurable behavior when the table already exists.  The
    transaction is committed automatically.

    Parameters:
        data (pd.DataFrame): DataFrame to write.
        table_name (str): Destination table name.
        connection_string (str): SQLAlchemy-compatible connection URI.
        if_exists (str): Behavior when the table already exists:
            ``'fail'`` (raise), ``'replace'`` (drop and recreate),
            or ``'append'`` (insert rows, default).
        **kwargs: Additional keyword arguments forwarded to
            :meth:`pandas.DataFrame.to_sql`.

    Example:
        >>> write_sql(df, "prices", "sqlite:///data.db")  # doctest: +SKIP

    See Also:
        read_sql: Read data from a SQL database.
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

    connectorx can be 5--10x faster than pandas+SQLAlchemy for large
    result sets because it uses native database drivers and avoids
    Python-level row iteration.  If connectorx is not installed or
    encounters an unsupported driver, the function transparently falls
    back to the standard SQLAlchemy path.

    Parameters:
        query (str): SQL query string.
        connection_string (str): Database connection URI.

    Returns:
        pd.DataFrame: DataFrame with the query results.

    Example:
        >>> df = read_sql_fast("SELECT * FROM ticks", "postgresql://user:pw@host/db")  # doctest: +SKIP

    See Also:
        read_sql: Standard SQLAlchemy-based reader.
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
