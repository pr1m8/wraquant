"""Cloud storage connectors for S3 and Google Cloud Storage.

Functions are gated behind optional dependencies (``s3fs``/``boto3`` for
AWS S3, ``gcsfs`` for Google Cloud Storage) which are part of the ``etl``
extra group.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "read_s3",
    "write_s3",
    "list_s3",
    "read_gcs",
    "write_gcs",
]


@requires_extra("etl")
def read_s3(
    bucket: str,
    key: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a file from Amazon S3 into a DataFrame.

    Supports Parquet and CSV formats, determined by the file extension.
    Requires ``s3fs`` (part of the ``etl`` extra).

    Parameters:
        bucket: S3 bucket name.
        key: Object key (path) within the bucket.
        **kwargs: Additional keyword arguments forwarded to the
            underlying pandas reader (``read_parquet`` or ``read_csv``).

    Returns:
        DataFrame with the file contents.
    """
    s3_path = f"s3://{bucket}/{key}"

    if key.endswith(".parquet") or key.endswith(".pq"):
        return pd.read_parquet(s3_path, **kwargs)
    else:
        return pd.read_csv(s3_path, **kwargs)


@requires_extra("etl")
def write_s3(
    data: pd.DataFrame,
    bucket: str,
    key: str,
    **kwargs: Any,
) -> None:
    """Write a DataFrame to Amazon S3.

    Supports Parquet and CSV formats, determined by the file extension.
    Requires ``s3fs`` (part of the ``etl`` extra).

    Parameters:
        data: DataFrame to write.
        bucket: S3 bucket name.
        key: Object key (path) within the bucket.
        **kwargs: Additional keyword arguments forwarded to the
            underlying pandas writer (``to_parquet`` or ``to_csv``).
    """
    s3_path = f"s3://{bucket}/{key}"

    if key.endswith(".parquet") or key.endswith(".pq"):
        data.to_parquet(s3_path, **kwargs)
    else:
        data.to_csv(s3_path, **kwargs)


@requires_extra("etl")
def list_s3(
    bucket: str,
    prefix: str = "",
    **kwargs: Any,
) -> list[str]:
    """List files in an S3 bucket under a given prefix.

    Requires ``s3fs`` (part of the ``etl`` extra).

    Parameters:
        bucket: S3 bucket name.
        prefix: Key prefix to filter results. Defaults to listing the
            entire bucket.
        **kwargs: Additional keyword arguments forwarded to
            ``s3fs.S3FileSystem.ls``.

    Returns:
        List of object keys matching the prefix.
    """
    import s3fs

    fs = s3fs.S3FileSystem(**kwargs)
    path = f"{bucket}/{prefix}" if prefix else bucket
    return fs.ls(path)


@requires_extra("etl")
def read_gcs(
    bucket: str,
    blob: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a file from Google Cloud Storage into a DataFrame.

    Supports Parquet and CSV formats, determined by the file extension.
    Requires ``gcsfs`` (part of the ``etl`` extra).

    Parameters:
        bucket: GCS bucket name.
        blob: Blob path within the bucket.
        **kwargs: Additional keyword arguments forwarded to the
            underlying pandas reader.

    Returns:
        DataFrame with the file contents.
    """
    gcs_path = f"gs://{bucket}/{blob}"

    if blob.endswith(".parquet") or blob.endswith(".pq"):
        return pd.read_parquet(gcs_path, **kwargs)
    else:
        return pd.read_csv(gcs_path, **kwargs)


@requires_extra("etl")
def write_gcs(
    data: pd.DataFrame,
    bucket: str,
    blob: str,
    **kwargs: Any,
) -> None:
    """Write a DataFrame to Google Cloud Storage.

    Supports Parquet and CSV formats, determined by the file extension.
    Requires ``gcsfs`` (part of the ``etl`` extra).

    Parameters:
        data: DataFrame to write.
        bucket: GCS bucket name.
        blob: Blob path within the bucket.
        **kwargs: Additional keyword arguments forwarded to the
            underlying pandas writer.
    """
    gcs_path = f"gs://{bucket}/{blob}"

    if blob.endswith(".parquet") or blob.endswith(".pq"):
        data.to_parquet(gcs_path, **kwargs)
    else:
        data.to_csv(gcs_path, **kwargs)
