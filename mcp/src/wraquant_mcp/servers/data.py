"""Data management MCP tools.

Tools: load_csv, load_json, export_dataset, merge_datasets,
filter_dataset, rename_columns, resample_dataset.
"""

from __future__ import annotations

import json
from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_data_tools(mcp, ctx: AnalysisContext) -> None:
    """Register data management tools on the MCP server."""

    @mcp.tool()
    def load_csv(
        file_path: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Load a CSV file into the workspace DuckDB store.

        Reads the CSV into a pandas DataFrame and stores it as a named
        dataset available for all subsequent analysis tools.

        Parameters:
            file_path: Path to the CSV file on disk.
            name: Dataset name for referencing later.  If None, derived
                from the filename (e.g., "prices.csv" -> "prices").
        """
        from pathlib import Path

        import pandas as pd

        filepath = Path(file_path).expanduser().resolve()
        if not filepath.exists():
            return {"error": f"File not found: {filepath}"}

        df = pd.read_csv(filepath, parse_dates=True)

        # Try to set a datetime index if there's an obvious date column
        for col in df.columns:
            if col.lower() in ("date", "datetime", "timestamp", "time"):
                try:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                except Exception:
                    pass
                break

        if name is None:
            name = filepath.stem

        stored = ctx.store_dataset(name, df, source_op="load_csv")

        ctx._log("load_csv", name, file_path=str(filepath))

        return _sanitize_for_json({
            "tool": "load_csv",
            "file_path": str(filepath),
            **stored,
        })

    @mcp.tool()
    def load_json(
        data_json: str,
        name: str,
    ) -> dict[str, Any]:
        """Load inline JSON data as a dataset.

        Parses a JSON string into a DataFrame and stores it in the
        workspace.  Supports both array-of-objects and dict-of-arrays
        formats.

        Parameters:
            data_json: JSON string containing the data.  Accepted formats:
                - Array of objects: '[{"col1": 1, "col2": 2}, ...]'
                - Dict of arrays: '{"col1": [1, 2], "col2": [3, 4]}'
            name: Dataset name for referencing later.
        """
        import pandas as pd

        data = json.loads(data_json)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            return {"error": "JSON must be an array of objects or a dict of arrays"}

        stored = ctx.store_dataset(name, df, source_op="load_json")

        ctx._log("load_json", name)

        return _sanitize_for_json({
            "tool": "load_json",
            **stored,
        })

    @mcp.tool()
    def export_dataset(
        dataset: str,
        format: str = "csv",
        path: str | None = None,
    ) -> dict[str, Any]:
        """Export a dataset from the workspace to a file.

        Parameters:
            dataset: Name of the dataset to export.
            format: Output format ('csv', 'parquet', 'json').
            path: Output file path.  If None, saves to workspace dir.
        """
        from pathlib import Path as _Path

        df = ctx.get_dataset(dataset)

        if path is None:
            ext = {"csv": ".csv", "parquet": ".parquet", "json": ".json"}
            out_path = _Path(ctx.workspace_dir) / f"{dataset}{ext.get(format, '.csv')}"
        else:
            out_path = _Path(path).expanduser().resolve()

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            df.to_csv(out_path)
        elif format == "parquet":
            df.to_parquet(out_path)
        elif format == "json":
            df.to_json(out_path, orient="records", indent=2)
        else:
            return {"error": f"Unknown format '{format}'. Options: csv, parquet, json"}

        ctx._log("export_dataset", dataset, path=str(out_path), format=format)

        return _sanitize_for_json({
            "tool": "export_dataset",
            "dataset": dataset,
            "format": format,
            "path": str(out_path),
            "rows": len(df),
            "columns": list(df.columns),
        })

    @mcp.tool()
    def merge_datasets(
        dataset_a: str,
        dataset_b: str,
        on: str | None = None,
        how: str = "inner",
    ) -> dict[str, Any]:
        """Join two datasets together.

        Merges two datasets on a shared key column (or index if no key
        is specified) and stores the result as a new dataset.

        Parameters:
            dataset_a: Name of the left dataset.
            dataset_b: Name of the right dataset.
            on: Column name to join on.  If None, joins on the index.
            how: Join type ('inner', 'outer', 'left', 'right').
        """
        df_a = ctx.get_dataset(dataset_a)
        df_b = ctx.get_dataset(dataset_b)

        if on is not None:
            merged = df_a.merge(df_b, on=on, how=how, suffixes=("_a", "_b"))
        else:
            merged = df_a.merge(
                df_b, left_index=True, right_index=True,
                how=how, suffixes=("_a", "_b"),
            )

        result_name = f"{dataset_a}_{dataset_b}_merged"
        stored = ctx.store_dataset(
            result_name, merged,
            source_op="merge_datasets",
            parent=dataset_a,
        )

        ctx._log(
            "merge_datasets", result_name,
            dataset_a=dataset_a, dataset_b=dataset_b,
            on=on, how=how,
        )

        return _sanitize_for_json({
            "tool": "merge_datasets",
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "join_type": how,
            "join_key": on or "index",
            **stored,
        })

    @mcp.tool()
    def filter_dataset(
        dataset: str,
        condition_sql: str,
    ) -> dict[str, Any]:
        """Filter dataset rows using a SQL WHERE clause.

        Executes a SQL query against the dataset's DuckDB table,
        filtering rows that match the condition, and stores the
        result as a new dataset.

        Parameters:
            dataset: Name of the dataset to filter.
            condition_sql: SQL WHERE clause (without the WHERE keyword).
                e.g., "returns > 0.01 AND volume > 1000000"
        """
        query = f'SELECT * FROM "{dataset}" WHERE {condition_sql}'

        try:
            result_df = ctx.db.sql(query).df()
        except Exception as e:
            return {"error": f"SQL filter failed: {e}"}

        result_name = f"{dataset}_filtered"
        stored = ctx.store_dataset(
            result_name, result_df,
            source_op="filter_dataset",
            parent=dataset,
        )

        ctx._log(
            "filter_dataset", result_name,
            source=dataset, condition=condition_sql,
        )

        return _sanitize_for_json({
            "tool": "filter_dataset",
            "source_dataset": dataset,
            "condition": condition_sql,
            **stored,
        })

    @mcp.tool()
    def rename_columns(
        dataset: str,
        mapping_json: str,
    ) -> dict[str, Any]:
        """Rename columns in a dataset.

        Applies a column name mapping and stores the result as a new
        dataset.  Columns not in the mapping are left unchanged.

        Parameters:
            dataset: Name of the dataset.
            mapping_json: JSON string mapping old names to new names
                (e.g., '{"close": "price", "vol": "volume"}').
        """
        mapping = json.loads(mapping_json)

        df = ctx.get_dataset(dataset)
        renamed = df.rename(columns=mapping)

        result_name = f"{dataset}_renamed"
        stored = ctx.store_dataset(
            result_name, renamed,
            source_op="rename_columns",
            parent=dataset,
        )

        ctx._log("rename_columns", result_name, mapping=str(mapping))

        return _sanitize_for_json({
            "tool": "rename_columns",
            "source_dataset": dataset,
            "mapping": mapping,
            **stored,
        })

    @mcp.tool()
    def resample_dataset(
        dataset: str,
        freq: str = "W",
        method: str = "last",
    ) -> dict[str, Any]:
        """Resample a time series dataset to a different frequency.

        Requires the dataset to have a DatetimeIndex.  Common use case
        is converting daily data to weekly or monthly for longer-horizon
        analysis.

        Parameters:
            dataset: Name of the dataset.
            freq: Target frequency ('D', 'W', 'M', 'Q', 'Y').
            method: Aggregation method ('last', 'first', 'mean', 'sum',
                'ohlc').
        """
        import numpy as np
        import pandas as pd

        df = ctx.get_dataset(dataset)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in df.columns:
                if col.lower() in ("date", "datetime", "timestamp", "time"):
                    try:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        break
                    except Exception:
                        pass

        if not isinstance(df.index, pd.DatetimeIndex):
            return {"error": "Dataset does not have a DatetimeIndex. Cannot resample."}

        if method == "last":
            resampled = df.resample(freq).last()
        elif method == "first":
            resampled = df.resample(freq).first()
        elif method == "mean":
            resampled = df.resample(freq).mean()
        elif method == "sum":
            resampled = df.resample(freq).sum()
        elif method == "ohlc":
            numeric = df.select_dtypes(include=[np.number])
            resampled = numeric.resample(freq).ohlc()
            resampled.columns = ["_".join(c) for c in resampled.columns]
        else:
            return {"error": f"Unknown method: {method}. Use 'last', 'first', 'mean', 'sum', or 'ohlc'."}

        resampled = resampled.dropna(how="all")

        result_name = f"{dataset}_{freq}"
        stored = ctx.store_dataset(
            result_name, resampled,
            source_op="resample_dataset",
            parent=dataset,
        )

        ctx._log(
            "resample_dataset", result_name,
            source=dataset, freq=freq, method=method,
        )

        return _sanitize_for_json({
            "tool": "resample_dataset",
            "source_dataset": dataset,
            "freq": freq,
            "method": method,
            **stored,
        })
