"""Data management MCP tools.

Tools: load_csv, load_json, export_dataset, merge_datasets,
filter_dataset, rename_columns, resample_dataset,
fetch_yahoo, fetch_ohlcv, clean_dataset, validate_returns,
resample_ohlcv, align_datasets, compute_log_returns,
add_column, describe_dataset, split_dataset.
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
        try:
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

            return _sanitize_for_json(
                {
                    "tool": "load_csv",
                    "file_path": str(filepath),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "load_csv"}

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
        try:
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

            return _sanitize_for_json(
                {
                    "tool": "load_json",
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "load_json"}

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
        try:
            from pathlib import Path as _Path

            df = ctx.get_dataset(dataset)

            if path is None:
                ext = {"csv": ".csv", "parquet": ".parquet", "json": ".json"}
                out_path = (
                    _Path(ctx.workspace_dir) / f"{dataset}{ext.get(format, '.csv')}"
                )
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
                return {
                    "error": f"Unknown format '{format}'. Options: csv, parquet, json"
                }

            ctx._log("export_dataset", dataset, path=str(out_path), format=format)

            return _sanitize_for_json(
                {
                    "tool": "export_dataset",
                    "dataset": dataset,
                    "format": format,
                    "path": str(out_path),
                    "rows": len(df),
                    "columns": list(df.columns),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "export_dataset"}

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
        try:
            df_a = ctx.get_dataset(dataset_a)
            df_b = ctx.get_dataset(dataset_b)

            if on is not None:
                merged = df_a.merge(df_b, on=on, how=how, suffixes=("_a", "_b"))
            else:
                merged = df_a.merge(
                    df_b,
                    left_index=True,
                    right_index=True,
                    how=how,
                    suffixes=("_a", "_b"),
                )

            result_name = f"{dataset_a}_{dataset_b}_merged"
            stored = ctx.store_dataset(
                result_name,
                merged,
                source_op="merge_datasets",
                parent=dataset_a,
            )

            ctx._log(
                "merge_datasets",
                result_name,
                dataset_a=dataset_a,
                dataset_b=dataset_b,
                on=on,
                how=how,
            )

            return _sanitize_for_json(
                {
                    "tool": "merge_datasets",
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "join_type": how,
                    "join_key": on or "index",
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "merge_datasets"}

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
            result_name,
            result_df,
            source_op="filter_dataset",
            parent=dataset,
        )

        ctx._log(
            "filter_dataset",
            result_name,
            source=dataset,
            condition=condition_sql,
        )

        return _sanitize_for_json(
            {
                "tool": "filter_dataset",
                "source_dataset": dataset,
                "condition": condition_sql,
                **stored,
            }
        )

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
        try:
            mapping = json.loads(mapping_json)

            df = ctx.get_dataset(dataset)
            renamed = df.rename(columns=mapping)

            result_name = f"{dataset}_renamed"
            stored = ctx.store_dataset(
                result_name,
                renamed,
                source_op="rename_columns",
                parent=dataset,
            )

            ctx._log("rename_columns", result_name, mapping=str(mapping))

            return _sanitize_for_json(
                {
                    "tool": "rename_columns",
                    "source_dataset": dataset,
                    "mapping": mapping,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "rename_columns"}

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
        try:
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
                return {
                    "error": "Dataset does not have a DatetimeIndex. Cannot resample."
                }

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
                return {
                    "error": f"Unknown method: {method}. Use 'last', 'first', 'mean', 'sum', or 'ohlc'."
                }

            resampled = resampled.dropna(how="all")

            result_name = f"{dataset}_{freq}"
            stored = ctx.store_dataset(
                result_name,
                resampled,
                source_op="resample_dataset",
                parent=dataset,
            )

            ctx._log(
                "resample_dataset",
                result_name,
                source=dataset,
                freq=freq,
                method=method,
            )

            return _sanitize_for_json(
                {
                    "tool": "resample_dataset",
                    "source_dataset": dataset,
                    "freq": freq,
                    "method": method,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "resample_dataset"}

    # ------------------------------------------------------------------
    # Data fetching tools (wraquant.data)
    # ------------------------------------------------------------------

    @mcp.tool()
    def fetch_yahoo(
        ticker: str,
        start: str,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Fetch closing prices for a ticker from Yahoo Finance via wraquant.data.fetch_prices.

        Downloads daily close prices and stores them as a named dataset
        in the workspace DuckDB store.

        Parameters:
            ticker: Ticker symbol (e.g., 'AAPL', 'MSFT', 'BTC-USD').
            start: Start date string (e.g., '2020-01-01').
            end: End date string. If None, fetches up to today.
        """
        import pandas as pd

        from wraquant.data import fetch_prices

        try:
            prices = fetch_prices(ticker, start=start, end=end, source="yahoo")
        except Exception as e:
            return {"error": f"Failed to fetch prices for {ticker}: {e}"}

        df = pd.DataFrame({"close": prices})
        name = f"{ticker.lower().replace('-', '_')}_prices"
        stored = ctx.store_dataset(name, df, source_op="fetch_yahoo")

        ctx._log("fetch_yahoo", name, ticker=ticker, start=start, end=end)

        return _sanitize_for_json(
            {
                "tool": "fetch_yahoo",
                "ticker": ticker,
                "start": start,
                "end": end,
                **stored,
            }
        )

    @mcp.tool()
    def fetch_ohlcv(
        ticker: str,
        start: str,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Fetch OHLCV data for a ticker and store as an OHLCVFrame in DuckDB.

        Downloads daily OHLCV bars (open, high, low, close, volume) from
        Yahoo Finance and stores them for downstream analysis, backtesting,
        and technical analysis tools.

        Parameters:
            ticker: Ticker symbol (e.g., 'AAPL', 'SPY').
            start: Start date string (e.g., '2020-01-01').
            end: End date string. If None, fetches up to today.
        """
        from wraquant.data import fetch_ohlcv as _fetch_ohlcv

        try:
            ohlcv = _fetch_ohlcv(ticker, start=start, end=end, source="yahoo")
        except Exception as e:
            return {"error": f"Failed to fetch OHLCV for {ticker}: {e}"}

        name = f"{ticker.lower().replace('-', '_')}_ohlcv"
        stored = ctx.store_dataset(name, ohlcv, source_op="fetch_ohlcv")

        ctx._log("fetch_ohlcv", name, ticker=ticker, start=start, end=end)

        return _sanitize_for_json(
            {
                "tool": "fetch_ohlcv",
                "ticker": ticker,
                "start": start,
                "end": end,
                **stored,
            }
        )

    # ------------------------------------------------------------------
    # Data cleaning / validation tools
    # ------------------------------------------------------------------

    @mcp.tool()
    def clean_dataset(
        dataset: str,
        method: str = "dropna",
    ) -> dict[str, Any]:
        """Clean a dataset by handling missing values, outliers, and gaps.

        Applies cleaning operations from wraquant.data.cleaning and stores
        the cleaned result as a new dataset.

        Parameters:
            dataset: Name of the dataset to clean.
            method: Cleaning method to apply:
                - 'dropna' — drop rows with any NaN values
                - 'ffill' — forward-fill missing values
                - 'bfill' — back-fill missing values
                - 'interpolate' — linear interpolation of gaps
                - 'outliers_zscore' — remove outliers via z-score (>3σ)
                - 'outliers_iqr' — remove outliers via IQR method
                - 'winsorize' — clip extreme values at 1st/99th percentile
        """
        try:
            from wraquant.data.cleaning import (
                fill_missing,
                remove_outliers,
            )
            from wraquant.data.cleaning import winsorize as _winsorize

            df = ctx.get_dataset(dataset)
            original_rows = len(df)

            if method == "dropna":
                cleaned = df.dropna()
            elif method in ("ffill", "bfill", "interpolate"):
                cleaned = fill_missing(df, method=method)
            elif method == "outliers_zscore":
                cleaned = remove_outliers(df, method="zscore", threshold=3.0)
            elif method == "outliers_iqr":
                cleaned = remove_outliers(df, method="iqr", threshold=1.5)
            elif method == "winsorize":
                cleaned = _winsorize(df, limits=(0.01, 0.01))
            else:
                return {
                    "error": f"Unknown method '{method}'. Options: dropna, ffill, "
                    "bfill, interpolate, outliers_zscore, outliers_iqr, winsorize",
                }

            result_name = f"{dataset}_cleaned"
            stored = ctx.store_dataset(
                result_name,
                cleaned,
                source_op="clean_dataset",
                parent=dataset,
            )

            ctx._log(
                "clean_dataset",
                result_name,
                source=dataset,
                method=method,
            )

            return _sanitize_for_json(
                {
                    "tool": "clean_dataset",
                    "source_dataset": dataset,
                    "method": method,
                    "rows_before": original_rows,
                    "rows_after": len(cleaned),
                    "rows_removed": original_rows - len(cleaned),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "clean_dataset"}

    @mcp.tool()
    def validate_returns_tool(
        dataset: str,
        column: str = "returns",
    ) -> dict[str, Any]:
        """Validate a return series for suspicious values (e.g. >50% daily moves).

        Checks for extreme returns, NaN values, and other data quality
        issues using wraquant.data.validate_returns.

        Parameters:
            dataset: Name of the dataset containing the returns.
            column: Column name with the return series to validate.
        """
        try:
            from wraquant.data.validation import validate_returns as _validate_returns

            df = ctx.get_dataset(dataset)

            if column not in df.columns:
                return {
                    "error": f"Column '{column}' not found. Available: {list(df.columns)}"
                }

            returns = df[column].dropna()
            result = _validate_returns(returns, max_abs=0.5)

            ctx._log(
                "validate_returns",
                dataset,
                column=column,
            )

            return _sanitize_for_json(
                {
                    "tool": "validate_returns",
                    "dataset": dataset,
                    "column": column,
                    "observations": len(returns),
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "validate_returns_tool"}

    @mcp.tool()
    def resample_ohlcv(
        dataset: str,
        freq: str = "W",
    ) -> dict[str, Any]:
        """Resample OHLCV data using proper financial conventions.

        Aggregates OHLCV data to a lower frequency using standard rules:
        open=first, high=max, low=min, close=last, volume=sum.

        Parameters:
            dataset: Name of the OHLCV dataset.
            freq: Target frequency ('W' weekly, 'M' monthly, 'Q' quarterly).
        """
        import pandas as pd

        from wraquant.data.cleaning import resample_ohlcv as _resample_ohlcv

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

        try:
            resampled = _resample_ohlcv(df, freq=freq)
        except Exception as e:
            return {
                "error": f"Resample failed: {e}. Ensure dataset has open/high/low/close/volume columns."
            }

        result_name = f"{dataset}_ohlcv_{freq}"
        stored = ctx.store_dataset(
            result_name,
            resampled,
            source_op="resample_ohlcv",
            parent=dataset,
        )

        ctx._log(
            "resample_ohlcv",
            result_name,
            source=dataset,
            freq=freq,
        )

        return _sanitize_for_json(
            {
                "tool": "resample_ohlcv",
                "source_dataset": dataset,
                "freq": freq,
                **stored,
            }
        )

    @mcp.tool()
    def align_datasets(
        datasets_json: str,
    ) -> dict[str, Any]:
        """Align multiple datasets to a common date range (inner join on index).

        Takes two or more datasets and aligns them so they share the same
        DatetimeIndex, keeping only dates present in all datasets.

        Parameters:
            datasets_json: JSON array of dataset names to align
                (e.g., '["aapl_prices", "msft_prices", "spy_prices"]').
        """
        try:
            import pandas as pd

            dataset_names = json.loads(datasets_json)

            if len(dataset_names) < 2:
                return {"error": "At least two datasets are required for alignment."}

            dfs = {}
            for name in dataset_names:
                try:
                    dfs[name] = ctx.get_dataset(name)
                except KeyError:
                    return {"error": f"Dataset '{name}' not found."}

            # Find common date range via inner join
            combined = pd.concat(
                [df.add_suffix(f"_{name}") for name, df in dfs.items()],
                axis=1,
                join="inner",
            )

            # Store individual aligned datasets and the combined one
            results = {}
            for name, df in dfs.items():
                aligned = df.reindex(combined.index).dropna(how="all")
                aligned_name = f"{name}_aligned"
                stored = ctx.store_dataset(
                    aligned_name,
                    aligned,
                    source_op="align_datasets",
                    parent=name,
                )
                results[aligned_name] = stored

            # Also store the combined dataset
            combined_name = "_".join(dataset_names[:3]) + "_aligned"
            combined_stored = ctx.store_dataset(
                combined_name,
                combined,
                source_op="align_datasets",
            )

            ctx._log(
                "align_datasets",
                combined_name,
                sources=str(dataset_names),
            )

            return _sanitize_for_json(
                {
                    "tool": "align_datasets",
                    "source_datasets": dataset_names,
                    "combined_dataset": combined_name,
                    "aligned_datasets": list(results.keys()),
                    "common_rows": len(combined),
                    **combined_stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "align_datasets"}

    @mcp.tool()
    def compute_log_returns(
        dataset: str,
        column: str = "close",
    ) -> dict[str, Any]:
        """Compute log returns from a price column.

        Calculates ln(P_t / P_{t-1}) and stores the result as a new
        dataset.  Log returns are additive across time and more suitable
        for statistical modeling than simple returns.

        Parameters:
            dataset: Name of the dataset containing prices.
            column: Price column name to compute returns from.
        """
        try:
            import pandas as pd

            from wraquant.data.transforms import to_returns

            df = ctx.get_dataset(dataset)

            if column not in df.columns:
                return {
                    "error": f"Column '{column}' not found. Available: {list(df.columns)}"
                }

            prices = df[column]
            log_returns = to_returns(prices, method="log")

            result_df = pd.DataFrame({"log_returns": log_returns})
            result_name = f"{dataset}_log_returns"
            stored = ctx.store_dataset(
                result_name,
                result_df,
                source_op="compute_log_returns",
                parent=dataset,
            )

            clean = log_returns.dropna()

            ctx._log(
                "compute_log_returns",
                result_name,
                source=dataset,
                column=column,
            )

            return _sanitize_for_json(
                {
                    "tool": "compute_log_returns",
                    "source_dataset": dataset,
                    "column": column,
                    "mean_log_return": float(clean.mean()) if len(clean) > 0 else None,
                    "std_log_return": float(clean.std()) if len(clean) > 0 else None,
                    "annualized_vol": (
                        float(clean.std() * (252**0.5)) if len(clean) > 0 else None
                    ),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "compute_log_returns"}

    @mcp.tool()
    def add_column(
        dataset: str,
        column_name: str,
        expression: str,
    ) -> dict[str, Any]:
        """Add a computed column to a dataset via a SQL expression.

        Uses DuckDB SQL to evaluate the expression against the dataset,
        adding the result as a new column.

        Parameters:
            dataset: Name of the dataset.
            column_name: Name for the new column.
            expression: SQL expression to compute the column value
                (e.g., '(close - open) / open' for intraday return,
                'volume * close' for dollar volume).
        """
        try:
            query = f'SELECT *, ({expression}) AS "{column_name}" FROM "{dataset}"'
            result_df = ctx.db.sql(query).df()
        except Exception as e:
            return {"error": f"SQL expression failed: {e}"}

        result_name = f"{dataset}_ext"
        stored = ctx.store_dataset(
            result_name,
            result_df,
            source_op="add_column",
            parent=dataset,
        )

        ctx._log(
            "add_column",
            result_name,
            source=dataset,
            column_name=column_name,
            expression=expression,
        )

        return _sanitize_for_json(
            {
                "tool": "add_column",
                "source_dataset": dataset,
                "column_name": column_name,
                "expression": expression,
                **stored,
            }
        )

    @mcp.tool()
    def describe_dataset(
        dataset: str,
    ) -> dict[str, Any]:
        """Generate detailed statistics about a dataset.

        Reports missing values, date range, inferred frequency, outlier
        counts, summary statistics, and data types for each column.

        Parameters:
            dataset: Name of the dataset to describe.
        """
        try:
            import numpy as np
            import pandas as pd

            from wraquant.data.cleaning import detect_outliers

            df = ctx.get_dataset(dataset)

            # Basic shape info
            info: dict[str, Any] = {
                "tool": "describe_dataset",
                "dataset": dataset,
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
            }

            # Missing values
            missing = {c: int(df[c].isna().sum()) for c in df.columns}
            info["missing_values"] = missing
            info["total_missing"] = sum(missing.values())

            # Date range and frequency
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
                info["date_range"] = {
                    "start": df.index.min().isoformat(),
                    "end": df.index.max().isoformat(),
                }
                freq = pd.infer_freq(df.index)
                info["inferred_frequency"] = freq or "irregular"
            else:
                info["date_range"] = None
                info["inferred_frequency"] = None

            # Summary statistics for numeric columns
            numeric = df.select_dtypes(include=[np.number])
            if len(numeric.columns) > 0 and len(numeric) > 0:
                desc = numeric.describe()
                info["statistics"] = desc.to_dict()

                # Outlier detection
                try:
                    outlier_mask = detect_outliers(
                        numeric, method="zscore", threshold=3.0
                    )
                    info["outlier_rows_zscore_3sigma"] = int(outlier_mask.sum())
                except Exception:
                    info["outlier_rows_zscore_3sigma"] = None

            ctx._log("describe_dataset", dataset)

            return _sanitize_for_json(info)
        except Exception as e:
            return {"error": str(e), "tool": "describe_dataset"}

    @mcp.tool()
    def split_dataset(
        dataset: str,
        split_date: str,
    ) -> dict[str, Any]:
        """Split a dataset into train and test sets at a given date.

        Splits the dataset into two subsets: rows before the split date
        (train) and rows on or after the split date (test).  Both subsets
        are stored as new datasets.

        Parameters:
            dataset: Name of the dataset to split.
            split_date: Date string to split on (e.g., '2023-01-01').
                Rows before this date go to train; rows on or after go
                to test.
        """
        try:
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
                return {
                    "error": "Dataset does not have a DatetimeIndex. Cannot split by date."
                }

            cutoff = pd.Timestamp(split_date)
            train = df[df.index < cutoff]
            test = df[df.index >= cutoff]

            train_name = f"{dataset}_train"
            test_name = f"{dataset}_test"

            train_stored = ctx.store_dataset(
                train_name,
                train,
                source_op="split_dataset",
                parent=dataset,
            )
            test_stored = ctx.store_dataset(
                test_name,
                test,
                source_op="split_dataset",
                parent=dataset,
            )

            ctx._log(
                "split_dataset",
                dataset,
                split_date=split_date,
                train_rows=len(train),
                test_rows=len(test),
            )

            return _sanitize_for_json(
                {
                    "tool": "split_dataset",
                    "source_dataset": dataset,
                    "split_date": split_date,
                    "train_dataset": train_name,
                    "train_rows": len(train),
                    "test_dataset": test_name,
                    "test_rows": len(test),
                    "train_pct": round(len(train) / max(len(df), 1) * 100, 1),
                    "test_pct": round(len(test) / max(len(df), 1) * 100, 1),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "split_dataset"}
