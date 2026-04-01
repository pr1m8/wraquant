"""Data management MCP tools.

Tools: load_csv, export_dataset, dataset_summary.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_data_tools(mcp, ctx: AnalysisContext) -> None:
    """Register data management tools on the MCP server."""

    @mcp.tool()
    def load_csv(
        path: str,
        name: str | None = None,
        parse_dates: bool = True,
        index_col: int | str | None = 0,
    ) -> dict[str, Any]:
        """Load a CSV file into the workspace as a named dataset.

        Parameters:
            path: Path to the CSV file.
            name: Dataset name. If None, derived from filename.
            parse_dates: Whether to parse date columns.
            index_col: Column to use as index (0 = first column,
                None = no index column).
        """
        import os
        from pathlib import Path

        import pandas as pd

        filepath = Path(path).expanduser().resolve()
        if not filepath.exists():
            return {"error": f"File not found: {filepath}"}

        df = pd.read_csv(
            filepath,
            parse_dates=parse_dates,
            index_col=index_col,
        )

        if name is None:
            name = filepath.stem

        stored = ctx.store_dataset(
            name, df,
            source_op="load_csv",
        )

        return _sanitize_for_json({
            "tool": "load_csv",
            "path": str(filepath),
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
            path: Output file path. If None, saves to workspace dir.
        """
        from pathlib import Path

        df = ctx.get_dataset(dataset)

        if path is None:
            ext = {"csv": ".csv", "parquet": ".parquet", "json": ".json"}
            path = str(ctx.workspace_dir / f"{dataset}{ext.get(format, '.csv')}")

        out_path = Path(path).expanduser().resolve()

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
    def dataset_summary(
        dataset: str,
    ) -> dict[str, Any]:
        """Get a comprehensive summary of a dataset.

        Returns shape, dtypes, descriptive statistics, head/tail,
        missing values, and date range (if applicable).

        Parameters:
            dataset: Name of the dataset to summarize.
        """
        import numpy as np
        import pandas as pd

        df = ctx.get_dataset(dataset)

        summary: dict[str, Any] = {
            "tool": "dataset_summary",
            "dataset": dataset,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        }

        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) > 0:
            summary["statistics"] = numeric.describe().to_dict()

        summary["head"] = df.head(3).to_dict(orient="records")
        summary["tail"] = df.tail(3).to_dict(orient="records")

        if isinstance(df.index, pd.DatetimeIndex):
            summary["date_range"] = {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "periods": len(df.index),
            }

        return _sanitize_for_json(summary)
