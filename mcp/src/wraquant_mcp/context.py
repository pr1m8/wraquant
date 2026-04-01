"""AnalysisContext — DuckDB-backed state manager for wraquant-mcp.

All tabular data lives in DuckDB (zero-copy pandas registration).
Fitted models are stored as joblib files. An append-only journal
tracks every operation for lineage and undo.

The context is shared across all MCP tool calls within a session.
Multiple MCP servers (wraquant + DuckDB MCP) can share the same
DuckDB file for seamless composition.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wraquant_mcp.ids import IDRegistry


class AnalysisContext:
    """Server-side state for an MCP session.

    Manages datasets (DuckDB), models (joblib), and operation
    journal (JSONL) within a workspace directory.

    Parameters:
        workspace_dir: Path to workspace directory.
            Created if it doesn't exist.
    """

    def __init__(self, workspace_dir: str | Path | None = None) -> None:
        import duckdb

        if workspace_dir is None:
            workspace_dir = Path.home() / ".wraquant" / "workspaces" / "default"

        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "models").mkdir(exist_ok=True)
        (self.workspace_dir / "notebooks").mkdir(exist_ok=True)
        (self.workspace_dir / "snapshots").mkdir(exist_ok=True)

        # DuckDB for tabular data
        db_path = str(self.workspace_dir / "data.duckdb")
        self.db = duckdb.connect(db_path)

        # ID registry
        self.registry = IDRegistry()

        # Journal
        self._journal_path = self.workspace_dir / "journal.jsonl"

        # Manifest
        self._manifest_path = self.workspace_dir / "manifest.json"
        self._load_or_create_manifest()

        # In-memory model store
        self._models: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def store_dataset(
        self,
        name: str,
        df: pd.DataFrame,
        source_op: str | None = None,
        parent: str | None = None,
    ) -> dict[str, Any]:
        """Store a DataFrame as a named DuckDB table.

        Returns metadata dict (never raw data).
        """
        # Register in DuckDB (zero-copy for pandas)
        actual_name = self.registry.register(
            name,
            "dataset",
            source_op=source_op,
            parent=parent,
            rows=len(df),
            columns=list(df.columns),
            dtypes={c: str(df[c].dtype) for c in df.columns},
        )

        self.db.register(actual_name, df)

        # Log
        self._log("store_dataset", actual_name, source_op=source_op, parent=parent)

        return {
            "dataset_id": actual_name,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        }

    def get_dataset(self, name: str) -> pd.DataFrame:
        """Retrieve a DataFrame by name from DuckDB."""
        try:
            return self.db.sql(f'SELECT * FROM "{name}"').df()
        except Exception as e:
            raise KeyError(f"Dataset '{name}' not found: {e}") from e

    def dataset_exists(self, name: str) -> bool:
        """Check if a dataset exists in DuckDB."""
        try:
            self.db.sql(f'SELECT 1 FROM "{name}" LIMIT 1')
            return True
        except Exception:
            return False

    def list_datasets(self) -> list[str]:
        """List all dataset names in DuckDB."""
        tables = self.db.sql("SHOW TABLES").fetchall()
        return [t[0] for t in tables]

    def dataset_info(self, name: str) -> dict[str, Any]:
        """Get metadata about a dataset."""
        df = self.get_dataset(name)
        meta = self.registry.get(name)
        return {
            "dataset_id": name,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "head": _sanitize_for_json(df.head(3).to_dict(orient="records")),
            "stats": _sanitize_for_json(
                df.describe().to_dict()
            ) if len(df) > 0 else {},
            "source_op": meta.source_op if meta else None,
            "parent": meta.parent if meta else None,
            "lineage": self.registry.lineage(name),
        }

    # ------------------------------------------------------------------
    # Model operations
    # ------------------------------------------------------------------

    def store_model(
        self,
        name: str,
        model: Any,
        model_type: str = "",
        source_dataset: str = "",
        metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Store a fitted model (in memory + optionally to disk)."""
        actual_name = self.registry.register(
            name,
            "model",
            source_op=f"fit_{model_type}",
            model_type=model_type,
            source_dataset=source_dataset,
            metrics=metrics or {},
        )

        self._models[actual_name] = model

        # Persist to disk via joblib
        try:
            import joblib

            model_path = self.workspace_dir / "models" / f"{actual_name}.joblib"
            joblib.dump(model, model_path)
        except Exception:
            pass  # In-memory only if joblib fails

        self._log(
            "store_model",
            actual_name,
            model_type=model_type,
            source_dataset=source_dataset,
        )

        summary = {}
        if hasattr(model, "summary"):
            try:
                summary = {"summary": str(model.summary())}
            except Exception:
                pass
        if metrics:
            summary["metrics"] = _sanitize_for_json(metrics)

        return {
            "model_id": actual_name,
            "model_type": model_type,
            "source_dataset": source_dataset,
            **summary,
        }

    def get_model(self, name: str) -> Any:
        """Retrieve a fitted model by name."""
        if name in self._models:
            return self._models[name]

        # Try loading from disk
        try:
            import joblib

            model_path = self.workspace_dir / "models" / f"{name}.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
                self._models[name] = model
                return model
        except Exception:
            pass

        raise KeyError(f"Model '{name}' not found")

    def list_models(self) -> list[str]:
        """List all model names."""
        return self.registry.list_models()

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------

    def _log(self, operation: str, resource: str, **kwargs: Any) -> None:
        """Append an entry to the operation journal."""
        entry = {
            "ts": datetime.now().isoformat(),
            "op": operation,
            "resource": resource,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        with open(self._journal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def history(self, n: int = 20) -> list[dict[str, Any]]:
        """Read the last N journal entries."""
        if not self._journal_path.exists():
            return []
        with open(self._journal_path) as f:
            lines = f.readlines()
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        return entries

    # ------------------------------------------------------------------
    # Workspace management
    # ------------------------------------------------------------------

    def _load_or_create_manifest(self) -> None:
        """Load or create workspace manifest."""
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                self._manifest = json.load(f)
        else:
            self._manifest = {
                "workspace": self.workspace_dir.name,
                "created": datetime.now().isoformat(),
                "last_session": datetime.now().isoformat(),
            }
            self._save_manifest()

    def _save_manifest(self) -> None:
        """Save workspace manifest."""
        self._manifest["last_session"] = datetime.now().isoformat()
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def workspace_status(self) -> dict[str, Any]:
        """Full workspace status for the agent."""
        return {
            "workspace": self.workspace_dir.name,
            "path": str(self.workspace_dir),
            "datasets": self.list_datasets(),
            "models": self.list_models(),
            "journal_entries": len(self.history(n=10000)),
            "last_session": self._manifest.get("last_session"),
            **self.registry.summary(),
        }

    def add_note(self, text: str) -> dict[str, str]:
        """Add a research note to the journal."""
        self._log("note", "user", text=text)
        return {"status": "noted", "text": text}

    def close(self) -> None:
        """Close the DuckDB connection and save manifest."""
        self._save_manifest()
        self.db.close()


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _sanitize_for_json(obj.to_dict(orient="list"))
    if isinstance(obj, pd.Series):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, pd.Index):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass
    return obj
