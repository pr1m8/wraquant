"""Workspace management tools for wraquant-mcp.

Create, open, list, snapshot, and restore workspaces.
Each workspace is an isolated research environment with its own
DuckDB database, models, journal, and notebooks.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from wraquant_mcp.context import AnalysisContext


WORKSPACES_DIR = Path.home() / ".wraquant" / "workspaces"


def register_workspace_tools(mcp: Any, ctx_holder: list) -> None:
    """Register workspace management tools.

    ctx_holder is a mutable list containing [ctx] so tools can
    swap the active context when opening a different workspace.
    """

    @mcp.tool()
    def create_workspace(name: str, description: str = "") -> dict[str, Any]:
        """Create a new research workspace.

        Each workspace is an isolated environment with its own datasets,
        models, and journal. Use workspaces to organize research by topic.

        Parameters:
            name: Workspace name (alphanumeric + underscores).
            description: Optional description of the research goal.
        """
        workspace_dir = WORKSPACES_DIR / name
        if workspace_dir.exists():
            return {"error": f"Workspace '{name}' already exists. Use open_workspace."}

        # Create new context (creates directory + DuckDB)
        new_ctx = AnalysisContext(workspace_dir)
        if description:
            new_ctx.add_note(f"Workspace created: {description}")

        # Swap active context
        ctx_holder[0] = new_ctx

        return {
            "workspace": name,
            "path": str(workspace_dir),
            "status": "created",
            "description": description,
        }

    @mcp.tool()
    def open_workspace(name: str) -> dict[str, Any]:
        """Open an existing workspace, restoring all datasets and models.

        Parameters:
            name: Workspace name to open.
        """
        workspace_dir = WORKSPACES_DIR / name
        if not workspace_dir.exists():
            available = list_workspaces()
            return {
                "error": f"Workspace '{name}' not found.",
                "available": available.get("workspaces", []),
            }

        # Close current context
        ctx_holder[0].close()

        # Open new context
        new_ctx = AnalysisContext(workspace_dir)
        ctx_holder[0] = new_ctx

        return new_ctx.workspace_status()

    @mcp.tool()
    def list_workspaces() -> dict[str, Any]:
        """List all available workspaces with metadata."""
        WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
        workspaces = []
        for d in sorted(WORKSPACES_DIR.iterdir()):
            if d.is_dir():
                manifest_path = d / "manifest.json"
                info = {"name": d.name}
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest = json.load(f)
                        info["last_session"] = manifest.get("last_session", "unknown")
                        info["created"] = manifest.get("created", "unknown")
                    except Exception:
                        pass
                # Count datasets
                db_path = d / "data.duckdb"
                if db_path.exists():
                    try:
                        import duckdb

                        con = duckdb.connect(str(db_path), read_only=True)
                        tables = con.sql("SHOW TABLES").fetchall()
                        info["datasets"] = len(tables)
                        con.close()
                    except Exception:
                        info["datasets"] = 0
                # Count models
                models_dir = d / "models"
                if models_dir.exists():
                    info["models"] = len(list(models_dir.glob("*.joblib")))
                else:
                    info["models"] = 0
                workspaces.append(info)
        return {"workspaces": workspaces, "count": len(workspaces)}

    @mcp.tool()
    def snapshot(name: str | None = None) -> dict[str, Any]:
        """Create a named checkpoint of the current workspace.

        Saves a copy of the DuckDB database and models that can be
        restored later with restore_snapshot.

        Parameters:
            name: Snapshot name. Auto-generated if not provided.
        """
        ctx = ctx_holder[0]
        if name is None:
            name = datetime.now().strftime("snap_%Y%m%d_%H%M%S")

        snap_dir = ctx.workspace_dir / "snapshots" / name
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Copy DuckDB
        src_db = ctx.workspace_dir / "data.duckdb"
        if src_db.exists():
            shutil.copy2(src_db, snap_dir / "data.duckdb")

        # Copy models
        models_src = ctx.workspace_dir / "models"
        if models_src.exists():
            models_dst = snap_dir / "models"
            if models_dst.exists():
                shutil.rmtree(models_dst)
            shutil.copytree(models_src, models_dst)

        # Save manifest
        manifest = {
            "snapshot": name,
            "created": datetime.now().isoformat(),
            "datasets": ctx.list_datasets(),
            "models": ctx.list_models(),
        }
        with open(snap_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        ctx._log("snapshot", name)

        return {"snapshot": name, "path": str(snap_dir), "status": "created"}

    @mcp.tool()
    def restore_snapshot(name: str) -> dict[str, Any]:
        """Restore workspace to a previous snapshot.

        Parameters:
            name: Snapshot name to restore.
        """
        ctx = ctx_holder[0]
        snap_dir = ctx.workspace_dir / "snapshots" / name

        if not snap_dir.exists():
            snaps = [
                d.name for d in (ctx.workspace_dir / "snapshots").iterdir()
                if d.is_dir()
            ] if (ctx.workspace_dir / "snapshots").exists() else []
            return {"error": f"Snapshot '{name}' not found.", "available": snaps}

        # Close current DB
        ctx.db.close()

        # Restore DuckDB
        snap_db = snap_dir / "data.duckdb"
        if snap_db.exists():
            shutil.copy2(snap_db, ctx.workspace_dir / "data.duckdb")

        # Restore models
        snap_models = snap_dir / "models"
        if snap_models.exists():
            dst = ctx.workspace_dir / "models"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(snap_models, dst)

        # Reopen context
        import duckdb

        ctx.db = duckdb.connect(str(ctx.workspace_dir / "data.duckdb"))
        ctx._log("restore_snapshot", name)

        return {"snapshot": name, "status": "restored", **ctx.workspace_status()}

    @mcp.tool()
    def delete_workspace(name: str) -> dict[str, Any]:
        """Delete a workspace and all its data.

        Parameters:
            name: Workspace name to delete.
        """
        workspace_dir = WORKSPACES_DIR / name
        if not workspace_dir.exists():
            return {"error": f"Workspace '{name}' not found."}

        # Don't delete active workspace
        if ctx_holder[0].workspace_dir == workspace_dir:
            return {"error": "Cannot delete active workspace. Open a different one first."}

        shutil.rmtree(workspace_dir)
        return {"workspace": name, "status": "deleted"}

    @mcp.tool()
    def query_data(sql: str) -> dict[str, Any]:
        """Run a SQL query against the workspace DuckDB.

        Use this to inspect, filter, or aggregate stored datasets.
        All datasets are tables in DuckDB.

        Parameters:
            sql: SQL query (SELECT only for safety).

        Example:
            query_data("SELECT date, close, rsi FROM prices_aapl_rsi WHERE rsi < 30")
        """
        ctx = ctx_holder[0]
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT") and not sql_upper.startswith("SHOW") and not sql_upper.startswith("DESCRIBE"):
            return {"error": "Only SELECT, SHOW, and DESCRIBE queries are allowed."}

        try:
            result = ctx.db.sql(sql).df()
            from wraquant_mcp.context import _sanitize_for_json

            return {
                "rows": len(result),
                "columns": list(result.columns),
                "data": _sanitize_for_json(result.head(50).to_dict(orient="records")),
            }
        except Exception as e:
            return {"error": str(e)}
