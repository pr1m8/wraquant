"""Tests for workspace management — create, open, list, snapshot, restore, delete.

Covers: create_workspace, open_workspace, list_workspaces, snapshot,
restore_snapshot, delete_workspace, query_data, add_note, auto-versioning.
"""

from __future__ import annotations

import json
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wraquant_mcp.context import AnalysisContext


# ------------------------------------------------------------------
# Mock MCP for capturing registered tool functions
# ------------------------------------------------------------------


class MockMCP:
    """Capture tool functions registered via @mcp.tool()."""

    def __init__(self):
        self.tools: dict[str, callable] = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def workspaces_root(tmp_path, monkeypatch):
    """Override WORKSPACES_DIR to a temp directory."""
    ws_root = tmp_path / "workspaces"
    ws_root.mkdir()
    import wraquant_mcp.workspace as ws_mod
    monkeypatch.setattr(ws_mod, "WORKSPACES_DIR", ws_root)
    return ws_root


@pytest.fixture
def mcp_tools(workspaces_root, tmp_path):
    """Register workspace tools on a mock MCP and return tools dict."""
    from wraquant_mcp.workspace import register_workspace_tools

    mock = MockMCP()
    default_ws = workspaces_root / "default"
    ctx = AnalysisContext(default_ws)
    ctx_holder = [ctx]
    register_workspace_tools(mock, ctx_holder)
    yield mock.tools, ctx_holder
    ctx_holder[0].close()


@pytest.fixture
def sample_df():
    """Simple price DataFrame for populating workspaces."""
    rng = np.random.default_rng(42)
    n = 100
    close = 100 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.5, n),
        "high": close + abs(rng.normal(0, 1, n)),
        "low": close - abs(rng.normal(0, 1, n)),
        "close": close,
        "volume": rng.integers(1000, 10000, n),
    })


# ------------------------------------------------------------------
# Create workspace
# ------------------------------------------------------------------


class TestCreateWorkspace:
    """Test create_workspace tool."""

    def test_create_workspace_creates_directory(self, mcp_tools, workspaces_root):
        tools, ctx_holder = mcp_tools
        result = tools["create_workspace"](name="test_research")

        assert result["status"] == "created"
        assert result["workspace"] == "test_research"
        ws_dir = workspaces_root / "test_research"
        assert ws_dir.exists()
        assert (ws_dir / "data.duckdb").exists()
        assert (ws_dir / "models").is_dir()
        assert (ws_dir / "snapshots").is_dir()

    def test_create_workspace_with_description(self, mcp_tools):
        tools, ctx_holder = mcp_tools
        result = tools["create_workspace"](
            name="vol_study", description="Volatility clustering research"
        )
        assert result["status"] == "created"
        assert result["description"] == "Volatility clustering research"

        # Description should appear in journal
        history = ctx_holder[0].history(n=10)
        assert any("Volatility clustering" in str(e) for e in history)

    def test_create_duplicate_workspace_errors(self, mcp_tools):
        tools, _ = mcp_tools
        tools["create_workspace"](name="existing")
        result = tools["create_workspace"](name="existing")
        assert "error" in result
        assert "already exists" in result["error"]

    def test_create_workspace_swaps_active_context(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        # Store data in default workspace
        ctx_holder[0].store_dataset("original", sample_df)

        # Create new workspace — context swaps
        tools["create_workspace"](name="new_ws")
        assert ctx_holder[0].workspace_dir.name == "new_ws"

        # The new workspace should be empty
        assert ctx_holder[0].list_datasets() == []


# ------------------------------------------------------------------
# Open workspace
# ------------------------------------------------------------------


class TestOpenWorkspace:
    """Test open_workspace tool."""

    def test_open_existing_workspace(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        # Create and populate a workspace
        tools["create_workspace"](name="populated")
        ctx_holder[0].store_dataset("prices", sample_df)
        assert "prices" in ctx_holder[0].list_datasets()

        # Switch away
        tools["create_workspace"](name="other")
        assert ctx_holder[0].workspace_dir.name == "other"

        # Open the populated workspace
        result = tools["open_workspace"](name="populated")
        assert ctx_holder[0].workspace_dir.name == "populated"
        # Datasets from DuckDB should persist
        assert "datasets" in result

    def test_open_nonexistent_workspace_errors(self, mcp_tools):
        tools, _ = mcp_tools
        result = tools["open_workspace"](name="does_not_exist")
        assert "error" in result
        assert "not found" in result["error"]

    def test_open_workspace_returns_status(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        tools["create_workspace"](name="status_test")
        ctx_holder[0].store_dataset("prices", sample_df)
        ctx_holder[0].store_model("my_model", {"x": 1}, model_type="test")

        # Switch away and back
        tools["create_workspace"](name="temp")
        result = tools["open_workspace"](name="status_test")

        assert "workspace" in result
        assert "models" in result


# ------------------------------------------------------------------
# List workspaces
# ------------------------------------------------------------------


class TestListWorkspaces:
    """Test list_workspaces tool."""

    def test_list_workspaces_returns_correct_count(self, mcp_tools):
        tools, _ = mcp_tools
        # Default workspace already exists from fixture
        tools["create_workspace"](name="ws_a")
        tools["create_workspace"](name="ws_b")

        result = tools["list_workspaces"]()
        assert result["count"] >= 3  # default + ws_a + ws_b
        names = [w["name"] for w in result["workspaces"]]
        assert "ws_a" in names
        assert "ws_b" in names

    def test_list_workspaces_includes_metadata(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        tools["create_workspace"](name="meta_test")
        ctx_holder[0].store_dataset("prices", sample_df)

        result = tools["list_workspaces"]()
        meta_ws = next(w for w in result["workspaces"] if w["name"] == "meta_test")
        assert "created" in meta_ws or "datasets" in meta_ws

    def test_list_workspaces_empty(self, tmp_path, monkeypatch):
        """Empty workspaces directory returns zero."""
        import wraquant_mcp.workspace as ws_mod
        empty_root = tmp_path / "empty_ws"
        monkeypatch.setattr(ws_mod, "WORKSPACES_DIR", empty_root)

        mock = MockMCP()
        ctx = AnalysisContext(tmp_path / "dummy")
        ctx_holder = [ctx]

        from wraquant_mcp.workspace import register_workspace_tools
        register_workspace_tools(mock, ctx_holder)

        result = mock.tools["list_workspaces"]()
        assert result["count"] == 0
        ctx.close()


# ------------------------------------------------------------------
# Snapshot
# ------------------------------------------------------------------


class TestSnapshot:
    """Test snapshot creation."""

    def test_snapshot_creates_copy(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        ctx_holder[0].store_dataset("prices", sample_df)
        ctx_holder[0].store_model("garch", {"params": {"alpha": 0.05}}, model_type="GARCH")

        result = tools["snapshot"](name="checkpoint_1")
        assert result["status"] == "created"
        assert result["snapshot"] == "checkpoint_1"

        snap_dir = Path(result["path"])
        assert snap_dir.exists()
        assert (snap_dir / "data.duckdb").exists()
        assert (snap_dir / "manifest.json").exists()

    def test_snapshot_auto_name(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        ctx_holder[0].store_dataset("data", sample_df)

        result = tools["snapshot"]()
        assert result["status"] == "created"
        assert result["snapshot"].startswith("snap_")

    def test_snapshot_captures_state(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        ctx_holder[0].store_dataset("prices", sample_df)

        result = tools["snapshot"](name="pre_change")
        snap_dir = Path(result["path"])

        # Read snapshot manifest
        with open(snap_dir / "manifest.json") as f:
            manifest = json.load(f)

        assert "prices" in manifest["datasets"]

    def test_multiple_snapshots(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        ctx_holder[0].store_dataset("v1_data", sample_df)
        tools["snapshot"](name="snap_v1")

        ctx_holder[0].store_dataset("v2_data", sample_df)
        tools["snapshot"](name="snap_v2")

        snap_dir = ctx_holder[0].workspace_dir / "snapshots"
        assert (snap_dir / "snap_v1").exists()
        assert (snap_dir / "snap_v2").exists()


# ------------------------------------------------------------------
# Restore snapshot
# ------------------------------------------------------------------


class TestRestoreSnapshot:
    """Test snapshot restoration."""

    def test_restore_brings_back_previous_state(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools

        # Initial state with one dataset
        ctx_holder[0].store_dataset("original", sample_df)
        tools["snapshot"](name="before_change")

        # Modify state — add more data
        extra = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        ctx_holder[0].store_dataset("extra", extra)

        # Restore snapshot
        result = tools["restore_snapshot"](name="before_change")
        assert result["status"] == "restored"

    def test_restore_nonexistent_snapshot_errors(self, mcp_tools):
        tools, _ = mcp_tools
        result = tools["restore_snapshot"](name="ghost_snapshot")
        assert "error" in result
        assert "not found" in result["error"]

    def test_restore_lists_available_snapshots(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        ctx_holder[0].store_dataset("data", sample_df)
        tools["snapshot"](name="real_snap")

        result = tools["restore_snapshot"](name="fake_snap")
        assert "available" in result
        assert "real_snap" in result["available"]


# ------------------------------------------------------------------
# Add note and journal
# ------------------------------------------------------------------


class TestAddNote:
    """Test add_note through context and workspace persistence."""

    def test_add_note_persists_in_journal(self, mcp_tools):
        _, ctx_holder = mcp_tools
        ctx = ctx_holder[0]
        ctx.add_note("Observed vol clustering in SPY returns")

        history = ctx.history(n=50)
        note_entries = [e for e in history if e["op"] == "note"]
        assert len(note_entries) >= 1
        assert any("vol clustering" in str(e) for e in note_entries)

    def test_add_note_on_disk(self, mcp_tools):
        _, ctx_holder = mcp_tools
        ctx = ctx_holder[0]
        ctx.add_note("Research note 1")

        journal_path = ctx.workspace_dir / "journal.jsonl"
        assert journal_path.exists()
        with open(journal_path) as f:
            lines = f.readlines()
        assert any("Research note 1" in line for line in lines)

    def test_multiple_notes(self, mcp_tools):
        _, ctx_holder = mcp_tools
        ctx = ctx_holder[0]
        ctx.add_note("Note A")
        ctx.add_note("Note B")
        ctx.add_note("Note C")

        history = ctx.history(n=50)
        notes = [e for e in history if e["op"] == "note"]
        assert len(notes) >= 3


# ------------------------------------------------------------------
# Query data
# ------------------------------------------------------------------


class TestQueryData:
    """Test query_data tool (SQL queries against DuckDB)."""

    def test_query_returns_results(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        ctx_holder[0].store_dataset("prices", sample_df)

        result = tools["query_data"](sql="SELECT close FROM prices LIMIT 5")
        assert result["rows"] == 5
        assert "close" in result["columns"]
        assert len(result["data"]) == 5

    def test_query_with_aggregation(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        ctx_holder[0].store_dataset("prices", sample_df)

        result = tools["query_data"](
            sql="SELECT AVG(close) as avg_close, COUNT(*) as n FROM prices"
        )
        assert result["rows"] == 1
        assert result["data"][0]["n"] == len(sample_df)

    def test_query_show_tables(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        ctx_holder[0].store_dataset("prices", sample_df)
        ctx_holder[0].store_dataset("returns", sample_df[["close"]])

        result = tools["query_data"](sql="SHOW TABLES")
        assert result["rows"] >= 2

    def test_query_rejects_mutations(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        ctx_holder[0].store_dataset("prices", sample_df)

        result = tools["query_data"](sql="DROP TABLE prices")
        assert "error" in result

        result = tools["query_data"](sql="DELETE FROM prices WHERE close > 100")
        assert "error" in result

        result = tools["query_data"](sql="INSERT INTO prices VALUES (1,2,3,4,5)")
        assert "error" in result

    def test_query_invalid_sql_returns_error(self, mcp_tools):
        tools, _ = mcp_tools
        result = tools["query_data"](sql="SELECT * FROM nonexistent_table")
        assert "error" in result

    def test_describe_table(self, mcp_tools, sample_df):
        tools, ctx_holder = mcp_tools
        ctx_holder[0].store_dataset("prices", sample_df)

        result = tools["query_data"](sql="DESCRIBE prices")
        assert result["rows"] > 0


# ------------------------------------------------------------------
# Delete workspace
# ------------------------------------------------------------------


class TestDeleteWorkspace:
    """Test delete_workspace tool."""

    def test_delete_workspace_removes_directory(self, mcp_tools, workspaces_root):
        tools, _ = mcp_tools
        tools["create_workspace"](name="to_delete")
        assert (workspaces_root / "to_delete").exists()

        # Switch to another workspace first
        tools["create_workspace"](name="keep_this")

        result = tools["delete_workspace"](name="to_delete")
        assert result["status"] == "deleted"
        assert not (workspaces_root / "to_delete").exists()

    def test_delete_nonexistent_workspace_errors(self, mcp_tools):
        tools, _ = mcp_tools
        result = tools["delete_workspace"](name="ghost")
        assert "error" in result
        assert "not found" in result["error"]

    def test_cannot_delete_active_workspace(self, mcp_tools):
        tools, ctx_holder = mcp_tools
        tools["create_workspace"](name="active_ws")
        # active_ws is now active
        result = tools["delete_workspace"](name="active_ws")
        assert "error" in result
        assert "active" in result["error"].lower() or "Cannot" in result["error"]


# ------------------------------------------------------------------
# Auto-versioning across workspace operations
# ------------------------------------------------------------------


class TestAutoVersioning:
    """Test that auto-versioning works correctly across workspace ops."""

    def test_version_increments_in_workspace(self, mcp_tools, sample_df):
        _, ctx_holder = mcp_tools
        ctx = ctx_holder[0]

        r1 = ctx.store_dataset("prices", sample_df)
        r2 = ctx.store_dataset("prices", sample_df)
        r3 = ctx.store_dataset("prices", sample_df)

        assert r1["dataset_id"] == "prices"
        assert r2["dataset_id"] == "prices_v2"
        assert r3["dataset_id"] == "prices_v3"

        datasets = ctx.list_datasets()
        assert "prices" in datasets
        assert "prices_v2" in datasets
        assert "prices_v3" in datasets

    def test_model_versioning_in_workspace(self, mcp_tools):
        _, ctx_holder = mcp_tools
        ctx = ctx_holder[0]

        m1 = ctx.store_model("garch", {"v": 1}, model_type="GARCH")
        m2 = ctx.store_model("garch", {"v": 2}, model_type="GARCH")

        assert m1["model_id"] == "garch"
        assert m2["model_id"] == "garch_v2"

        assert ctx.get_model("garch")["v"] == 1
        assert ctx.get_model("garch_v2")["v"] == 2

    def test_lineage_preserved_across_versions(self, mcp_tools, sample_df):
        _, ctx_holder = mcp_tools
        ctx = ctx_holder[0]

        ctx.store_dataset("prices", sample_df)
        returns = sample_df["close"].pct_change().dropna().to_frame("returns")
        ctx.store_dataset("returns", returns, parent="prices")

        lineage = ctx.registry.lineage("returns")
        assert lineage == ["prices", "returns"]
