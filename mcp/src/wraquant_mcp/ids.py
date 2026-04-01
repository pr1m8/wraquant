"""Resource ID system for wraquant-mcp.

Everything in a workspace is referenced by human-readable ID:
- Datasets: "prices_aapl", "returns_aapl_v2"
- Models: "garch_aapl_gjr_t"
- Results: "backtest_rsi_regime"

IDs auto-version on collision and track lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ResourceMeta:
    """Metadata for any workspace resource."""

    name: str
    resource_type: str = ""  # "dataset", "model", "result"
    created: datetime = field(default_factory=datetime.now)
    source_op: str | None = None  # function that created this
    parent: str | None = None  # ID of parent resource (lineage)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetMeta(ResourceMeta):
    """Metadata for a dataset stored in DuckDB."""

    rows: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.resource_type = "dataset"


@dataclass
class ModelMeta(ResourceMeta):
    """Metadata for a fitted model stored as joblib."""

    model_type: str = ""  # "garch", "hmm", "random_forest"
    metrics: dict[str, float] = field(default_factory=dict)
    source_dataset: str = ""

    def __post_init__(self):
        self.resource_type = "model"


class IDRegistry:
    """Track all resource IDs in a workspace.

    Handles auto-versioning (name collision → name_v2),
    lineage tracking, and resolution.
    """

    def __init__(self) -> None:
        self._resources: dict[str, ResourceMeta] = {}
        self._version_counts: dict[str, int] = {}

    def register(
        self,
        name: str,
        resource_type: str,
        source_op: str | None = None,
        parent: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Register a resource, auto-version if name exists.

        Returns the actual ID (may have _v2 suffix).
        """
        base_name = name
        if name in self._resources:
            count = self._version_counts.get(base_name, 1) + 1
            self._version_counts[base_name] = count
            name = f"{base_name}_v{count}"
        else:
            self._version_counts[base_name] = 1

        if resource_type == "dataset":
            meta = DatasetMeta(
                name=name,
                source_op=source_op,
                parent=parent,
                **kwargs,
            )
        elif resource_type == "model":
            meta = ModelMeta(
                name=name,
                source_op=source_op,
                parent=parent,
                **kwargs,
            )
        else:
            meta = ResourceMeta(
                name=name,
                resource_type=resource_type,
                source_op=source_op,
                parent=parent,
            )

        self._resources[name] = meta
        return name

    def get(self, name: str) -> ResourceMeta | None:
        """Get metadata for a resource by ID."""
        return self._resources.get(name)

    def exists(self, name: str) -> bool:
        """Check if a resource ID exists."""
        return name in self._resources

    def list_datasets(self) -> list[str]:
        """List all dataset IDs."""
        return [
            k for k, v in self._resources.items() if v.resource_type == "dataset"
        ]

    def list_models(self) -> list[str]:
        """List all model IDs."""
        return [
            k for k, v in self._resources.items() if v.resource_type == "model"
        ]

    def list_all(self) -> list[str]:
        """List all resource IDs."""
        return list(self._resources.keys())

    def lineage(self, name: str) -> list[str]:
        """Trace the derivation chain of a resource.

        Returns list from oldest ancestor to the resource itself.
        """
        chain = []
        current = name
        while current and current in self._resources:
            chain.append(current)
            current = self._resources[current].parent
        return list(reversed(chain))

    def latest(self, base_name: str | None = None) -> str | None:
        """Get the most recently created resource.

        If base_name given, get latest version of that name.
        """
        if base_name:
            candidates = [
                k
                for k in self._resources
                if k == base_name or k.startswith(f"{base_name}_v")
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda k: self._resources[k].created)

        if not self._resources:
            return None
        return max(self._resources, key=lambda k: self._resources[k].created)

    def summary(self) -> dict[str, Any]:
        """Summary of all resources for workspace_status tool."""
        return {
            "datasets": self.list_datasets(),
            "models": self.list_models(),
            "total": len(self._resources),
        }
