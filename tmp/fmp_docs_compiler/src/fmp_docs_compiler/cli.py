"""Rich CLI for :mod:`~fmp_docs_compiler`.

Purpose:
    Provide a user-friendly command-line interface for compilation,
    verification, inspection, and artifact generation.

Design:
    The CLI uses :mod:`typer` for commands and :mod:`rich` for progress,
    summary tables, and status messaging.

Attributes:
    app:
        Root Typer application.

Examples:
    ::
        >>> callable(app)
        True
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .compiler import FMPDocsCompiler
from .generator_fastapi import render_fastapi_project
from .generator_mcp import render_mcp_project
from .generator_openapi import build_openapi_document
from .generator_tools import build_tool_schemas
from .http import RetryConfig
from .io_utils import write_json, write_text
from .models import CatalogIR, ManifestIR

app = typer.Typer(help="Compile FMP docs and generate wrapper artifacts.")
console = Console()


def _load_catalog(path: Path) -> CatalogIR:
    return CatalogIR.model_validate_json(path.read_text(encoding="utf-8"))


def _build_retry_config(
    requests_per_second: float,
    burst_capacity: int,
    max_retries: int,
    timeout_seconds: float,
) -> RetryConfig:
    return RetryConfig(
        requests_per_second=requests_per_second,
        burst_capacity=burst_capacity,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )


def _render_summary_table(catalog: CatalogIR) -> Table:
    table = Table(title="FMP Docs Compilation Summary")
    table.add_column("Source")
    table.add_column("Endpoints", justify="right")
    table.add_column("Premium flagged", justify="right")
    table.add_column("Warnings", justify="right")
    table.add_column("Legacy", justify="right")
    premium_count = sum(1 for endpoint in catalog.endpoints if endpoint.premium_signals)
    legacy_count = sum(1 for endpoint in catalog.endpoints if endpoint.deprecated)
    warning_count = sum(len(endpoint.parse_warnings) for endpoint in catalog.endpoints)
    table.add_row(
        catalog.source,
        str(catalog.endpoint_count),
        str(premium_count),
        str(warning_count),
        str(legacy_count),
    )
    return table


def _render_verification_table(catalog: CatalogIR) -> Table:
    counts: dict[str, int] = {}
    for endpoint in catalog.endpoints:
        counts[endpoint.verification_status.value] = (
            counts.get(endpoint.verification_status.value, 0) + 1
        )
    table = Table(title="Verification Summary")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    for status, count in sorted(counts.items()):
        table.add_row(status, str(count))
    return table


def _write_catalog_bundle(out_dir: Path, catalog: CatalogIR) -> None:
    write_json(
        out_dir / "catalog.json",
        catalog.model_dump(mode="json", exclude_computed_fields=True),
    )
    write_json(out_dir / "manifest.json", catalog.manifest.model_dump(mode="json"))


@app.command("compile-live")
def compile_live(
    out_dir: Annotated[
        Path, typer.Option(help="Output directory for generated catalog artifacts.")
    ],
    cache_dir: Annotated[
        Path | None, typer.Option(help="Optional raw HTML cache directory.")
    ] = None,
    max_pages: Annotated[int | None, typer.Option(help="Optional page limit.")] = None,
    requests_per_second: Annotated[
        float, typer.Option(help="Client-side request budget.")
    ] = 2.0,
    burst_capacity: Annotated[int, typer.Option(help="Burst capacity.")] = 4,
    max_retries: Annotated[int, typer.Option(help="Retry count.")] = 4,
    timeout_seconds: Annotated[float, typer.Option(help="Per-request timeout.")] = 30.0,
) -> None:
    retry_config = _build_retry_config(
        requests_per_second, burst_capacity, max_retries, timeout_seconds
    )
    compiler = FMPDocsCompiler(retry_config=retry_config)
    with console.status("Compiling live FMP docs..."):
        catalog = asyncio.run(
            compiler.compile_live(max_pages=max_pages, cache_dir=cache_dir)
        )
    _write_catalog_bundle(out_dir, catalog)
    console.print(_render_summary_table(catalog))
    console.print(f'[green]Wrote[/green] {out_dir / "catalog.json"}')
    console.print(f'[green]Wrote[/green] {out_dir / "manifest.json"}')


@app.command("compile-fixtures")
def compile_fixtures(
    fixtures_dir: Annotated[Path, typer.Option(help="Fixture site directory.")],
    out_dir: Annotated[Path, typer.Option(help="Output directory.")],
) -> None:
    compiler = FMPDocsCompiler(retry_config=RetryConfig())
    with console.status("Compiling local fixtures..."):
        catalog = compiler.compile_fixtures(fixtures_dir=fixtures_dir)
    _write_catalog_bundle(out_dir, catalog)
    console.print(_render_summary_table(catalog))
    console.print(f'[green]Wrote[/green] {out_dir / "catalog.json"}')


@app.command("verify-catalog")
def verify_catalog(
    catalog: Annotated[Path, typer.Option(help="Catalog JSON path.")],
    out: Annotated[Path, typer.Option(help="Verified catalog output path.")],
    api_key: Annotated[str | None, typer.Option(help="Optional FMP API key.")] = None,
    requests_per_second: Annotated[
        float, typer.Option(help="Client-side request budget.")
    ] = 1.0,
    burst_capacity: Annotated[int, typer.Option(help="Burst capacity.")] = 2,
    max_retries: Annotated[int, typer.Option(help="Retry count.")] = 2,
    timeout_seconds: Annotated[float, typer.Option(help="Per-request timeout.")] = 20.0,
) -> None:
    retry_config = _build_retry_config(
        requests_per_second, burst_capacity, max_retries, timeout_seconds
    )
    compiler = FMPDocsCompiler(retry_config=retry_config)
    with console.status("Verifying catalog..."):
        verified = asyncio.run(
            compiler.verify_catalog(_load_catalog(catalog), api_key=api_key)
        )
    write_json(out, verified.model_dump(mode="json", exclude_computed_fields=True))
    console.print(_render_verification_table(verified))
    console.print(f"[green]Wrote[/green] {out}")


@app.command("render-openapi")
def render_openapi(
    catalog: Annotated[Path, typer.Option(help="Catalog JSON path.")],
    out: Annotated[Path, typer.Option(help="OpenAPI JSON output path.")],
) -> None:
    write_json(out, build_openapi_document(_load_catalog(catalog)))
    console.print(f"[green]Wrote[/green] {out}")


@app.command("render-tools")
def render_tools(
    catalog: Annotated[Path, typer.Option(help="Catalog JSON path.")],
    out: Annotated[Path, typer.Option(help="Tool schema output path.")],
) -> None:
    write_json(out, build_tool_schemas(_load_catalog(catalog)))
    console.print(f"[green]Wrote[/green] {out}")


@app.command("render-fastapi")
def render_fastapi(
    catalog: Annotated[Path, typer.Option(help="Catalog JSON path.")],
    out_dir: Annotated[
        Path, typer.Option(help="Output directory for generated project files.")
    ],
) -> None:
    for relative_path, content in render_fastapi_project(
        _load_catalog(catalog)
    ).items():
        write_text(out_dir / relative_path, content)
    console.print(f"[green]Wrote[/green] generated wrapper under {out_dir}")


@app.command("render-mcp")
def render_mcp(
    catalog: Annotated[Path, typer.Option(help="Catalog JSON path.")],
    out_dir: Annotated[
        Path, typer.Option(help="Output directory for generated MCP files.")
    ],
) -> None:
    for relative_path, content in render_mcp_project(_load_catalog(catalog)).items():
        write_text(out_dir / relative_path, content)
    console.print(f"[green]Wrote[/green] generated MCP files under {out_dir}")


@app.command("build-all")
def build_all(
    catalog: Annotated[Path, typer.Option(help="Catalog JSON path.")],
    out_dir: Annotated[
        Path, typer.Option(help="Output directory for all generated artifacts.")
    ],
) -> None:
    loaded = _load_catalog(catalog)
    write_json(out_dir / "openapi.json", build_openapi_document(loaded))
    write_json(out_dir / "tools.json", build_tool_schemas(loaded))
    for relative_path, content in render_fastapi_project(loaded).items():
        write_text(out_dir / relative_path, content)
    for relative_path, content in render_mcp_project(loaded).items():
        write_text(out_dir / relative_path, content)
    console.print(_render_summary_table(loaded))
    console.print(f"[green]Wrote[/green] build artifacts under {out_dir}")


@app.command("inspect-endpoint")
def inspect_endpoint(
    catalog: Annotated[Path, typer.Option(help="Catalog JSON path.")],
    name: Annotated[str, typer.Option(help="Operation name or slug to inspect.")],
) -> None:
    loaded = _load_catalog(catalog)
    for endpoint in loaded.endpoints:
        if endpoint.operation_name == name or endpoint.slug == name:
            console.print(
                Panel.fit(
                    endpoint.model_dump_json(indent=2), title=endpoint.operation_name
                )
            )
            raise typer.Exit()
    console.print(f"[red]No endpoint matched[/red] {name}")
    raise typer.Exit(code=1)


@app.command("doctor")
def doctor(catalog: Annotated[Path, typer.Option(help="Catalog JSON path.")]) -> None:
    loaded = _load_catalog(catalog)
    seen_paths: set[str] = set()
    issues: list[str] = []
    for endpoint in loaded.endpoints:
        if endpoint.wrapper_path in seen_paths:
            issues.append(f"Duplicate wrapper path: {endpoint.wrapper_path}")
        seen_paths.add(endpoint.wrapper_path)
        if not endpoint.upstream_path.startswith("/"):
            issues.append(
                f"Upstream path does not start with /: {endpoint.operation_name}"
            )
        if not endpoint.summary:
            issues.append(f"Missing summary: {endpoint.operation_name}")
    console.print(
        Panel.fit("\n".join(issues) if issues else "No issues found.", title="Doctor")
    )
    if issues:
        raise typer.Exit(code=1)
