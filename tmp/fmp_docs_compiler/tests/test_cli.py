from __future__ import annotations

from pathlib import Path

from fmp_docs_compiler.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_compile_and_build_all(tmp_path: Path) -> None:
    fixtures_dir = Path("tests/fixtures/site")
    compile_result = runner.invoke(
        app,
        [
            "compile-fixtures",
            "--fixtures-dir",
            str(fixtures_dir),
            "--out-dir",
            str(tmp_path / "catalog_artifacts"),
        ],
    )
    assert compile_result.exit_code == 0, compile_result.stdout
    assert (tmp_path / "catalog_artifacts" / "catalog.json").exists()
    assert (tmp_path / "catalog_artifacts" / "manifest.json").exists()

    build_result = runner.invoke(
        app,
        [
            "build-all",
            "--catalog",
            str(tmp_path / "catalog_artifacts" / "catalog.json"),
            "--out-dir",
            str(tmp_path / "build_artifacts"),
        ],
    )
    assert build_result.exit_code == 0, build_result.stdout
    assert (tmp_path / "build_artifacts" / "openapi.json").exists()
    assert (tmp_path / "build_artifacts" / "tools.json").exists()
    assert (tmp_path / "build_artifacts" / "generated_wrapper" / "app.py").exists()
    assert (tmp_path / "build_artifacts" / "generated_wrapper" / "mcp_auto.py").exists()
    assert (
        tmp_path / "build_artifacts" / "generated_wrapper" / "mcp_manual.py"
    ).exists()
    assert (
        tmp_path / "build_artifacts" / "generated_wrapper" / "combined_app.py"
    ).exists()


def test_cli_render_mcp_and_inspect_and_doctor(tmp_path: Path) -> None:
    fixtures_dir = Path("tests/fixtures/site")
    runner.invoke(
        app,
        [
            "compile-fixtures",
            "--fixtures-dir",
            str(fixtures_dir),
            "--out-dir",
            str(tmp_path / "catalog_artifacts"),
        ],
        catch_exceptions=False,
    )

    mcp_result = runner.invoke(
        app,
        [
            "render-mcp",
            "--catalog",
            str(tmp_path / "catalog_artifacts" / "catalog.json"),
            "--out-dir",
            str(tmp_path / "mcp_artifacts"),
        ],
    )
    assert mcp_result.exit_code == 0, mcp_result.stdout
    assert (tmp_path / "mcp_artifacts" / "generated_wrapper" / "mcp_auto.py").exists()

    inspect_result = runner.invoke(
        app,
        [
            "inspect-endpoint",
            "--catalog",
            str(tmp_path / "catalog_artifacts" / "catalog.json"),
            "--name",
            "get_latest_sec_filings",
        ],
    )
    assert inspect_result.exit_code == 0, inspect_result.stdout
    assert "get_latest_sec_filings" in inspect_result.stdout

    doctor_result = runner.invoke(
        app,
        ["doctor", "--catalog", str(tmp_path / "catalog_artifacts" / "catalog.json")],
    )
    assert doctor_result.exit_code == 0, doctor_result.stdout
    assert "No issues found" in doctor_result.stdout
