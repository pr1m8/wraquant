from __future__ import annotations

from pathlib import Path

from fmp_docs_compiler.compiler import FMPDocsCompiler
from fmp_docs_compiler.generator_fastapi import render_fastapi_project
from fmp_docs_compiler.http import RetryConfig


def test_request_model_uses_alias_for_reserved_keyword() -> None:
    compiler = FMPDocsCompiler(retry_config=RetryConfig())
    catalog = compiler.compile_fixtures(fixtures_dir=Path("tests/fixtures/site"))
    project = render_fastapi_project(catalog)
    models_text = project["generated_wrapper/models.py"]
    assert "from_: str | None = Field(default=None, alias='from')" in models_text
    assert (
        "model_dump(exclude_none=True, by_alias=True)"
        in project["generated_wrapper/app.py"]
    )
