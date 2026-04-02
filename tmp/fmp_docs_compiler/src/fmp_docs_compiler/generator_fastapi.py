"""FastAPI scaffold generation for :mod:`~fmp_docs_compiler`.

Purpose:
    Generate a small but structured FastAPI wrapper project from the normalized
    catalog.

Design:
    The generated wrapper includes request models, settings, a thin upstream
    client adapter, and routes that can run in dry-run mode or call upstream
    when enabled.

Attributes:
    None.

Examples:
    ::
        >>> from fmp_docs_compiler.models import CatalogIR, ManifestIR
        >>> files = render_fastapi_project(CatalogIR(source="x", source_urls=["y"], endpoints=[], manifest=ManifestIR(source="x", source_urls=["y"])))
        >>> 'generated_wrapper/app.py' in files
        True
"""

from __future__ import annotations

import keyword
import re

from .models import CatalogIR, EndpointIR, ParameterIR


def render_fastapi_project(catalog: CatalogIR) -> dict[str, str]:
    """Render a generated FastAPI project.

    Args:
        catalog:
            Normalized catalog.

    Returns:
        A mapping of relative file path to file content.

    Raises:
        None.

    Examples:
        ::
            >>> from fmp_docs_compiler.models import CatalogIR, ManifestIR
            >>> files = render_fastapi_project(CatalogIR(source="x", source_urls=["y"], endpoints=[], manifest=ManifestIR(source="x", source_urls=["y"])))
            >>> 'generated_wrapper/models.py' in files
            True
    """
    request_models = "\n\n".join(
        _render_request_model(endpoint) for endpoint in catalog.endpoints
    )
    exports = ", ".join(
        f'"{_request_model_name(endpoint)}"' for endpoint in catalog.endpoints
    )
    routes = "\n\n".join(_render_route(endpoint) for endpoint in catalog.endpoints)

    models_text = f'''"""Generated request models.

Purpose:
    Store typed request models for the generated wrapper routes.

Design:
    Each endpoint receives a small Pydantic request model so route signatures
    stay clean and editable.

Attributes:
    __all__:
        Public request-model exports.

Examples:
    ::
        >>> len(__all__) >= 0
        True
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


{request_models}

__all__ = [{exports}]
'''

    settings_text = '''"""Generated runtime settings.

Purpose:
    Provide a minimal settings object for the generated wrapper.

Design:
    The settings are intentionally plain so users can replace them with
    Pydantic Settings or another configuration system later.

Attributes:
    None.

Examples:
    ::
        >>> get_settings().upstream_base_url.startswith('https://')
        True
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    upstream_base_url: str = 'https://financialmodelingprep.com'
    api_key: str | None = None
    proxy_enabled: bool = False


def get_settings() -> Settings:
    return Settings()
'''

    upstream_text = '''"""Generated upstream client adapter.

Purpose:
    Encapsulate upstream HTTP calls for the generated wrapper.

Design:
    The adapter keeps request construction in one place and supports dry-run
    mode when proxying is disabled.

Attributes:
    None.

Examples:
    ::
        >>> callable(get_upstream_client)
        True
"""

from __future__ import annotations

from typing import Any

import httpx

from .settings import Settings, get_settings


class FMPUpstreamClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def call(self, path: str, query: dict[str, Any]) -> dict[str, Any]:
        filtered_query = {key: value for key, value in query.items() if value is not None}
        if self.settings.api_key:
            filtered_query.setdefault('apikey', self.settings.api_key)
        if not self.settings.proxy_enabled:
            return {'mode': 'dry_run', 'path': path, 'query': filtered_query}
        async with httpx.AsyncClient(base_url=self.settings.upstream_base_url, follow_redirects=True) as client:
            response = await client.get(path, params=filtered_query)
            response.raise_for_status()
            return response.json()


def get_upstream_client(settings: Settings | None = None) -> FMPUpstreamClient:
    return FMPUpstreamClient(settings or get_settings())
'''

    app_text = f'''"""Generated FastAPI application.

Purpose:
    Provide a lightweight wrapper application scaffold generated from a docs
    catalog.

Design:
    The application uses generated request models plus a minimal upstream
    client adapter. Each route supports dry-run mode for safe editing and
    inspection before live proxying is enabled.

Attributes:
    app:
        FastAPI application instance.

Examples:
    ::
        >>> callable(app)
        True
"""

from fastapi import Depends, FastAPI

from .models import *
from .settings import Settings, get_settings
from .upstream import FMPUpstreamClient, get_upstream_client

app = FastAPI(title='FMP Wrapper', version='0.4.0')


@app.get('/health', operation_id='health_check')
async def health() -> dict[str, str]:
    return {{'status': 'ok'}}


{routes}
'''

    return {
        "generated_wrapper/README.md": "# Generated Wrapper\n\nThis scaffold was generated from the FMP docs compiler catalog.\n",
        "generated_wrapper/__init__.py": '"""Generated wrapper package."""\n',
        "generated_wrapper/models.py": models_text,
        "generated_wrapper/settings.py": settings_text,
        "generated_wrapper/upstream.py": upstream_text,
        "generated_wrapper/app.py": app_text,
    }


def _request_model_name(endpoint: EndpointIR) -> str:
    parts = re.split(r"[_\-]+", endpoint.operation_name)
    return "".join(part.capitalize() for part in parts if part) + "Request"


def _py_type(type_hint: str | None) -> str:
    mapping = {"integer": "int", "number": "float", "boolean": "bool", "string": "str"}
    return mapping.get(type_hint or "string", "str")


def pythonize_identifier(name: str) -> str:
    """Convert an arbitrary name into a valid Python identifier.

    Args:
        name:
            Source name.

    Returns:
        A valid Python identifier.

    Raises:
        None.

    Examples:
        ::
            >>> pythonize_identifier('from')
            'from_'
            >>> pythonize_identifier('earnings-calendar')
            'earnings_calendar'
    """
    identifier = re.sub(r"\W+", "_", name).strip("_")
    if not identifier:
        identifier = "value"
    if identifier[0].isdigit():
        identifier = f"field_{identifier}"
    if keyword.iskeyword(identifier):
        identifier = f"{identifier}_"
    return identifier


def _field_line(parameter: ParameterIR) -> str:
    py_name = pythonize_identifier(parameter.name)
    annotation = _py_type(parameter.type_hint)
    if parameter.required:
        field_type = annotation
        if py_name != parameter.name:
            default = f" = Field(..., alias={parameter.name!r})"
        else:
            default = " = ..."
    else:
        field_type = f"{annotation} | None"
        if py_name != parameter.name:
            default = f" = Field(default=None, alias={parameter.name!r})"
        else:
            default = " = None"
    return f"    {py_name}: {field_type}{default}"


def _render_request_model(endpoint: EndpointIR) -> str:
    lines = [
        f"class {_request_model_name(endpoint)}(BaseModel):",
        '    model_config = ConfigDict(extra="forbid", populate_by_name=True)',
    ]
    if not endpoint.parameters:
        lines.append("    pass")
    else:
        for parameter in endpoint.parameters:
            lines.append(_field_line(parameter))
    return "\n".join(lines)


def _render_route(endpoint: EndpointIR) -> str:
    model_name = _request_model_name(endpoint)
    return f"""@app.get("{endpoint.wrapper_path}", operation_id="{endpoint.operation_name}")
async def {endpoint.operation_name}(
    request: {model_name} = Depends(),
    client: FMPUpstreamClient = Depends(get_upstream_client),
    settings: Settings = Depends(get_settings),
):
    del settings
    payload = request.model_dump(exclude_none=True, by_alias=True)
    return await client.call(path="{endpoint.upstream_path}", query=payload)"""
