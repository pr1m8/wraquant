"""Agent tool generation for :mod:`~fmp_docs_compiler`.

Purpose:
    Generate agent-friendly tool schemas from the normalized endpoint catalog.

Design:
    Each endpoint maps to a single JSON-schema based function definition. The
    output is intentionally runtime-agnostic so it can be adapted to multiple
    agent systems.

Attributes:
    None.

Examples:
    ::
        >>> from fmp_docs_compiler.models import CatalogIR, ManifestIR
        >>> build_tool_schemas(CatalogIR(source='x', source_urls=['y'], endpoints=[], manifest=ManifestIR(source='x', source_urls=['y'])))
        []
"""

from __future__ import annotations

from typing import Any

from .models import CatalogIR


def build_tool_schemas(catalog: CatalogIR) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for endpoint in catalog.endpoints:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter in endpoint.parameters:
            schema: dict[str, Any] = {
                "type": parameter.type_hint or "string",
                "description": parameter.description or parameter.name,
            }
            if parameter.enum_values:
                schema["enum"] = parameter.enum_values
            if parameter.example is not None:
                schema["examples"] = [parameter.example]
            properties[parameter.name] = schema
            if parameter.required:
                required.append(parameter.name)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": endpoint.operation_name,
                    "description": endpoint.about or endpoint.summary,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                    "x-wrapper-path": endpoint.wrapper_path,
                    "x-upstream-path": endpoint.upstream_path,
                    "x-upstream-host": endpoint.upstream_host,
                    "x-docs-url": endpoint.docs_url,
                    "x-verification-status": endpoint.verification_status.value,
                    "x-docs-confidence": endpoint.docs_confidence,
                },
            }
        )
    return tools
