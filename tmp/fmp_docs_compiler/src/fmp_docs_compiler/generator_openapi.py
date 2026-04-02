"""OpenAPI generation for :mod:`~fmp_docs_compiler`.

Purpose:
    Generate an OpenAPI document from a normalized catalog.

Design:
    The output describes the internal wrapper API rather than claiming the
    upstream vendor publishes an official OpenAPI specification.

Attributes:
    None.

Examples:
    ::
        >>> from fmp_docs_compiler.models import CatalogIR, ManifestIR
        >>> build_openapi_document(CatalogIR(source='x', source_urls=['y'], endpoints=[], manifest=ManifestIR(source='x', source_urls=['y'])))['openapi']
        '3.1.0'
"""

from __future__ import annotations

from typing import Any

from .models import CatalogIR


def build_openapi_document(catalog: CatalogIR) -> dict[str, Any]:
    """Build an OpenAPI-like document from a catalog.

    Args:
        catalog:
            Normalized endpoint catalog.

    Returns:
        An OpenAPI document as a dictionary.

    Raises:
        None.

    Examples:
        ::
            >>> from fmp_docs_compiler.models import CatalogIR, ManifestIR
            >>> document = build_openapi_document(CatalogIR(source='x', source_urls=['y'], endpoints=[], manifest=ManifestIR(source='x', source_urls=['y'])))
            >>> document['info']['title']
            'FMP Wrapper API'
    """
    document: dict[str, Any] = {
        "openapi": "3.1.0",
        "info": {
            "title": "FMP Wrapper API",
            "version": "0.3.0",
            "description": "Generated from FMP docs pages by a docs-first compiler.",
        },
        "paths": {},
    }

    for endpoint in catalog.endpoints:
        operation = {
            "operationId": endpoint.operation_name,
            "summary": endpoint.summary,
            "description": endpoint.about or endpoint.summary,
            "tags": [endpoint.category],
            "parameters": [],
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {"application/json": {"schema": {"type": "object"}}},
                }
            },
            "x-upstream-host": endpoint.upstream_host,
            "x-upstream-path": endpoint.upstream_path,
            "x-docs-url": endpoint.docs_url,
            "x-premium-signals": endpoint.premium_signals,
            "x-verification-status": endpoint.verification_status.value,
            "x-docs-confidence": endpoint.docs_confidence,
        }
        for parameter in endpoint.parameters:
            entry = {
                "name": parameter.name,
                "in": parameter.location,
                "required": parameter.required,
                "schema": {"type": parameter.type_hint or "string"},
            }
            if parameter.description:
                entry["description"] = parameter.description
            if parameter.example is not None:
                entry["example"] = parameter.example
            if parameter.enum_values:
                entry["schema"]["enum"] = parameter.enum_values
            operation["parameters"].append(entry)
        if endpoint.response_examples:
            operation["responses"]["200"]["content"]["application/json"]["examples"] = {
                f"example_{index + 1}": {"value": example.payload}
                for index, example in enumerate(endpoint.response_examples)
            }
        document["paths"][endpoint.wrapper_path] = {
            endpoint.upstream_method.value.lower(): operation
        }
    return document
