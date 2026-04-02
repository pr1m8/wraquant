"""Normalization for :mod:`~fmp_docs_compiler`.

Purpose:
    Convert parsed docs pages into a stable intermediate representation.

Design:
    Normalization merges parameter information from parsed tables, lists, and
    URL query strings, then creates deterministic wrapper paths and operation
    names for each endpoint.

Attributes:
    None.

Examples:
    ::
        >>> normalizer = EndpointNormalizer()
        >>> isinstance(normalizer, EndpointNormalizer)
        True
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import parse_qsl, urlparse

from .models import (
    AuthMode,
    CatalogIR,
    EndpointIR,
    HttpMethod,
    ManifestIR,
    ParameterIR,
    ParsedEndpointPage,
    ResponseExampleIR,
)


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "endpoint"


def _operation_name_from_title(title: str) -> str:
    cleaned = re.sub(r"\bapi\b", "", title, flags=re.IGNORECASE)
    slug = _slugify(cleaned).replace("-", "_")
    return f"get_{slug}" if not slug.startswith("get_") else slug


def _infer_type_hint(example: str) -> str:
    if re.fullmatch(r"-?\d+", example):
        return "integer"
    if re.fullmatch(r"-?\d+\.\d+", example):
        return "number"
    if example.lower() in {"true", "false"}:
        return "boolean"
    return "string"


class EndpointNormalizer:
    """Normalize parsed pages into endpoint IR models."""

    def normalize_many(
        self,
        parsed_pages: Iterable[ParsedEndpointPage],
        source: str,
        source_urls: list[str],
        manifest: ManifestIR | None = None,
    ) -> CatalogIR:
        endpoints = [self.normalize_page(parsed_page) for parsed_page in parsed_pages]
        local_manifest = manifest or ManifestIR(source=source, source_urls=source_urls)
        local_manifest.stats.parsed_pages = len(endpoints)
        local_manifest.stats.premium_flagged_pages = sum(
            1 for endpoint in endpoints if endpoint.premium_signals
        )
        local_manifest.stats.warning_count = sum(
            len(endpoint.parse_warnings) for endpoint in endpoints
        )
        return CatalogIR(
            source=source,
            source_urls=source_urls,
            endpoints=endpoints,
            manifest=local_manifest,
        )

    def normalize_page(self, parsed_page: ParsedEndpointPage) -> EndpointIR:
        endpoint_url = (
            parsed_page.endpoint_url
            or "https://financialmodelingprep.com/stable/unknown"
        )
        parsed = urlparse(endpoint_url)
        query_parameters = self._parameters_from_url(endpoint_url)
        structured_parameters: dict[str, ParameterIR] = {
            parameter.name: ParameterIR(
                name=parameter.name,
                location=parameter.location,
                required=parameter.required,
                type_hint=parameter.type_hint,
                description=parameter.description,
                default=parameter.default,
                example=parameter.example,
                enum_values=list(parameter.enum_values),
                source=parameter.source,
            )
            for parameter in parsed_page.parsed_parameters
        }
        for parameter in query_parameters:
            existing = structured_parameters.get(parameter.name)
            if existing is None:
                structured_parameters[parameter.name] = parameter
            else:
                if existing.example is None:
                    existing.example = parameter.example
                if existing.type_hint is None:
                    existing.type_hint = parameter.type_hint

        wrapper_category = _slugify(parsed_page.category)
        operation_name = _operation_name_from_title(parsed_page.title)
        response_examples = [
            ResponseExampleIR(
                source=example.source,
                content_type=example.content_type,
                payload=example.payload,
                description=example.label,
            )
            for example in parsed_page.examples
        ]
        docs_confidence = 0.55
        if parsed_page.endpoint_url:
            docs_confidence += 0.15
        if structured_parameters:
            docs_confidence += 0.15
        if parsed_page.about:
            docs_confidence += 0.05
        if parsed_page.example_use_case:
            docs_confidence += 0.05
        if parsed_page.premium_signals:
            docs_confidence -= 0.05
        docs_confidence = max(0.0, min(docs_confidence, 1.0))

        return EndpointIR(
            operation_name=operation_name,
            docs_url=parsed_page.docs_url,
            category=wrapper_category,
            title=parsed_page.title,
            summary=parsed_page.summary or parsed_page.title,
            about=parsed_page.about,
            example_use_case=parsed_page.example_use_case,
            upstream_method=HttpMethod.GET,
            upstream_host=(
                f"{parsed.scheme}://{parsed.netloc}"
                if parsed.scheme and parsed.netloc
                else "https://financialmodelingprep.com"
            ),
            upstream_path=parsed.path or "/stable/unknown",
            wrapper_path=f'/fmp/{wrapper_category}/{operation_name.replace("_", "-")}',
            auth_mode=AuthMode.APIKEY_QUERY,
            parameters=list(structured_parameters.values()),
            response_examples=response_examples,
            premium_signals=list(parsed_page.premium_signals),
            parse_warnings=list(parsed_page.parse_warnings),
            deprecated=parsed_page.legacy,
            docs_confidence=docs_confidence,
        )

    def _parameters_from_url(self, endpoint_url: str) -> list[ParameterIR]:
        params: list[ParameterIR] = []
        for key, value in parse_qsl(
            urlparse(endpoint_url).query, keep_blank_values=True
        ):
            params.append(
                ParameterIR(
                    name=key,
                    location="query",
                    required=False,
                    type_hint=_infer_type_hint(value) if value else None,
                    example=value or None,
                    source="url_query",
                )
            )
        return params
