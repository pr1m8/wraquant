"""Core models for :mod:`~fmp_docs_compiler`.

Purpose:
    Define the typed intermediate representation used across discovery,
    parsing, normalization, verification, and generation phases.

Design:
    The package uses a layered model design:

    - discovery models for candidate docs pages
    - parsed models for page-level extraction
    - normalized endpoint and catalog models for downstream generation
    - manifest and verification models for repeatable builds

Attributes:
    None.

Examples:
    ::
        >>> endpoint = EndpointIR(
        ...     operation_name="get_quote",
        ...     docs_url="https://example.test/docs/quote",
        ...     category="quotes",
        ...     title="Quote API",
        ...     summary="Get quote data.",
        ...     upstream_method=HttpMethod.GET,
        ...     upstream_host="https://financialmodelingprep.com",
        ...     upstream_path="/stable/quote",
        ...     wrapper_path="/fmp/quotes/get-quote",
        ... )
        >>> endpoint.slug
        'get-quote'
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class HttpMethod(StrEnum):
    """Supported HTTP methods.

    Args:
        None.

    Returns:
        A string enum value.

    Raises:
        None.

    Examples:
        ::
            >>> HttpMethod.GET.value
            'GET'
    """

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class AuthMode(StrEnum):
    """Supported upstream authentication modes.

    Args:
        None.

    Returns:
        A string enum value.

    Raises:
        None.

    Examples:
        ::
            >>> AuthMode.APIKEY_QUERY.value
            'apikey_query'
    """

    APIKEY_QUERY = "apikey_query"
    APIKEY_HEADER = "apikey_header"
    UNKNOWN = "unknown"


class VerificationStatus(StrEnum):
    """Verification status for an endpoint.

    Args:
        None.

    Returns:
        A string enum value.

    Raises:
        None.

    Examples:
        ::
            >>> VerificationStatus.UNVERIFIED.value
            'unverified'
    """

    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    RATE_LIMITED = "rate_limited"
    PLAN_GATED = "plan_gated"
    INVALID_API_KEY = "invalid_api_key"
    SKIPPED = "skipped"
    FAILED = "failed"


class RetryOutcome(StrEnum):
    """High-level fetch outcome classification.

    Args:
        None.

    Returns:
        A string enum value.

    Raises:
        None.

    Examples:
        ::
            >>> RetryOutcome.SUCCESS.value
            'success'
    """

    SUCCESS = "success"
    RETRIED = "retried"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"


class DiscoveredDocPage(BaseModel):
    """Metadata describing one docs page to crawl.

    Args:
        url:
            Absolute docs page URL or local fixture path.
        category:
            Best-effort category inferred from the index page.
        label:
            Link label from the docs index.
        source_index_url:
            Source index page URL.
        legacy:
            Whether the page appears to come from a legacy section.

    Returns:
        A validated discovery record.

    Raises:
        ValueError:
            Raised by Pydantic validation.

    Examples:
        ::
            >>> page = DiscoveredDocPage(
            ...     url="https://example.test/docs/a",
            ...     category="reference",
            ...     label="Available Countries",
            ...     source_index_url="https://example.test/docs",
            ... )
            >>> page.label
            'Available Countries'
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    category: str
    label: str
    source_index_url: str
    legacy: bool = False


class ParsedParameterDoc(BaseModel):
    """Parameter parsed directly from vendor docs.

    Args:
        name:
            Parameter name.
        type_hint:
            Best-effort type hint.
        description:
            Human-readable description.
        required:
            Whether the docs imply the parameter is required.
        default:
            Optional default value.
        example:
            Optional example value.
        location:
            Parameter location.
        enum_values:
            Optional allowed values.
        source:
            Parser source such as ``table`` or ``list``.

    Returns:
        A validated parsed parameter record.

    Raises:
        ValueError:
            Raised by Pydantic validation.

    Examples:
        ::
            >>> ParsedParameterDoc(name="symbol", source="table").name
            'symbol'
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type_hint: str | None = None
    description: str | None = None
    required: bool = False
    default: str | None = None
    example: Any | None = None
    location: str = "query"
    enum_values: list[str] = Field(default_factory=list)
    source: str = "unknown"


class ParsedExampleDoc(BaseModel):
    """Example payload parsed from docs text or code blocks.

    Args:
        label:
            Human-readable example label.
        content_type:
            MIME-like content type.
        payload:
            Example payload.
        source:
            Parser source.

    Returns:
        A validated example object.

    Raises:
        ValueError:
            Raised by Pydantic validation.

    Examples:
        ::
            >>> ParsedExampleDoc(label="endpoint", payload="https://example.test", source="text").label
            'endpoint'
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    content_type: str = "text/plain"
    payload: Any
    source: str = "unknown"


class ParsedEndpointPage(BaseModel):
    """Heuristic extraction result for one docs page.

    Args:
        docs_url:
            Docs page URL.
        category:
            Category inferred during discovery.
        label:
            Link label from the docs index.
        title:
            Page title.
        summary:
            Summary paragraph.
        about:
            About section text.
        example_use_case:
            Example use case text.
        endpoint_url:
            Example upstream endpoint URL.
        parsed_parameters:
            Structured parameters extracted from the page.
        examples:
            Parsed example artifacts.
        premium_signals:
            Gating or premium hints found on the page.
        parse_warnings:
            Non-fatal parser warnings.
        raw_text:
            Full normalized page text.
        legacy:
            Whether the page appears legacy.

    Returns:
        A validated parsed page model.

    Raises:
        ValueError:
            Raised by Pydantic validation.

    Examples:
        ::
            >>> page = ParsedEndpointPage(
            ...     docs_url="https://example.test/docs",
            ...     category="search",
            ...     label="Search",
            ...     title="Search API",
            ... )
            >>> page.title
            'Search API'
    """

    model_config = ConfigDict(extra="forbid")

    docs_url: str
    category: str
    label: str
    title: str
    summary: str | None = None
    about: str | None = None
    example_use_case: str | None = None
    endpoint_url: str | None = None
    parsed_parameters: list[ParsedParameterDoc] = Field(default_factory=list)
    examples: list[ParsedExampleDoc] = Field(default_factory=list)
    premium_signals: list[str] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)
    raw_text: str | None = None
    legacy: bool = False


class ParameterIR(BaseModel):
    """Normalized parameter description.

    Args:
        name:
            Canonical parameter name.
        location:
            Parameter location such as ``query`` or ``path``.
        required:
            Whether the parameter appears required.
        type_hint:
            Best-effort type hint.
        description:
            Human-readable description.
        default:
            Optional default value.
        example:
            Optional example value.
        enum_values:
            Optional allowed values.
        aliases:
            Alternate names discovered during normalization.
        source:
            Normalization source.

    Returns:
        A validated parameter description.

    Raises:
        ValueError:
            Raised by Pydantic validation.

    Examples:
        ::
            >>> ParameterIR(name="symbol", required=True).required
            True
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    location: str = "query"
    required: bool = False
    type_hint: str | None = None
    description: str | None = None
    default: str | None = None
    example: Any | None = None
    enum_values: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    source: str = "normalized"


class ResponseExampleIR(BaseModel):
    """Example response or example payload.

    Args:
        source:
            Example source such as ``docs`` or ``live_probe``.
        content_type:
            Payload content type.
        payload:
            Example payload.
        description:
            Optional description.

    Returns:
        A validated response example.

    Raises:
        ValueError:
            Raised by Pydantic validation.

    Examples:
        ::
            >>> ResponseExampleIR(source="docs", payload={"ok": True}).source
            'docs'
    """

    model_config = ConfigDict(extra="forbid")

    source: str = "docs"
    content_type: str = "application/json"
    payload: Any
    description: str | None = None


class EndpointIR(BaseModel):
    """Canonical intermediate representation for one endpoint.

    Args:
        operation_name:
            Stable internal operation identifier.
        docs_url:
            Source docs page URL.
        category:
            Category or tag.
        title:
            Endpoint title.
        summary:
            Short endpoint summary.
        about:
            Longer endpoint description.
        example_use_case:
            Example use case text.
        upstream_method:
            Upstream HTTP method.
        upstream_host:
            Upstream host.
        upstream_path:
            Upstream request path.
        wrapper_path:
            Generated internal wrapper path.
        auth_mode:
            Authentication convention.
        parameters:
            Normalized parameters.
        response_examples:
            Response examples.
        premium_signals:
            Premium hints from docs.
        parse_warnings:
            Non-fatal parser warnings.
        deprecated:
            Whether the docs page appears legacy or deprecated.
        docs_confidence:
            Confidence score for docs extraction.
        verification_status:
            Verification state.
        verification_notes:
            Optional verification note.

    Returns:
        A validated endpoint IR record.

    Raises:
        ValueError:
            Raised by Pydantic validation.

    Examples:
        ::
            >>> endpoint = EndpointIR(
            ...     operation_name="get_company_profile",
            ...     docs_url="https://example.test/docs/profile",
            ...     category="company",
            ...     title="Company Profile API",
            ...     summary="Get profile.",
            ...     upstream_method=HttpMethod.GET,
            ...     upstream_host="https://financialmodelingprep.com",
            ...     upstream_path="/stable/profile",
            ...     wrapper_path="/fmp/company/get-company-profile",
            ... )
            >>> endpoint.wrapper_path
            '/fmp/company/get-company-profile'
    """

    model_config = ConfigDict(extra="forbid")

    operation_name: str
    docs_url: str
    category: str
    title: str
    summary: str
    about: str | None = None
    example_use_case: str | None = None
    upstream_method: HttpMethod = HttpMethod.GET
    upstream_host: str = "https://financialmodelingprep.com"
    upstream_path: str
    wrapper_path: str
    auth_mode: AuthMode = AuthMode.APIKEY_QUERY
    parameters: list[ParameterIR] = Field(default_factory=list)
    response_examples: list[ResponseExampleIR] = Field(default_factory=list)
    premium_signals: list[str] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)
    deprecated: bool = False
    docs_confidence: float = 0.5
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verification_notes: str | None = None

    @computed_field
    @property
    def slug(self) -> str:
        """Return a filesystem-safe slug derived from the operation name.

        Args:
            None.

        Returns:
            A slug string.

        Raises:
            None.

        Examples:
            ::
                >>> EndpointIR(
                ...     operation_name="get_quote",
                ...     docs_url="https://example.test/docs",
                ...     category="quotes",
                ...     title="Quote API",
                ...     summary="Get quote.",
                ...     upstream_path="/stable/quote",
                ...     wrapper_path="/fmp/quotes/get-quote",
                ... ).slug
                'get-quote'
        """
        return self.operation_name.replace("_", "-")


class CrawlStats(BaseModel):
    """Aggregate crawl statistics.

    Args:
        discovered_pages:
            Number of discovered page links.
        parsed_pages:
            Number of successfully parsed pages.
        cached_pages:
            Number of pages written to cache.
        premium_flagged_pages:
            Number of pages with premium signals.
        warning_count:
            Total warning count.

    Returns:
        A validated stats object.

    Raises:
        ValueError:
            Raised by Pydantic validation.

    Examples:
        ::
            >>> CrawlStats().parsed_pages
            0
    """

    model_config = ConfigDict(extra="forbid")

    discovered_pages: int = 0
    parsed_pages: int = 0
    cached_pages: int = 0
    premium_flagged_pages: int = 0
    warning_count: int = 0


class ManifestIR(BaseModel):
    """Build manifest for one compiler run.

    Args:
        source:
            Human-readable source name.
        source_urls:
            Source entrypoint URLs.
        generated_at:
            UTC timestamp for the build.
        parser_version:
            Internal parser version.
        stats:
            Aggregate crawl stats.
        warnings:
            Global warnings.

    Returns:
        A validated manifest.

    Raises:
        ValueError:
            Raised by Pydantic validation.

    Examples:
        ::
            >>> ManifestIR(source="fixtures", source_urls=["file://x"]).parser_version
            '0.3.0'
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    source_urls: list[str]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parser_version: str = "0.4.0"
    stats: CrawlStats = Field(default_factory=CrawlStats)
    warnings: list[str] = Field(default_factory=list)


class CatalogIR(BaseModel):
    """Full normalized catalog emitted by the compiler.

    Args:
        source:
            Human-readable source name.
        source_urls:
            Source entrypoint URLs.
        generated_at:
            UTC timestamp for the catalog build.
        endpoints:
            Normalized endpoints.
        manifest:
            Build manifest.

    Returns:
        A validated catalog.

    Raises:
        ValueError:
            Raised by Pydantic validation.

    Examples:
        ::
            >>> catalog = CatalogIR(
            ...     source="fixtures",
            ...     source_urls=["file://fixtures"],
            ...     endpoints=[],
            ...     manifest=ManifestIR(source="fixtures", source_urls=["file://fixtures"]),
            ... )
            >>> catalog.endpoint_count
            0
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    source_urls: list[str]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    endpoints: list[EndpointIR]
    manifest: ManifestIR

    @computed_field
    @property
    def endpoint_count(self) -> int:
        """Return the number of endpoints in the catalog.

        Args:
            None.

        Returns:
            Endpoint count.

        Raises:
            None.

        Examples:
            ::
                >>> CatalogIR(
                ...     source="x",
                ...     source_urls=["y"],
                ...     endpoints=[],
                ...     manifest=ManifestIR(source="x", source_urls=["y"]),
                ... ).endpoint_count
                0
        """
        return len(self.endpoints)
