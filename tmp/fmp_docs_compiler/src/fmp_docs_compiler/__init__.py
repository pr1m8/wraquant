"""Package overview for :mod:`~fmp_docs_compiler`.

Purpose:
    Provide a docs-first compiler that discovers Financial Modeling Prep
    documentation pages, parses them into structured intermediate
    representations, and generates wrapper artifacts such as OpenAPI,
    agent tool schemas, and FastAPI scaffolds.

Design:
    The package is intentionally split into stages:

    - discovery
    - parsing
    - normalization
    - optional verification
    - generation

    This allows offline fixture testing, deterministic compilation from docs
    alone, and optional live verification later.

Attributes:
    __all__:
        Curated public API.

Examples:
    ::
        >>> from fmp_docs_compiler import FMPDocsCompiler
        >>> callable(FMPDocsCompiler)
        True
"""

from .compiler import FMPDocsCompiler
from .generator_mcp import render_mcp_project
from .models import CatalogIR, EndpointIR, ManifestIR, RetryOutcome, VerificationStatus

__all__ = [
    "CatalogIR",
    "EndpointIR",
    "FMPDocsCompiler",
    "render_mcp_project",
    "ManifestIR",
    "RetryOutcome",
    "VerificationStatus",
]
