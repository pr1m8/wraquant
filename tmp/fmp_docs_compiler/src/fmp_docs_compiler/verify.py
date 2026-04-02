"""Optional live verification for :mod:`~fmp_docs_compiler`.

Purpose:
    Verify generated catalog endpoints against live example URLs without
    making verification a prerequisite for compilation.

Design:
    Verification is intentionally separate from parsing and normalization so
    docs compilation remains deterministic and offline-friendly.

Attributes:
    None.

Examples:
    ::
        >>> callable(classify_verification)
        True
"""

from __future__ import annotations

from copy import deepcopy
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .http import ResilientAsyncClient, RetryConfig
from .models import CatalogIR, VerificationStatus


def classify_verification(status_code: int, body_text: str) -> VerificationStatus:
    lowered = body_text.lower()
    if status_code == 429:
        return VerificationStatus.RATE_LIMITED
    if status_code == 403 and "api key" in lowered and "invalid" in lowered:
        return VerificationStatus.INVALID_API_KEY
    if status_code == 403 and any(
        token in lowered for token in ("premium", "upgrade", "plan")
    ):
        return VerificationStatus.PLAN_GATED
    if 200 <= status_code < 300:
        return VerificationStatus.VERIFIED
    return VerificationStatus.FAILED


class CatalogVerifier:
    """Optional live verifier for a normalized catalog."""

    def __init__(self, retry_config: RetryConfig) -> None:
        self.retry_config = retry_config

    async def verify_catalog(
        self, catalog: CatalogIR, api_key: str | None
    ) -> CatalogIR:
        verified = deepcopy(catalog)
        if not api_key:
            for endpoint in verified.endpoints:
                endpoint.verification_status = VerificationStatus.SKIPPED
                endpoint.verification_notes = "Skipped because no API key was provided."
            return verified

        async with ResilientAsyncClient(self.retry_config) as client:
            for endpoint in verified.endpoints:
                url = self._build_verification_url(
                    endpoint.upstream_host, endpoint.upstream_path, api_key
                )
                response, _ = await client.request(url)
                endpoint.verification_status = classify_verification(
                    response.status_code, response.text
                )
                endpoint.verification_notes = f"HTTP {response.status_code}"
        return verified

    def _build_verification_url(
        self, upstream_host: str, upstream_path: str, api_key: str
    ) -> str:
        parsed = urlparse(f"{upstream_host}{upstream_path}")
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["apikey"] = api_key
        return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
