"""HTTP client utilities for :mod:`~fmp_docs_compiler`.

Purpose:
    Provide a resilient async HTTP client with bounded request rate,
    backoff-aware retries, and small convenience helpers for text and JSON
    retrieval.

Design:
    The client combines:

    - a token-bucket limiter
    - bounded retry logic
    - support for ``Retry-After``
    - explicit helper methods for HTML-oriented use cases

Attributes:
    None.

Examples:
    ::
        >>> RetryConfig().max_retries >= 1
        True
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, field_validator


class RetryConfig(BaseModel):
    """Configuration for the resilient async client.

    Args:
        requests_per_second:
            Token refill rate per second.
        burst_capacity:
            Maximum number of immediately available tokens.
        max_retries:
            Maximum retries for transient failures.
        timeout_seconds:
            Request timeout.
        user_agent:
            HTTP user-agent header.

    Returns:
        A validated retry configuration.

    Raises:
        ValueError:
            Raised when values are invalid.

    Examples:
        ::
            >>> RetryConfig(requests_per_second=2.0).requests_per_second
            2.0
    """

    model_config = ConfigDict(extra="forbid")

    requests_per_second: float = 2.0
    burst_capacity: int = 4
    max_retries: int = 4
    timeout_seconds: float = 30.0
    user_agent: str = "Mozilla/5.0 (compatible; fmp-docs-compiler/0.3.0)"

    @field_validator("requests_per_second")
    @classmethod
    def _validate_rps(cls, value: float) -> float:
        """Validate positive request rate.

        Args:
            value:
                Candidate value.

        Returns:
            The validated value.

        Raises:
            ValueError:
                Raised when the value is not positive.

        Examples:
            ::
                >>> RetryConfig(requests_per_second=1.0).requests_per_second
                1.0
        """
        if value <= 0:
            raise ValueError("requests_per_second must be > 0")
        return value


class AsyncTokenBucket:
    """Simple async token-bucket limiter.

    Args:
        rate:
            Token refill rate.
        capacity:
            Maximum token count.

    Returns:
        A limiter instance.

    Raises:
        ValueError:
            Raised for invalid rate or capacity.

    Examples:
        ::
            >>> AsyncTokenBucket(rate=2.0, capacity=2).capacity
            2
    """

    def __init__(self, rate: float, capacity: int) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire one token.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.

        Examples:
        ::
            >>> isinstance(AsyncTokenBucket(rate=1.0, capacity=1), AsyncTokenBucket)
            True
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._updated_at
                self._updated_at = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                await asyncio.sleep((1 - self._tokens) / self.rate)


@dataclass(slots=True)
class RequestMetadata:
    """Metadata describing one request attempt.

    Args:
        attempts:
            Number of attempts used.
        rate_limited:
            Whether a 429 was observed.

    Returns:
        A metadata object.

    Raises:
        None.

    Examples:
        ::
            >>> RequestMetadata(attempts=1, rate_limited=False).attempts
            1
    """

    attempts: int
    rate_limited: bool


class ResilientAsyncClient:
    """Resilient async HTTP client.

    Args:
        retry_config:
            Retry configuration.

    Returns:
        A configured client manager.

    Raises:
        ValueError:
            Raised when the configuration is invalid.

    Examples:
        ::
            >>> isinstance(ResilientAsyncClient(RetryConfig()), ResilientAsyncClient)
            True
    """

    def __init__(self, retry_config: RetryConfig) -> None:
        self.retry_config = retry_config
        self._limiter = AsyncTokenBucket(
            rate=retry_config.requests_per_second,
            capacity=retry_config.burst_capacity,
        )
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ResilientAsyncClient":
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=True,
                headers={"user-agent": self.retry_config.user_agent},
                timeout=httpx.Timeout(self.retry_config.timeout_seconds),
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def request(self, url: str) -> tuple[httpx.Response, RequestMetadata]:
        """Issue a GET request with retries and rate limiting.

        Args:
            url:
                Target URL.

        Returns:
            A tuple of response and request metadata.

        Raises:
            RuntimeError:
                Raised when the client is not open.
            httpx.HTTPError:
                Raised when transport errors persist.

        Examples:
            ::
                >>> callable(ResilientAsyncClient.request)
                True
        """
        if self._client is None:
            raise RuntimeError("Client is not open.")

        rate_limited = False
        last_error: Exception | None = None
        response: httpx.Response | None = None

        for attempt in range(1, self.retry_config.max_retries + 2):
            await self._limiter.acquire()
            try:
                response = await self._client.get(url)
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(
                    min(2 ** (attempt - 1), 8) + random.uniform(0.0, 0.25)
                )
                continue

            if response.status_code != 429:
                return response, RequestMetadata(
                    attempts=attempt, rate_limited=rate_limited
                )

            rate_limited = True
            retry_after = response.headers.get("retry-after")
            sleep_seconds = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else min(2 ** (attempt - 1), 8)
            )
            await asyncio.sleep(sleep_seconds + random.uniform(0.05, 0.25))

        if response is not None:
            return response, RequestMetadata(
                attempts=self.retry_config.max_retries + 1, rate_limited=rate_limited
            )
        if last_error is not None:
            raise last_error
        raise RuntimeError("Request failed without response or exception.")

    async def get_text(self, url: str) -> str:
        """Fetch text content for a URL.

        Args:
            url:
                Target URL.

        Returns:
            Response text.

        Raises:
            httpx.HTTPError:
                Raised when the final HTTP status is not successful.
            RuntimeError:
                Raised when the client is not open.

        Examples:
            ::
                >>> callable(ResilientAsyncClient.get_text)
                True
        """
        response, _ = await self.request(url)
        response.raise_for_status()
        return response.text

    async def get_json(self, url: str) -> Any:
        """Fetch JSON content for a URL.

        Args:
            url:
                Target URL.

        Returns:
            Decoded JSON payload.

        Raises:
            httpx.HTTPError:
                Raised when the final HTTP status is not successful.
            RuntimeError:
                Raised when the client is not open.

        Examples:
            ::
                >>> callable(ResilientAsyncClient.get_json)
                True
        """
        response, _ = await self.request(url)
        response.raise_for_status()
        return response.json()
