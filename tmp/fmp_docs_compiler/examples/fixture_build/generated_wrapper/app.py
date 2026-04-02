"""Generated FastAPI application.

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

app = FastAPI(title="FMP Wrapper", version="0.4.0")


@app.get("/health", operation_id="health_check")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/fmp/reference-data/get-available-countries",
    operation_id="get_available_countries",
)
async def get_available_countries(
    request: GetAvailableCountriesRequest = Depends(),
    client: FMPUpstreamClient = Depends(get_upstream_client),
    settings: Settings = Depends(get_settings),
):
    del settings
    payload = request.model_dump(exclude_none=True, by_alias=True)
    return await client.call(path="/stable/available-countries", query=payload)


@app.get(
    "/fmp/reference-data/get-available-sectors", operation_id="get_available_sectors"
)
async def get_available_sectors(
    request: GetAvailableSectorsRequest = Depends(),
    client: FMPUpstreamClient = Depends(get_upstream_client),
    settings: Settings = Depends(get_settings),
):
    del settings
    payload = request.model_dump(exclude_none=True, by_alias=True)
    return await client.call(path="/stable/available-sectors", query=payload)


@app.get("/fmp/sec/get-latest-sec-filings", operation_id="get_latest_sec_filings")
async def get_latest_sec_filings(
    request: GetLatestSecFilingsRequest = Depends(),
    client: FMPUpstreamClient = Depends(get_upstream_client),
    settings: Settings = Depends(get_settings),
):
    del settings
    payload = request.model_dump(exclude_none=True, by_alias=True)
    return await client.call(path="/stable/sec-filings-financials", query=payload)


@app.get("/fmp/search/get-symbol-search", operation_id="get_symbol_search")
async def get_symbol_search(
    request: GetSymbolSearchRequest = Depends(),
    client: FMPUpstreamClient = Depends(get_upstream_client),
    settings: Settings = Depends(get_settings),
):
    del settings
    payload = request.model_dump(exclude_none=True, by_alias=True)
    return await client.call(path="/stable/search-symbol", query=payload)
