"""Generated request models.

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


class GetAvailableCountriesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    pass


class GetAvailableSectorsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    pass


class GetLatestSecFilingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    from_: str | None = Field(default=None, alias="from")
    to: str | None = None
    page: int | None = None
    limit: int | None = None


class GetSymbolSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    query: str = ...
    exchange: str | None = None
    limit: int | None = None


__all__ = [
    "GetAvailableCountriesRequest",
    "GetAvailableSectorsRequest",
    "GetLatestSecFilingsRequest",
    "GetSymbolSearchRequest",
]
