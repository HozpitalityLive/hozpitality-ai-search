from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EntityType = Literal[
    "job",
    "professional",
    "company",
    "product",
    "article",
    "event",
    "award",
    "faq",
]


class LocationFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str | None = None
    country: str | None = None


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=1, max_length=300)
    entity: EntityType | None = None
    city: str | None = Field(default=None, max_length=120)
    country: str | None = Field(default=None, max_length=120)
    status: str | None = Field(default=None, max_length=80)
    is_live: bool | None = None
    limit: int = Field(default=5, ge=1, le=5)


class SearchResult(BaseModel):
    entity_type: str
    entity_id: str
    title: str
    description: str | None = None
    location: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None
    url: str | None = None
    image: str | None = None
    score: float
    matched_by: list[str] = Field(default_factory=list)
    corrected_query: str | None = None


class SearchResponse(BaseModel):
    query: str
    corrected_query: str | None = None
    total: int
    results: list[SearchResult]
