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


class SearchUnderstanding(BaseModel):
    intent: str = "search"
    entity: str | None = None
    keywords: list[str] = Field(default_factory=list)
    city: str | None = None
    country: str | None = None
    experience: int | None = None
    level: str | None = None
    department: str | None = None
    industry: str | None = None
    category: str | None = None
    date: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    clarification: str | None = None
    clarification_options: list[str] = Field(default_factory=list)
    corrected_keywords: list[str] = Field(default_factory=list)
    corrections: list[dict[str, Any]] = Field(default_factory=list)
    explicit_location: bool = False


class SearchResponse(BaseModel):
    query: str
    corrected_query: str | None = None
    total: int
    results: list[SearchResult]
    message: str | None = None
    related_results: list[SearchResult] = Field(default_factory=list)
    understanding: SearchUnderstanding | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=128)
    limit: int = Field(default=5, ge=1, le=5)


class ChatResponse(BaseModel):
    conversation_id: str
    action: Literal["search", "clarify", "more", "compare", "reset"]
    answer: str
    results: list[SearchResult] = Field(default_factory=list)
    related_results: list[SearchResult] = Field(default_factory=list)
    understanding: SearchUnderstanding | dict[str, Any] | None = None
    state: dict[str, Any] = Field(default_factory=dict)
