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


class CompanyRef(BaseModel):
    id: Any = None
    name: str | None = None
    slug: str | None = None
    url: str | None = None


class SearchResult(BaseModel):
    """Normalized result (results.py). Every result is a real record."""

    id: str | None = None
    entity_type: str
    entity_id: str
    doc_id: str | None = None
    title: str
    slug: str | None = None
    url: str | None = None
    url_source: Literal["record", "template"] | None = None
    description: str | None = None
    snippet: str | None = None
    company: str | None = None
    company_ref: CompanyRef | None = None
    location: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None
    external_url: str | None = None
    external_label: str | None = None
    image: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    match_type: str | None = None
    score: float
    matched_by: list[str] = Field(default_factory=list)
    corrected_query: str | None = None
    # Position in the list shown to the user (compare/detail responses).
    number: int | None = None


class SearchUnderstanding(BaseModel):
    intent: str = "search"
    entity: str | None = None
    entity_reason: str | None = None
    is_faq: bool = False
    facet: str | None = None
    related: str | None = None
    notes: list[str] = Field(default_factory=list)
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
    entity_source: str | None = None
    location_text: str | None = None
    strict_filters: list[str] = Field(default_factory=list)
    browse: bool = False


class SearchResponse(BaseModel):
    query: str
    corrected_query: str | None = None
    total: int
    results: list[SearchResult]
    message: str | None = None
    related_results: list[SearchResult] = Field(default_factory=list)
    facets: list[dict[str, Any]] = Field(default_factory=list)
    understanding: SearchUnderstanding | None = None


ConversationId = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_-]{8,128}$")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = ConversationId
    limit: int = Field(default=5, ge=1, le=5)


ChatAction = Literal[
    "search",
    "clarify",
    "more",
    "compare",
    "reset",
    "detail",
    "related_entity",
    "smalltalk",
    "faq",
    "facet",
]


class ResultReference(BaseModel):
    number: int
    key: str
    entity_type: str | None = None
    entity_id: str | None = None
    title: str | None = None
    url: str | None = None
    related: bool = False


class Comparison(BaseModel):
    fields: list[dict[str, str]] = Field(default_factory=list)
    items: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    conversation_id: str
    action: ChatAction
    answer: str
    results: list[SearchResult] = Field(default_factory=list)
    related_results: list[SearchResult] = Field(default_factory=list)
    message: str | None = None
    understanding: SearchUnderstanding | dict[str, Any] | None = None
    state: dict[str, Any] = Field(default_factory=dict)
    references: list[ResultReference] = Field(default_factory=list)
    comparison: Comparison | None = None
    detail: dict[str, Any] | None = None
    suggestions: list[str] = Field(default_factory=list)
    facets: list[dict[str, Any]] = Field(default_factory=list)
    llm: dict[str, Any] | None = None
    request_id: str | None = None


class ConversationView(BaseModel):
    conversation_id: str
    state: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: Any = None
