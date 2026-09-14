"""Deterministic query planner for Hozpitality AI Search V6."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SearchPlan:
    query: str
    normalized: str
    strategy: str
    entity_type: Optional[str] = None
    analytics: bool = False
    historical: bool = False
    limit: int = 10
    filters: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0


class QueryPlanner:
    """Cheap front-door query understanding; no LLM call required."""

    ANALYTICS = re.compile(
        r"\b(how many|count|average|avg|sum|total|maximum|max|minimum|min|"
        r"compare|comparison|trend|percentage|percent|growth|distribution|"
        r"breakdown|per (day|week|month|year)|top \d+|most|least)\b"
    )
    HISTORICAL = re.compile(r"\b(expired|historical|history|old|past|archived)\b")

    # Ordered from the most specific/common content types to the broader ones.
    ENTITY_HINTS = {
        "job": r"\b(job|jobs|vacancy|vacancies|career|careers|employment|opening|openings|position|positions)\b",
        "professional": r"\b(professional|professionals|candidate|candidates|profile|profiles|resume|cv)\b",
        "company": r"\b(company|companies|employer|employers|organization|organisations|organisation)\b",
        "article": r"\b(article|articles|news|blog|blogs|story|stories)\b",
        "event": r"\b(event|events|conference|conferences|expo|exhibition|exhibitions)\b",
        "product": r"\b(product|products|marketplace|supplier|suppliers)\b",
        "faq": r"\b(faq|faqs|question|questions|help)\b",
        "award": r"\b(award|awards|recognition|winner|winners|nomination|nominations)\b",
    }

    @staticmethod
    def normalize(query: str) -> str:
        query = query or ""
        query = query.replace("’", "'").replace("–", "-").replace("—", "-")
        query = re.sub(r"\s+", " ", query.strip().lower())
        return query

    def plan(self, query: str, limit: int = 10) -> SearchPlan:
        normalized = self.normalize(query)
        analytics = bool(self.ANALYTICS.search(normalized))
        historical = bool(self.HISTORICAL.search(normalized))

        entity = None
        for name, pattern in self.ENTITY_HINTS.items():
            if re.search(pattern, normalized):
                entity = name
                break

        if analytics:
            strategy = "SQL_ANALYTICS"
            confidence = 0.95
        elif entity:
            strategy = "GLOBAL_SEARCH"
            confidence = 0.94
        else:
            strategy = "GLOBAL_SEARCH"
            confidence = 0.72

        return SearchPlan(
            query=query,
            normalized=normalized,
            strategy=strategy,
            entity_type=entity,
            analytics=analytics,
            historical=historical,
            limit=max(1, min(int(limit), 20)),
            confidence=confidence,
        )
