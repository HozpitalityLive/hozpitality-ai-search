"""
Hozpitality V6 schema intelligence.

Runtime database metadata is authoritative. This module intentionally avoids
hard-coding individual category names, content-type names, or country names.
It resolves those values from PostgreSQL and provides compact semantic context
for the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ResolvedValue:
    id: int
    name: str
    score: float = 1.0


class HozpitalitySchemaIntelligence:
    """Database-backed semantic resolver for Hozpitality V6."""

    CORE_TABLES = {
        "jobs": "base_job",
        "professionals": "professionals",
        "companies": "companies",
        "accounts": "user_accounts",
        "articles": "base_article",
        "events": "base_event",
        "products": "marketplace_product",
        "faqs": "base_faq",
        "search_index": "master_search_mastersearchindex",
        "categories": "base_category",
        "countries": "countries",
    }

    def __init__(self, connect):
        self._connect = connect
        self._cache: dict[str, Any] = {}

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", value.lower()))

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        import psycopg2.extras

        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())

    def _cached(self, key: str, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def categories(self) -> list[dict]:
        return self._cached(
            "categories",
            lambda: self._fetchall(
                """
                SELECT id, name
                FROM base_category
                WHERE name IS NOT NULL AND BTRIM(name) <> ''
                ORDER BY LENGTH(name) DESC
                """
            ),
        )

    def countries(self) -> list[dict]:
        return self._cached(
            "countries",
            lambda: self._fetchall(
                """
                SELECT id, name, country_code, code, ac_name
                FROM countries
                WHERE name IS NOT NULL AND BTRIM(name) <> ''
                ORDER BY LENGTH(name) DESC
                """
            ),
        )

    def content_types(self) -> list[dict]:
        return self._cached(
            "content_types",
            lambda: self._fetchall(
                """
                SELECT id, app_label, model
                FROM django_content_type
                WHERE model IS NOT NULL
                ORDER BY app_label, model
                """
            ),
        )

    def relationships(self) -> list[dict]:
        return self._cached(
            "relationships",
            lambda: self._fetchall(
                """
                SELECT
                    tc.table_schema AS source_schema,
                    tc.table_name AS source_table,
                    kcu.column_name AS source_column,
                    ccu.table_schema AS target_schema,
                    ccu.table_name AS target_table,
                    ccu.column_name AS target_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.constraint_schema = kcu.constraint_schema
                 AND tc.table_schema = kcu.table_schema
                 AND tc.table_name = kcu.table_name
                JOIN information_schema.constraint_column_usage ccu
                  ON ccu.constraint_name = tc.constraint_name
                 AND ccu.constraint_schema = tc.constraint_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public'
                ORDER BY tc.table_name, kcu.column_name
                """
            ),
        )

    def _resolve(self, message: str, values: list[dict], aliases: tuple[str, ...]) -> Optional[ResolvedValue]:
        text = self._normalize(message)
        tokens = self._tokens(text)

        # Prefer exact/substring name matches.
        for row in values:
            name = self._normalize(row.get("name"))
            if name and (name in text or text == name):
                return ResolvedValue(int(row["id"]), row["name"], 1.0)

        # Support country aliases stored in additional columns.
        for row in values:
            candidates = [row.get("name"), *(row.get(a) for a in aliases)]
            for candidate in candidates:
                candidate = self._normalize(candidate)
                if candidate and candidate in text:
                    return ResolvedValue(int(row["id"]), row.get("name") or candidate, 0.95)

        # Token overlap fallback, requiring at least half the value's tokens.
        best = None
        best_score = 0.0
        for row in values:
            name = self._normalize(row.get("name"))
            name_tokens = self._tokens(name)
            if not name_tokens:
                continue
            overlap = len(tokens & name_tokens)
            score = overlap / len(name_tokens)
            if overlap and score >= 0.5 and score > best_score:
                best_score = score
                best = ResolvedValue(int(row["id"]), row["name"], score)
        return best

    def resolve_category(self, message: str) -> Optional[ResolvedValue]:
        return self._resolve(message, self.categories(), ())

    def resolve_country(self, message: str) -> Optional[ResolvedValue]:
        return self._resolve(
            message,
            self.countries(),
            ("country_code", "code", "ac_name"),
        )

    def resolve_content_type(self, message: str) -> Optional[ResolvedValue]:
        text = self._normalize(message)
        # Content type is intentionally inferred from the user's wording, but
        # the actual IDs/models always come from django_content_type.
        hints = {
            "article": ("article", "articles", "news", "blog", "blogs", "story", "stories"),
            "job": ("job", "jobs", "vacancy", "vacancies", "opening", "openings"),
            "event": ("event", "events"),
            "professional": ("professional", "professionals", "candidate", "candidates", "profile", "profiles"),
            "company": ("company", "companies", "employer", "employers"),
            "product": ("product", "products", "marketplace", "supplier", "suppliers"),
            "faq": ("faq", "faqs", "question", "questions"),
        }
        model = next((m for m, words in hints.items() if any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)), None)
        if not model:
            return None

        matches = [x for x in self.content_types() if self._normalize(x.get("model")) == model]
        if matches:
            # Hozpitality has duplicate Django content types in some apps.
            # The master search index uses the canonical public `base` content
            # types (e.g. base.job = 18, while app.job may also exist).
            matches.sort(key=lambda x: (
                0 if self._normalize(x.get("app_label")) == "base" else 1,
                0 if self._normalize(x.get("model")) == model else 1,
                int(x.get("id") or 0),
            ))
            return ResolvedValue(int(matches[0]["id"]), matches[0]["model"], 1.0)
        return None

    def article_query_context(self, message: str) -> dict[str, Any]:
        category = self.resolve_category(message)
        country = self.resolve_country(message)
        return {
            "category": category,
            "country": country,
            "terms": self._tokens(message),
        }

    def system_context(self) -> str:
        """Compact, factual semantic context for the LLM prompt."""
        rels = self.relationships()
        lines = [
            "Hozpitality schema intelligence is database-backed.",
            "Never invent category names, country names, content-type IDs, table names, columns, or joins.",
            "Core source tables: " + ", ".join(f"{k}={v}" for k, v in self.CORE_TABLES.items()) + ".",
            "Article category: base_article.category_id -> base_category.id.",
            "Article locations: base_article_location.article_id -> base_article.id and country_id -> countries.id.",
            "Polymorphic search: master_search_mastersearchindex.content_type_id -> django_content_type.id; object_id identifies the source object.",
            "Polymorphic tables such as base_bookmark/base_impression/base_interaction use content_type_id + object_id and must not be joined to an arbitrary source table without resolving content_type.",
            "Use source tables for authoritative fields; use master_search_mastersearchindex for cross-module discovery.",
        ]
        # Include only relevant relationships to keep prompt size bounded.
        core = set(self.CORE_TABLES.values()) | {"base_category", "countries", "django_content_type", "base_article_location"}
        for r in rels:
            if r["source_table"] in core or r["target_table"] in core:
                lines.append(
                    f"FK: {r['source_table']}.{r['source_column']} -> "
                    f"{r['target_table']}.{r['target_column']}."
                )
        return "\n".join(lines)
