from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pymongo.collection import Collection

from .normalization import normalize, tokens


class SearchDocumentsRepository:
    """MongoDB retrieval layer for the Phase 1 canonical search documents."""

    def __init__(self, collection: Collection):
        self.collection = collection

    def ensure_indexes(self) -> None:
        # Phase 1 replaces the earlier bootstrap index definitions. MongoDB
        # permits only one text index per collection, so remove the old V1
        # text index before creating the V2 canonical index.
        for legacy_name in (
            "search_documents_text_v1",
            "search_documents_entity_live_status_v1",
            "search_documents_location_v1",
            "search_documents_entity_id_v1",
            "search_documents_aliases_v1",
        ):
            try:
                self.collection.drop_index(legacy_name)
            except Exception:
                pass

        # The canonical V6 ai_search_text is the primary searchable payload.
        # Keep title/taxonomy fields separately weighted so exact identity
        # signals outrank broad document matches.
        self.collection.create_index(
            [
                ("title", "text"),
                ("entity_name", "text"),
                ("category", "text"),
                ("company_name", "text"),
                ("user_name", "text"),
                ("ai_keywords", "text"),
                ("city", "text"),
                ("country", "text"),
                ("ai_search_text", "text"),
            ],
            name="search_documents_text_v2",
            weights={
                "title": 12,
                "entity_name": 10,
                "category": 7,
                "company_name": 6,
                "user_name": 5,
                "ai_keywords": 7,
                "city": 5,
                "country": 5,
                "ai_search_text": 3,
            },
        )

        self.collection.create_index(
            [("entity_type", 1), ("is_live", 1), ("status_normalized", 1)],
            name="search_documents_entity_live_status_v2",
        )

        self.collection.create_index(
            [
                ("city_normalized", 1),
                ("country_normalized", 1),
            ],
            name="search_documents_location_v2",
        )

        self.collection.create_index(
            [("entity_type", 1), ("entity_id", 1)],
            name="search_documents_entity_id_v2",
            unique=True,
        )

        self.collection.create_index(
            [("aliases_normalized", 1)],
            name="search_documents_aliases_v2",
        )

        self.collection.create_index(
            [("expires_at", 1)],
            name="search_documents_expiry_v2",
        )

    def vocabulary(self, limit: int = 50000) -> list[str]:
        """Build a bounded typo vocabulary from compact searchable fields.

        We intentionally do not tokenize the full ai_search_text collection
        here; it can contain hundreds of thousands of large documents.
        """
        projection = {
            "_id": 0,
            "title": 1,
            "entity_name": 1,
            "category": 1,
            "company_name": 1,
            "user_name": 1,
            "ai_keywords": 1,
            "aliases": 1,
            "city": 1,
            "country": 1,
        }

        values: set[str] = set()

        for doc in self.collection.find({}, projection):
            for field in projection:
                if field == "_id":
                    continue
                value = doc.get(field, [])
                if isinstance(value, str):
                    value = [value]
                elif not isinstance(value, list):
                    value = [value]

                for item in value:
                    if not isinstance(item, str):
                        continue
                    values.update(
                        token
                        for token in re.findall(
                            r"[a-z0-9][a-z0-9&-]*",
                            item.casefold(),
                        )
                        if len(token) >= 3
                    )

            if len(values) >= limit:
                break

        return sorted(values)[:limit]

    @staticmethod
    def _filter(
        entity: str | None,
        city: str | None,
        country: str | None,
        status: str | None,
        is_live: bool | None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {}

        if entity:
            query["entity_type"] = entity

        if is_live is not None:
            query["is_live"] = is_live

            # Treat an expired document as not live even if a stale source row
            # still has is_live=true.
            if is_live is True:
                query["$or"] = [
                    {"expires_at": None},
                    {"expires_at": {"$exists": False}},
                    {"expires_at": {"$gte": datetime.now(timezone.utc)}},
                ]

        if status:
            query["status_normalized"] = normalize(status)

        if city:
            query["city_normalized"] = normalize(city)

        if country:
            query["country_normalized"] = normalize(country)

        return query

    def search(
        self,
        query: str,
        *,
        entity: str | None,
        city: str | None,
        country: str | None,
        status: str | None,
        is_live: bool | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        filters = self._filter(entity, city, country, status, is_live)
        q_tokens = tokens(query)

        if not q_tokens:
            return []

        # Candidate pool is intentionally larger than the API limit so the
        # application ranking layer can apply exact/phrase/alias signals.
        candidate_limit = 100

        docs: list[dict[str, Any]] = []

        # MongoDB text search is the primary lexical retrieval mechanism.
        # Phrase quotes are used only when the query contains multiple tokens;
        # this preserves normal keyword behavior for ordinary searches.
        text_query = query.strip()
        mongo_query: dict[str, Any] = {
            "$text": {"$search": text_query},
            **filters,
        }

        projection = {
            "text_score": {"$meta": "textScore"},
        }

        try:
            docs = list(
                self.collection.find(
                    mongo_query,
                    projection,
                )
                .sort([("text_score", {"$meta": "textScore"})])
                .limit(candidate_limit)
            )
        except Exception:
            # A malformed text query should not take the API down.
            docs = []

        # Alias / normalized field fallback. This is especially useful after
        # typo correction when MongoDB's tokenizer does not find the original.
        fallback_terms = []
        for token in q_tokens:
            if len(token) >= 3:
                fallback_terms.append(
                    {
                        "$or": [
                            {"aliases_normalized": {"$regex": re.escape(token)}},
                            {"title_normalized": {"$regex": re.escape(token)}},
                            {"ai_keywords_normalized": {"$regex": re.escape(token)}},
                        ]
                    }
                )

        if fallback_terms:
            fallback_query = {
                **filters,
                "$and": fallback_terms,
            }
            fallback = list(
                self.collection.find(
                    fallback_query,
                    projection,
                ).limit(candidate_limit)
            )
            seen = {str(d.get("_id")) for d in docs}
            docs.extend(
                d for d in fallback
                if str(d.get("_id")) not in seen
            )

        return docs[:candidate_limit]
