from __future__ import annotations

import re
from typing import Any

from pymongo.collection import Collection

from .normalization import normalize, tokens


class SearchDocumentsRepository:
    def __init__(self, collection: Collection):
        self.collection = collection

    def ensure_indexes(self) -> None:
        self.collection.create_index(
            [
                ("title", "text"),
                ("aliases", "text"),
                ("keywords", "text"),
                ("category", "text"),
                ("location.city", "text"),
                ("location.country", "text"),
                ("description", "text"),
                ("search_text", "text"),
            ],
            name="search_documents_text_v1",
            weights={
                "title": 10,
                "aliases": 8,
                "keywords": 6,
                "category": 4,
                "location.city": 3,
                "location.country": 3,
                "search_text": 2,
                "description": 1,
            },
        )
        self.collection.create_index(
            [("entity_type", 1), ("is_live", 1), ("status", 1)],
            name="search_documents_entity_live_status_v1",
        )
        self.collection.create_index(
            [("location.city_normalized", 1), ("location.country_normalized", 1)],
            name="search_documents_location_v1",
        )
        self.collection.create_index(
            [("entity_type", 1), ("entity_id", 1)],
            name="search_documents_entity_id_v1",
            unique=True,
        )
        self.collection.create_index(
            [("aliases_normalized", 1)],
            name="search_documents_aliases_v1",
        )

    def vocabulary(self, limit: int = 10000) -> list[str]:
        projection = {
            "_id": 0,
            "title": 1,
            "keywords": 1,
            "aliases": 1,
            "category": 1,
        }
        cursor = self.collection.find({}, projection).limit(limit)
        values: set[str] = set()
        for doc in cursor:
            for field in ("title", "keywords", "aliases", "category"):
                value = doc.get(field, [])
                if isinstance(value, str):
                    value = [value]
                for item in value:
                    if isinstance(item, str):
                        values.update(
                            token for token in re.findall(r"[a-z0-9][a-z0-9&-]*", item.casefold())
                            if len(token) >= 3
                        )
        return list(values)

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
        if status:
            query["status_normalized"] = normalize(status)
        if city:
            query["location.city_normalized"] = normalize(city)
        if country:
            query["location.country_normalized"] = normalize(country)
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

        # MongoDB text search handles phrase/keyword retrieval. The application
        # layer below performs deterministic exact/alias scoring.
        mongo_query = {"$text": {"$search": query}, **filters}
        projection = {"score": {"$meta": "textScore"}}
        docs = list(
            self.collection.find(
                mongo_query,
                projection,
            ).sort([("score", {"$meta": "textScore"})]).limit(50)
        )

        # Alias-only / corrected-term fallback. This also gives short queries
        # a path when MongoDB's text parser does not produce candidates.
        if len(docs) < 20:
            regex_terms = [
                {"aliases_normalized": {"$regex": re.escape(t)}}
                for t in q_tokens
                if len(t) >= 3
            ]
            if regex_terms:
                fallback = list(
                    self.collection.find(
                        {"$and": [filters, {"$or": regex_terms}]},
                        projection,
                    ).limit(50)
                )
                seen = {str(d.get("_id")) for d in docs}
                docs.extend(d for d in fallback if str(d.get("_id")) not in seen)

        return docs[: max(5, min(limit, 5))]
