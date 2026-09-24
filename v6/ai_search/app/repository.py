from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pymongo.collection import Collection

from .normalization import normalize, tokens


class SearchDocumentsRepository:
    """MongoDB retrieval layer over the existing search_documents collection."""

    ENTITY_TYPES = {
        "job", "professional", "company", "product",
        "article", "event", "award", "faq",
    }

    def __init__(self, collection: Collection):
        self.collection = collection

    def ensure_indexes(self) -> None:
        """Keep the existing Mongo indexes intact.

        The production collection already has a MongoDB text index on
        ``ai_search_text``. MongoDB permits only one text index per collection,
        so Phase 1 must use that index rather than attempting to create a
        second text index or dropping production indexes.
        """
        required = {"_id_", "idx_ai_search_text"}
        try:
            names = {index["name"] for index in self.collection.list_indexes()}
            missing = required - names
            if missing:
                raise RuntimeError(
                    "Missing required MongoDB search index(es): "
                    + ", ".join(sorted(missing))
                )
        except RuntimeError:
            raise
        except Exception:
            # Connection errors are surfaced by the route/health endpoint.
            raise

    @staticmethod
    def _regex(value: str) -> dict[str, Any]:
        return {"$regex": re.escape(normalize(value)), "$options": "i"}

    @staticmethod
    def _country_terms(value: str) -> list[str]:
        normalized = normalize(value)
        aliases = {
            "uae": ["United Arab Emirates", "AE", "UAE"],
            "u.a.e": ["United Arab Emirates", "AE", "UAE"],
            "usa": ["United States", "US", "USA"],
            "us": ["United States", "US", "USA"],
            "uk": ["United Kingdom", "GB", "UK"],
            "ua": ["Ukraine", "UA"],
        }
        return aliases.get(normalized, [value])

    @staticmethod
    def _filter(
        entity: str | None,
        city: str | None,
        country: str | None,
        status: str | None,
        is_live: bool | None,
    ) -> dict[str, Any]:
        clauses: list[dict[str, Any]] = []

        if entity:
            clauses.append({"entity_type": entity})

        if is_live is not None:
            clauses.append({
                "$or": [
                    {"is_live": is_live},
                    {"metadata.is_live": is_live},
                ]
            })
            if is_live is True:
                # If an expiry exists, it must still be in the future.
                clauses.append({
                    "$or": [
                        {"expires_at": {"$exists": False}},
                        {"expires_at": None},
                        {"expires_at": {"$gte": datetime.now(timezone.utc)}},
                    ]
                })

        if status:
            pattern = re.escape(normalize(status))
            clauses.append({
                "$or": [
                    {"status": {"$regex": f"^{pattern}$", "$options": "i"}},
                    {"metadata.status": {"$regex": f"^{pattern}$", "$options": "i"}},
                    {"metadata.job_status": {"$regex": f"^{pattern}$", "$options": "i"}},
                ]
            })

        if city:
            pattern = re.escape(normalize(city))
            clauses.append({
                "$or": [
                    {"location.city": {"$regex": pattern, "$options": "i"}},
                    {"location.current_location": {"$regex": pattern, "$options": "i"}},
                    {"location.prime_city": {"$regex": pattern, "$options": "i"}},
                    {"company.city": {"$regex": pattern, "$options": "i"}},
                    {"author.city_town": {"$regex": pattern, "$options": "i"}},
                ]
            })

        if country:
            country_clauses = []
            for term in SearchDocumentsRepository._country_terms(country):
                pattern = re.escape(normalize(term))
                country_clauses.extend([
                    {"location.country.name": {"$regex": pattern, "$options": "i"}},
                    {"location.country.ac_name": {"$regex": pattern, "$options": "i"}},
                    {"location.country.code": {"$regex": f"^{pattern}$", "$options": "i"}},
                    {"location.countries.name": {"$regex": pattern, "$options": "i"}},
                    {"location.countries.code": {"$regex": f"^{pattern}$", "$options": "i"}},
                    {"country.name": {"$regex": pattern, "$options": "i"}},
                    {"country.code": {"$regex": f"^{pattern}$", "$options": "i"}},
                    {"company.country.name": {"$regex": pattern, "$options": "i"}},
                ])
            clauses.append({"$or": country_clauses})

        return {"$and": clauses} if clauses else {}

    @staticmethod
    def _projection() -> dict[str, Any]:
        return {
            "_id": 1,
            "entity_type": 1,
            "title": 1,
            "short_title": 1,
            "subtitle": 1,
            "description": 1,
            "question": 1,
            "answer": 1,
            "ai_search_text": 1,
            "search_aliases": 1,
            "search_keywords": 1,
            "keywords": 1,
            "location": 1,
            "country": 1,
            "category": 1,
            "company": 1,
            "author": 1,
            "user": 1,
            "professional": 1,
            "job": 1,
            "media": 1,
            "profile_image": 1,
            "cover_image": 1,
            "slug": 1,
            "links": 1,
            "source": 1,
            "status": 1,
            "is_live": 1,
            "metadata": 1,
            "created_at": 1,
            "dates": 1,
            "start_datetime": 1,
            "end_datetime": 1,
            "award_date": 1,
            "text_score": {"$meta": "textScore"},
        }

    def _exact_candidates(
        self,
        query: str,
        filters: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Retrieve high-confidence title/alias/keyword phrase matches first."""
        phrase = re.escape(normalize(query))
        if not phrase:
            return []

        field_clauses = [
            {"title": {"$regex": f"^{phrase}$", "$options": "i"}},
            {"search_aliases": {"$regex": f"^{phrase}$", "$options": "i"}},
            {"search_keywords": {"$regex": f"^{phrase}$", "$options": "i"}},
        ]
        query_filter = dict(filters)
        query_filter["$or"] = field_clauses

        return list(
            self.collection.find(
                query_filter,
                self._projection(),
            ).limit(limit)
        )

    def vocabulary(self, limit: int = 60000) -> list[str]:
        """Build a bounded vocabulary from existing compact search fields.

        This is lazy and cached by SearchService. It deliberately avoids
        scanning the large ai_search_text field.
        """
        values: set[str] = set()
        projection = {"_id": 0, "title": 1, "search_aliases": 1, "search_keywords": 1}

        cursor = self.collection.find({}, projection, batch_size=5000)
        for doc in cursor:
            raw_values: list[Any] = [doc.get("title")]
            raw_values.extend(doc.get("search_aliases") or [])
            raw_values.extend(doc.get("search_keywords") or [])
            for value in raw_values:
                if not isinstance(value, str):
                    continue
                for token in re.findall(r"[a-z0-9][a-z0-9&'-]*", value.casefold()):
                    if len(token) >= 3:
                        values.add(token)
                        if len(values) >= limit:
                            return sorted(values)
        return sorted(values)

    def suggest_vocabulary(self, token: str, limit: int = 200) -> list[str]:
        """Return likely vocabulary terms for one misspelled token.

        This avoids building a 300k-document vocabulary on every process
        start. Candidate strings come only from compact title/alias/keyword
        fields, never the large ai_search_text payload.
        """
        normalized = normalize(token)
        if len(normalized) < 3:
            return []
        prefix = re.escape(normalized[:2])
        suffix = re.escape(normalized[-1])
        pattern = f"^{prefix}.*{suffix}$"
        projection = {"_id": 0, "title": 1, "search_aliases": 1, "search_keywords": 1}
        query = {
            "$or": [
                {"title": {"$regex": pattern, "$options": "i"}},
                {"search_aliases": {"$regex": pattern, "$options": "i"}},
                {"search_keywords": {"$regex": pattern, "$options": "i"}},
            ]
        }
        terms: set[str] = set()
        try:
            for doc in self.collection.find(query, projection).limit(limit):
                values = [doc.get("title"), *(doc.get("search_aliases") or []), *(doc.get("search_keywords") or [])]
                for value in values:
                    if isinstance(value, str):
                        for candidate in re.findall(r"[a-z0-9][a-z0-9&'-]*", value.casefold()):
                            if len(candidate) >= 3:
                                terms.add(candidate)
        except Exception:
            return []
        return sorted(terms)

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
        if not tokens(query):
            return []

        candidate_limit = 100
        projection = self._projection()
        docs: list[dict[str, Any]] = []

        # 1. Exact title/alias/keyword phrase matches.
        docs.extend(self._exact_candidates(query, filters, candidate_limit))
        seen = {str(doc.get("_id")) for doc in docs}

        # 2. MongoDB's existing ai_search_text text index.
        try:
            text_docs = list(
                self.collection.find(
                    {"$text": {"$search": query.strip()}, **filters},
                    projection,
                )
                .sort([("text_score", {"$meta": "textScore"})])
                .limit(candidate_limit)
            )
            docs.extend(doc for doc in text_docs if str(doc.get("_id")) not in seen)
            seen.update(str(doc.get("_id")) for doc in text_docs)
        except Exception:
            # Exact matching still works if MongoDB rejects an unusual text
            # expression. The service will not fail solely because of it.
            pass

        # 3. For a multi-token phrase, retrieve documents containing all query
        # tokens in the existing searchable aliases/keywords/title fields.
        q_tokens = [t for t in tokens(query) if len(t) >= 3]
        if q_tokens and len(docs) < candidate_limit:
            token_clauses = []
            for token in q_tokens[:8]:
                pattern = re.escape(token)
                token_clauses.append({
                    "$or": [
                        {"title": {"$regex": pattern, "$options": "i"}},
                        {"search_aliases": {"$regex": pattern, "$options": "i"}},
                        {"search_keywords": {"$regex": pattern, "$options": "i"}},
                    ]
                })
            fallback_filter = {"$and": token_clauses}
            if filters:
                fallback_filter = {"$and": [filters, fallback_filter]}
            try:
                fallback_docs = list(
                    self.collection.find(fallback_filter, projection).limit(candidate_limit)
                )
                docs.extend(doc for doc in fallback_docs if str(doc.get("_id")) not in seen)
            except Exception:
                pass

        return docs[:candidate_limit]
