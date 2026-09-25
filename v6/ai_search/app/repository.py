from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pymongo.collection import Collection

from .config import settings
from .normalization import normalize, tokens
from .observability import timed
from . import schema_map


def _singular(token: str) -> str:
    token = token.casefold()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "sses", "xes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _token_pattern(token: str) -> str:
    """Regex for a query token that also matches its singular form.

    The $text index stems words, but the regex fallback does not: without this
    "chefs" would never match a document titled "Executive Chef".
    """
    singular = _singular(token)
    if singular == token:
        return re.escape(token)
    return f"(?:{re.escape(token)}|{re.escape(singular)})"


# Filters without a structured field in the migrated documents: handled as
# evidence (accommodation, experience) or ranking signals by the service.
SOFT_FILTER_KEYS = {"salary_min", "salary_currency", "experience", "accommodation"}
# Schema facts that define WHAT a record is (supplier companies, supplier
# industries, supplier categories): a record without them never matches.
HARD_CONCEPT_KEYS = {"is_supplier", "industry_context", "supplier_category"}
# Date fields per module (jobs/articles store ISO strings in metadata).
DATE_PATHS = ("created_at", "dates.created_at", "metadata.created_at", "start_datetime", "award_date")


def _parse_date(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and len(value) >= 10:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _object_id(value: str) -> Any:
    try:
        from bson import ObjectId

        if len(value) == 24 and ObjectId.is_valid(value):
            return ObjectId(value)
    except Exception:  # pragma: no cover - bson ships with pymongo
        return None
    return None

class SearchDocumentsRepository:
    """
    MongoDB retrieval layer over the existing search_documents collection.

    Phase 1:
    - MongoDB only
    - Uses the existing ai_search_text MongoDB text index
    - Does not create/drop production indexes
    - Supports exact phrase matching
    - Supports aliases and keywords
    - Supports entity filtering
    - Supports city/country filtering
    - Supports status/live filtering
    - Supports fallback token matching
    """

    ENTITY_TYPES = {
        "job",
        "professional",
        "company",
        "product",
        "article",
        "event",
        "award",
        "faq",
    }

    def __init__(self, collection: Collection):
        self.collection = collection

    def ensure_indexes(self) -> None:
        """
        Keep the existing MongoDB indexes intact.

        The production collection already has a MongoDB text index on
        ai_search_text. MongoDB permits only one text index per collection,
        so Phase 1 uses the existing index rather than creating another one.
        """
        required = {
            "_id_",
            "idx_ai_search_text",
        }

        try:
            names = {
                index["name"]
                for index in self.collection.list_indexes()
            }

            missing = required - names

            if missing:
                raise RuntimeError(
                    "Missing required MongoDB search index(es): "
                    + ", ".join(sorted(missing))
                )

        except RuntimeError:
            raise

        except Exception:
            
            raise

    @staticmethod
    def _regex(value: str) -> dict[str, Any]:
        """
        Build a case-insensitive escaped regex.
        """
        return {
            "$regex": re.escape(normalize(value)),
            "$options": "i",
        }

    @staticmethod
    def _country_terms(value: str) -> list[str]:
        """
        Convert common country aliases into searchable variants.
        """
        normalized = normalize(value)

        aliases = {
            "uae": [
                "United Arab Emirates",
                "AE",
                "UAE",
            ],
            "u.a.e": [
                "United Arab Emirates",
                "AE",
                "UAE",
            ],
            "usa": [
                "United States",
                "US",
                "USA",
            ],
            "us": [
                "United States",
                "US",
                "USA",
            ],
            "uk": [
                "United Kingdom",
                "GB",
                "UK",
            ],
            "ua": [
                "Ukraine",
                "UA",
            ],
        }

        return aliases.get(normalized, [value])

    @staticmethod
    def _location_fields(entity: str | None, kind: str) -> list[str]:
        """Authoritative location paths per entity (schema_map.py).

        Each module stores its own location: a job's location never comes
        from its company, an award's city is the `location` string and its
        country is top-level, articles only carry countries.
        """
        return list(schema_map.location_paths(entity, kind))

    @staticmethod
    def _nested_values(doc: dict[str, Any], path: str) -> list[Any]:
        current: list[Any] = [doc]
        for part in path.split("."):
            nxt: list[Any] = []
            for item in current:
                if isinstance(item, dict):
                    value = item.get(part)
                    if isinstance(value, list):
                        nxt.extend(value)
                    elif value is not None:
                        nxt.append(value)
            current = nxt
        return current

    @staticmethod
    def _filter(
        entity: str | None,
        city: str | None,
        country: str | None,
        status: str | None,
        is_live: bool | None,
        structured: dict[str, Any] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Build MongoDB filters.

        Filters are intentionally based on the actual MongoDB schema used
        by search_documents.
        """
        clauses: list[dict[str, Any]] = []

        if entity:
            clauses.append({
                "entity_type": entity,
            })

        if is_live is not None:
            # Awards store liveness as is_active; FAQs only as status=active.
            live_clauses: list[dict[str, Any]] = [
                {"is_live": is_live},
                {"metadata.is_live": is_live},
                {"$and": [{"entity_type": "award"}, {"is_active": is_live}]},
            ]
            if is_live:
                live_clauses.append({"$and": [{"entity_type": "faq"}, {"status": "active"}]})
            clauses.append({"$or": live_clauses})

            if is_live is True:
                clauses.append({
                    "$or": [
                        {
                            "expires_at": {
                                "$exists": False,
                            }
                        },
                        {
                            "expires_at": None,
                        },
                        {
                            "expires_at": {
                                "$gte": datetime.now(timezone.utc),
                            }
                        },
                    ]
                })

        if status:
            pattern = re.escape(
                normalize(status)
            )

            clauses.append({
                "$or": [
                    {
                        "status": {
                            "$regex": f"^{pattern}$",
                            "$options": "i",
                        }
                    },
                    {
                        "metadata.status": {
                            "$regex": f"^{pattern}$",
                            "$options": "i",
                        }
                    },
                    {
                        "metadata.job_status": {
                            "$regex": f"^{pattern}$",
                            "$options": "i",
                        }
                    },
                ]
            })

        if city:
            pattern = re.escape(normalize(city))
            clauses.append({
                "$or": [
                    {path: {"$regex": pattern, "$options": "i"}}
                    for path in SearchDocumentsRepository._location_fields(entity, "city")
                ]
            })

        if country:
            country_clauses: list[dict[str, Any]] = []
            for term in SearchDocumentsRepository._country_terms(country):
                pattern = re.escape(normalize(term))
                country_clauses.extend(
                    {path: {"$regex": pattern, "$options": "i"}}
                    for path in SearchDocumentsRepository._location_fields(entity, "country")
                )
            clauses.append({"$or": country_clauses})

        structured = structured or {}

        # Structured filters use the exact paths written by the migration
        # scripts. Boolean schema facts (company.is_supplier) are equality
        # checks; names are case-insensitive substring matches.
        for key, value in structured.items():
            if value is None or key in SOFT_FILTER_KEYS:
                continue
            clause = SearchDocumentsRepository._structured_clause(entity, key, value)
            if clause:
                clauses.append(clause)

        if date_from or date_to:
            date_clauses = []
            for path in DATE_PATHS:
                condition: dict[str, Any] = {}
                text_condition: dict[str, Any] = {}
                if date_from:
                    condition["$gte"] = date_from
                    text_condition["$gte"] = date_from.date().isoformat()
                if date_to:
                    condition["$lt"] = date_to
                    text_condition["$lt"] = date_to.date().isoformat()
                date_clauses.append({path: condition})
                date_clauses.append({path: text_condition})
            clauses.append({"$or": date_clauses})

        if not clauses:
            return {}

        return {
            "$and": clauses,
        }

    @staticmethod
    def _structured_clause(entity: str | None, key: str, value: Any) -> dict[str, Any] | None:
        paths = schema_map.filter_paths(entity, key)
        if not paths:
            return None
        if isinstance(value, bool):
            return {"$or": [{path: value} for path in paths]}
        pattern = re.escape(normalize(str(value)))
        if key == "industry_context":
            pattern = f"^{pattern}$"
        return {"$or": [{path: {"$regex": pattern, "$options": "i"}} for path in paths]}

    @staticmethod
    def _projection() -> dict[str, Any]:
        """Exclude only heavy/irrelevant fields.

        The migrated documents carry many module-specific top-level fields
        (summary, sub_title, content, seller, categories, pricing, dates,
        flags, country, year, website, ...). An inclusion list silently
        dropped several of them, so exclude what search never needs instead.

        IMPORTANT: textScore ($meta) is added only to $text queries.
        """
        return {
            "embedding": 0,
            "migration": 0,
            "filter_questions": 0,
            "testimonials": 0,
            "education": 0,
            "certifications": 0,
            "package": 0,
            "posted_by": 0,
            "walk_in": 0,
            "coordinates": 0,
            "engagement": 0,
        }

    def _exact_candidates(
        self,
        query: str,
        filters: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Retrieve high-confidence exact title/alias/keyword matches.

        This query intentionally uses regex only.

        Therefore textScore MUST NOT be requested here.
        """

        phrase = re.escape(
            normalize(query)
        )

        if not phrase:
            return []

        field_clauses = [
            {
                "title": {
                    "$regex": f"^{phrase}$",
                    "$options": "i",
                }
            },
            {
                "search_aliases": {
                    "$regex": f"^{phrase}$",
                    "$options": "i",
                }
            },
            {
                "search_keywords": {
                    "$regex": f"^{phrase}$",
                    "$options": "i",
                }
            },
            {
                "question": {
                    "$regex": f"^{phrase}\\??$",
                    "$options": "i",
                }
            },
        ]

        query_filter: dict[str, Any]

        if filters:
            query_filter = {
                "$and": [
                    filters,
                    {
                        "$or": field_clauses,
                    },
                ]
            }
        else:
            query_filter = {
                "$or": field_clauses,
            }

        with timed("mongo_ms"):
            return list(
                self.collection.find(
                    query_filter,
                    self._projection(),
                )
                .limit(limit)
                .max_time_ms(settings.mongodb_max_time_ms)
            )

    def vocabulary(
        self,
        limit: int = 60000,
    ) -> list[str]:
        """
        Build a bounded vocabulary from compact searchable fields.

        This deliberately avoids scanning the large ai_search_text field.
        """

        values: set[str] = set()

        projection = {
            "_id": 0,
            "title": 1,
            "search_aliases": 1,
            "search_keywords": 1,
        }

        cursor = self.collection.find(
            {},
            projection,
            batch_size=5000,
        )

        for doc in cursor:
            raw_values: list[Any] = [
                doc.get("title")
            ]

            raw_values.extend(
                doc.get("search_aliases") or []
            )

            raw_values.extend(
                doc.get("search_keywords") or []
            )

            for value in raw_values:
                if not isinstance(value, str):
                    continue

                for token in re.findall(
                    r"[a-z0-9][a-z0-9&'-]*",
                    value.casefold(),
                ):
                    if len(token) < 3:
                        continue

                    values.add(token)

                    if len(values) >= limit:
                        return sorted(values)

        return sorted(values)

    def suggest_vocabulary(
        self,
        token: str,
        limit: int = 200,
    ) -> list[str]:
        """
        Return likely vocabulary terms for one misspelled token.

        Candidates are retrieved only from compact title/alias/keyword
        fields, avoiding a full ai_search_text scan.
        """

        normalized = normalize(token)

        if len(normalized) < 3:
            return []

        prefix = re.escape(
            normalized[:2]
        )

        suffix = re.escape(
            normalized[-1]
        )

        pattern = (
            f"^{prefix}.*{suffix}$"
        )

        projection = {
            "_id": 0,
            "title": 1,
            "search_aliases": 1,
            "search_keywords": 1,
        }

        query = {
            "$or": [
                {
                    "title": {
                        "$regex": pattern,
                        "$options": "i",
                    }
                },
                {
                    "search_aliases": {
                        "$regex": pattern,
                        "$options": "i",
                    }
                },
                {
                    "search_keywords": {
                        "$regex": pattern,
                        "$options": "i",
                    }
                },
            ]
        }

        terms: set[str] = set()

        try:
            cursor = self.collection.find(
                query,
                projection,
            ).limit(limit)

            for doc in cursor:
                values = [
                    doc.get("title"),
                    *(doc.get("search_aliases") or []),
                    *(doc.get("search_keywords") or []),
                ]

                for value in values:
                    if not isinstance(value, str):
                        continue

                    for candidate in re.findall(
                        r"[a-z0-9][a-z0-9&'-]*",
                        value.casefold(),
                    ):
                        if len(candidate) >= 3:
                            terms.add(candidate)

        except Exception:
            return []

        return sorted(terms)

    @staticmethod
    def _document_matches_filters(
        doc: dict[str, Any],
        *,
        entity: str | None,
        city: str | None,
        country: str | None,
        status: str | None,
        is_live: bool | None,
        structured: dict[str, Any] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> bool:
        """Defensive hard-filter check after MongoDB candidate retrieval."""
        if entity and str(doc.get("entity_type") or "").casefold() != entity.casefold():
            return False

        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}

        if is_live is not None:
            live_values = [doc.get("is_live"), metadata.get("is_live")]
            if doc.get("entity_type") == "award":
                live_values.append(doc.get("is_active"))
            if doc.get("entity_type") == "faq" and "is_live" not in doc:
                live_values.append(doc.get("status") == "active")
            if not any(value is is_live for value in live_values):
                return False
            if is_live is True:
                expiry = doc.get("expires_at")
                if isinstance(expiry, datetime) and expiry < datetime.now(timezone.utc):
                    return False

        if status:
            wanted = normalize(status)
            statuses = [doc.get("status"), metadata.get("status"), metadata.get("job_status")]
            if not any(isinstance(value, str) and normalize(value) == wanted for value in statuses):
                return False

        doc_entity = entity or str(doc.get("entity_type") or "") or None

        if city:
            wanted = normalize(city)
            values: list[Any] = []
            for path in SearchDocumentsRepository._location_fields(doc_entity, "city"):
                values.extend(SearchDocumentsRepository._nested_values(doc, path))
            if not any(isinstance(value, str) and wanted in normalize(value) for value in values):
                return False

        if country:
            wanted_terms = [normalize(term) for term in SearchDocumentsRepository._country_terms(country)]
            values = []
            for path in SearchDocumentsRepository._location_fields(doc_entity, "country"):
                values.extend(SearchDocumentsRepository._nested_values(doc, path))
            normalized_values = [normalize(value) for value in values if isinstance(value, str)]
            # Short codes ("in", "ae") must match exactly; substring matching
            # would let "IN" (India) satisfy "Argentina".
            if not any(
                term == value or (len(value) > 3 and len(term) > 3 and (term in value or value in term))
                for term in wanted_terms
                for value in normalized_values
            ):
                return False

        structured = structured or {}

        def nested_values(obj: Any, path: str) -> list[Any]:
            current = [obj]
            for part in path.split("."):
                nxt = []
                for item in current:
                    if isinstance(item, dict):
                        value = item.get(part)
                        if isinstance(value, list):
                            nxt.extend(value)
                        else:
                            nxt.append(value)
                current = nxt
            return [v for v in current if v is not None]

        for key, wanted in structured.items():
            if wanted is None or key in SOFT_FILTER_KEYS:
                continue
            paths = schema_map.filter_paths(doc_entity, key)
            if not paths:
                continue
            present = []
            for path in paths:
                present.extend(nested_values(doc, path))
            if not present:
                if key in HARD_CONCEPT_KEYS:
                    return False
                continue
            if isinstance(wanted, bool):
                if wanted not in [bool(v) for v in present]:
                    return False
            elif key == "industry_context":
                if normalize(str(wanted)) not in {normalize(str(v)) for v in present}:
                    return False
            elif not any(normalize(str(wanted)) in normalize(str(v)) for v in present):
                return False

        def numeric_values(paths: tuple[str, ...]) -> list[float]:
            out = []
            for path in paths:
                for value in nested_values(doc, path):
                    if isinstance(value, (int, float)):
                        out.append(float(value))
                    elif isinstance(value, str):
                        m = re.search(r"\d+(?:\.\d+)?", value)
                        if m:
                            out.append(float(m.group(0)))
            return out

        if date_from or date_to:
            date_values = []
            for path in DATE_PATHS:
                date_values.extend(nested_values(doc, path))
            date_values = [d for d in (_parse_date(v) for v in date_values) if d]
            if date_values:
                chosen = max(date_values)
                if date_from and chosen < date_from:
                    return False
                if date_to and chosen >= date_to:
                    return False

        return True

    def fetch_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Fetch documents by _id, preserving the requested order.

        Accepts string ids; 24-hex ids are also matched as ObjectId because
        production _id values may be either.
        """
        ids = [str(value) for value in ids if str(value).strip()]
        if not ids:
            return []
        lookup: list[Any] = list(ids)
        for value in ids:
            oid = _object_id(value)
            if oid is not None:
                lookup.append(oid)
        with timed("mongo_ms"):
            docs = list(
                self.collection.find(
                    {"_id": {"$in": lookup}},
                    self._projection(),
                ).max_time_ms(settings.mongodb_max_time_ms)
            )
        by_id = {str(doc.get("_id")): doc for doc in docs}
        return [by_id[value] for value in ids if value in by_id]

    def browse(
        self,
        *,
        entity: str | None,
        city: str | None,
        country: str | None,
        status: str | None,
        is_live: bool | None,
        limit: int = 100,
        structured: dict[str, Any] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Filter-only retrieval ("events in Dubai"), newest first.

        Requires an entity or a location so it can never become an unbounded
        collection scan of every document type.
        """
        if not (entity or city or country):
            return []
        filters = self._filter(
            entity=entity,
            city=city,
            country=country,
            status=status,
            is_live=is_live,
            structured=structured,
            date_from=date_from,
            date_to=date_to,
        )
        try:
            with timed("mongo_ms"):
                docs = list(
                    self.collection.find(filters, self._projection())
                    .sort([(path, -1) for path in (schema_map.schema(entity).created if schema_map.schema(entity) else ("created_at",))] + [("_id", 1)])
                    .limit(min(limit, 100))
                    .max_time_ms(settings.mongodb_max_time_ms)
                )
        except Exception:
            return []
        return [
            doc
            for doc in docs
            if self._document_matches_filters(
                doc,
                entity=entity,
                city=city,
                country=country,
                status=status,
                is_live=is_live,
                structured=structured,
                date_from=date_from,
                date_to=date_to,
            )
        ]

    def facet_values(
        self,
        *,
        entity: str,
        path: str,
        extra: dict[str, Any] | None = None,
        city: str | None = None,
        country: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Distinct values (with record counts) of a concept stored inside
        documents, e.g. company.supplier_categories.name. Database-backed:
        only values that exist on real records are returned."""
        extra = extra or {}
        unwind = path.rsplit(".", 1)[0]
        base = self._filter(entity=entity, city=city, country=country, status=None, is_live=None)
        doc_match = [base] if base else []
        doc_match += [{key: value} for key, value in extra.items()]
        element_match = {key: value for key, value in extra.items() if key.startswith(unwind + ".")}
        pipeline: list[dict[str, Any]] = [
            {"$match": {"$and": doc_match} if doc_match else {}},
            {"$project": {"_id": 1, unwind: 1}},
            {"$unwind": f"${unwind}"},
        ]
        if element_match:
            pipeline.append({"$match": element_match})
        pipeline += [
            {"$group": {"_id": f"${path}", "count": {"$sum": 1}}},
            {"$match": {"_id": {"$nin": [None, ""]}}},
            {"$sort": {"count": -1, "_id": 1}},
            {"$limit": limit},
        ]
        try:
            with timed("mongo_ms"):
                rows = list(self.collection.aggregate(pipeline, maxTimeMS=settings.mongodb_max_time_ms))
        except TypeError:  # test doubles without maxTimeMS support
            with timed("mongo_ms"):
                rows = list(self.collection.aggregate(pipeline))
        return [{"name": str(r["_id"]), "count": int(r["count"])} for r in rows if isinstance(r.get("_id"), str)]

    def find_by_titles(self, entity: str, titles: list[str], limit: int = 5) -> list[dict[str, Any]]:
        """Exact (case-insensitive) title lookup, e.g. company profiles by name."""
        clauses = [
            {"title": {"$regex": f"^{re.escape(title.strip())}$", "$options": "i"}}
            for title in titles
            if isinstance(title, str) and title.strip()
        ]
        if not clauses:
            return []
        with timed("mongo_ms"):
            return list(
                self.collection.find(
                    {"$and": [{"entity_type": entity}, {"$or": clauses}]},
                    self._projection(),
                )
                .limit(limit)
                .max_time_ms(settings.mongodb_max_time_ms)
            )

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
        structured: dict[str, Any] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search MongoDB search_documents.

        Retrieval order:
        1. Exact title/alias/keyword match
        2. Existing MongoDB ai_search_text text index
        3. Token fallback against title/aliases/keywords

        Ranking is handled separately by ranking.py.
        """

        filters = self._filter(
            entity=entity,
            city=city,
            country=country,
            status=status,
            is_live=is_live,
            structured=structured,
            date_from=date_from,
            date_to=date_to,
        )

        if not tokens(query):
            return []

        candidate_limit = 100

        docs: list[dict[str, Any]] = []

        docs.extend(
            self._exact_candidates(
                query,
                filters,
                candidate_limit,
            )
        )

        seen = {
            str(doc.get("_id"))
            for doc in docs
        }
        text_failed = False

        try:
            """
            IMPORTANT:

            textScore is added ONLY here because this query contains
            the MongoDB $text operator.

            The common projection intentionally does not contain
            $meta textScore.
            """

            text_projection = self._projection()

            text_projection["text_score"] = {
                "$meta": "textScore"
            }

            text_filter: dict[str, Any] = {
                "$text": {
                    "$search": query.strip(),
                }
            }

            if filters:
                text_filter = {
                    "$and": [
                        text_filter,
                        filters,
                    ]
                }

            with timed("mongo_ms"):
                text_docs = list(
                    self.collection.find(
                        text_filter,
                        text_projection,
                    )
                    .sort([("text_score", {"$meta": "textScore"})])
                    .limit(candidate_limit)
                    .max_time_ms(settings.mongodb_max_time_ms)
                )

            for doc in text_docs:
                doc_id = str(
                    doc.get("_id")
                )

                if doc_id not in seen:
                    docs.append(doc)
                    seen.add(doc_id)

        except Exception:
            """
            Do not make the whole search unavailable because MongoDB
            rejects a particular text expression.

            Exact and fallback retrieval can still return results.
            """
            text_failed = True

        q_tokens = [
            token
            for token in tokens(query)
            if len(token) >= 3
        ]

        if q_tokens and len(docs) < candidate_limit:

            token_clauses: list[dict[str, Any]] = []

            for token in q_tokens[:8]:

                pattern = _token_pattern(token)

                # Fields per module from the migration: professionals have no
                # keywords/aliases (role, resume title, skills instead); FAQs
                # have question/answer instead of a title.
                token_clauses.append({
                    "$or": [
                        {field: {"$regex": pattern, "$options": "i"}}
                        for field in schema_map.search_fields(entity)
                    ]
                })

            fallback_filter: dict[str, Any] = {
                "$and": token_clauses,
            }

            if filters:
                fallback_filter = {
                    "$and": [
                        filters,
                        fallback_filter,
                    ]
                }

            try:
                with timed("mongo_ms"):
                    fallback_docs = list(
                        self.collection.find(
                            fallback_filter,
                            self._projection(),
                        )
                        .limit(candidate_limit)
                        .max_time_ms(settings.mongodb_max_time_ms)
                    )

                for doc in fallback_docs:
                    doc_id = str(
                        doc.get("_id")
                    )

                    if doc_id not in seen:
                        docs.append(doc)
                        seen.add(doc_id)

            except Exception:
                pass

            # When the $text index path failed, the AND fallback is too strict
            # for natural-language queries. Emulate $text's OR semantics; the
            # ranker orders candidates by token coverage. Only used on $text
            # failure so healthy production traffic never pays for this scan.
            if text_failed and len(docs) < 5:
                # $text indexes ai_search_text, so the emulation must too.
                any_filter: dict[str, Any] = {"$or": [
                    {field: {"$regex": _token_pattern(token), "$options": "i"}}
                    for token in q_tokens[:8]
                    for field in (*schema_map.search_fields(entity), "ai_search_text")
                ]}
                if filters:
                    any_filter = {"$and": [filters, any_filter]}
                try:
                    with timed("mongo_ms"):
                        any_docs = list(
                            self.collection.find(any_filter, self._projection())
                            .limit(candidate_limit)
                            .max_time_ms(settings.mongodb_max_time_ms)
                        )
                    for doc in any_docs:
                        doc_id = str(doc.get("_id"))
                        if doc_id not in seen:
                            docs.append(doc)
                            seen.add(doc_id)
                except Exception:
                    pass

        # Defensive hard-filter after retrieval.
        docs = [
            doc
            for doc in docs
            if self._document_matches_filters(
                doc,
                entity=entity,
                city=city,
                country=country,
                status=status,
                is_live=is_live,
                structured=structured,
                date_from=date_from,
                date_to=date_to,
            )
        ]

        return docs[:candidate_limit]
