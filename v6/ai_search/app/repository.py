from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pymongo.collection import Collection

from .normalization import normalize, tokens

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
        """Return authoritative location fields for an entity type.

        For jobs/professionals, the document's own location is authoritative.
        A company's or author's location must not make a job appear to be in
        that city. Other entities may expose a root location, while companies
        may also keep location under company.*.
        """
        entity = (entity or "").casefold()
        if entity in {"job", "professional", "event", "award", "faq", "article"}:
            if kind == "city":
                return ["location.city", "location.current_location", "location.prime_city"]
            return [
                "location.country.name", "location.country.ac_name",
                "location.country.code", "location.countries.name",
                "location.countries.code",
            ]
        if entity == "company":
            if kind == "city":
                return ["location.city", "location.current_location", "company.city"]
            return [
                "location.country.name", "location.country.ac_name",
                "location.country.code", "company.country.name",
                "company.country.ac_name", "company.country.code",
            ]
        # Products can be associated with a supplier/company, so allow both.
        if kind == "city":
            return ["location.city", "location.current_location", "location.prime_city", "company.city"]
        return [
            "location.country.name", "location.country.ac_name",
            "location.country.code", "location.countries.name",
            "location.countries.code", "company.country.name",
            "company.country.ac_name", "company.country.code",
        ]

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
            clauses.append({
                "$or": [
                    {
                        "is_live": is_live,
                    },
                    {
                        "metadata.is_live": is_live,
                    },
                ]
            })

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

        # Structured Phase 2 filters are expressed as ORs across the known
        # document shapes. Missing fields are intentionally not treated as
        # mismatches here; the defensive post-filter below applies only when
        # a field is actually present.
        field_aliases = {
            "level": [
                "professional.job_level.name", "job.level.name",
                "job.job_level.name", "job_level.name", "metadata.job_level",
            ],
            "department": [
                "professional.department.name", "job.department.name",
                "department.name", "metadata.department",
            ],
            "industry": [
                "professional.industries.name", "job.industry.name",
                "industry.name", "company.industry.name", "metadata.industry",
            ],
            "category": [
                "category.name", "category", "job.category.name",
                "product.category.name", "metadata.category",
            ],
            "employment_type": [
                "job.employment_type", "employment_type", "metadata.employment_type",
            ],
        }
        for key, value in structured.items():
            if value is None or key in {"salary_min", "salary_currency", "experience", "verified", "featured", "currently_working"}:
                continue
            paths = field_aliases.get(key, [])
            if not paths:
                continue
            pattern = re.escape(normalize(str(value)))
            clauses.append({
                "$or": [
                    {path: {"$regex": pattern, "$options": "i"}}
                    for path in paths
                ]
            })

        if date_from or date_to:
            date_clauses = []
            for path in ("created_at", "dates.start", "start_datetime", "award_date"):
                condition: dict[str, Any] = {}
                if date_from:
                    condition["$gte"] = date_from
                if date_to:
                    condition["$lt"] = date_to
                date_clauses.append({path: condition})
            clauses.append({"$or": date_clauses})

        if not clauses:
            return {}

        return {
            "$and": clauses,
        }

    @staticmethod
    def _projection() -> dict[str, Any]:
        """
        Common projection.

        IMPORTANT:
        Do NOT include:
            {"$meta": "textScore"}

        here.

        This projection is also used by regex queries. MongoDB only allows
        textScore metadata when the query itself contains a $text operator.
        """

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

        return list(
            self.collection.find(
                query_filter,
                self._projection(),
            ).limit(limit)
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
            if not any(value is is_live for value in live_values):
                return False
            if is_live is True:
                expiry = doc.get("expires_at")
                if isinstance(expiry, datetime) and expiry < datetime.now(timezone.utc):
                    return False

        if status:
            wanted = normalize(status)
            values = [doc.get("status"), metadata.get("status"), metadata.get("job_status")]
            if not any(isinstance(value, str) and normalize(value) == wanted for value in values):
                return False

        if city:
            wanted = normalize(city)
            values: list[Any] = []
            for path in SearchDocumentsRepository._location_fields(entity, "city"):
                values.extend(SearchDocumentsRepository._nested_values(doc, path))
            if not any(isinstance(value, str) and wanted in normalize(value) for value in values):
                return False

        if country:
            wanted_terms = [normalize(term) for term in SearchDocumentsRepository._country_terms(country)]
            values: list[Any] = []
            for path in SearchDocumentsRepository._location_fields(entity, "country"):
                values.extend(SearchDocumentsRepository._nested_values(doc, path))
            normalized_values = [normalize(value) for value in values if isinstance(value, str)]
            if not any(
                term == value or term in value or value in term
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

        aliases = {
            "level": ["professional.job_level.name", "job.level.name", "job.job_level.name", "job_level.name", "metadata.job_level"],
            "department": ["professional.department.name", "job.department.name", "department.name", "metadata.department"],
            "industry": ["professional.industries.name", "job.industry.name", "industry.name", "company.industry.name", "metadata.industry"],
            "category": ["category.name", "category", "job.category.name", "product.category.name", "metadata.category"],
            "employment_type": ["job.employment_type", "employment_type", "metadata.employment_type"],
        }
        for key, wanted in structured.items():
            if key in {"salary_min", "salary_currency", "experience", "verified", "featured", "currently_working"}:
                continue
            present = []
            for path in aliases.get(key, []):
                present.extend(nested_values(doc, path))
            if present and not any(normalize(str(wanted)) in normalize(str(v)) for v in present):
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

        exp = structured.get("experience")
        if exp is not None:
            vals = numeric_values((
                "professional.experience_years", "professional.years_experience",
                "professional.experience", "job.experience_years", "job.years_experience",
                "job.experience", "experience_years", "experience", "metadata.experience_years",
                "metadata.experience",
            ))
            if vals and max(vals) < float(exp):
                return False

        for key, path in (
            ("verified", "metadata.verified"),
            ("featured", "metadata.is_featured"),
            ("currently_working", "professional.currently_working"),
        ):
            if key in structured:
                vals = nested_values(doc, path)
                if vals and bool(structured[key]) not in [bool(v) for v in vals]:
                    return False

        if "salary_min" in structured:
            vals = numeric_values((
                "job.salary_min", "job.salary.min", "salary_min", "salary.min",
                "metadata.salary_min", "metadata.salary.min"
            ))
            if vals and max(vals) < float(structured["salary_min"]):
                return False

        if date_from or date_to:
            date_values = []
            for path in ("created_at", "dates.start", "start_datetime", "award_date"):
                date_values.extend(nested_values(doc, path))
            date_values = [v for v in date_values if isinstance(v, datetime)]
            if date_values:
                chosen = max(date_values)
                if date_from and chosen < date_from:
                    return False
                if date_to and chosen >= date_to:
                    return False

        return True

    def fetch_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        docs = list(self.collection.find(
            {"_id": {"$in": ids}},
            self._projection(),
        ))
        by_id = {str(doc.get("_id")): doc for doc in docs}
        return [by_id[value] for value in ids if value in by_id]

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

            text_docs = list(
                self.collection.find(
                    text_filter,
                    text_projection,
                )
                .sort(
                    [
                        (
                            "text_score",
                            {
                                "$meta": "textScore",
                            },
                        )
                    ]
                )
                .limit(candidate_limit)
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
            pass

        q_tokens = [
            token
            for token in tokens(query)
            if len(token) >= 3
        ]

        if q_tokens and len(docs) < candidate_limit:

            token_clauses: list[dict[str, Any]] = []

            for token in q_tokens[:8]:

                pattern = re.escape(token)

                token_clauses.append({
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
                fallback_docs = list(
                    self.collection.find(
                        fallback_filter,
                        self._projection(),
                    ).limit(candidate_limit)
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
