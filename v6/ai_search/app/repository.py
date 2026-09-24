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
    def _filter(
        entity: str | None,
        city: str | None,
        country: str | None,
        status: str | None,
        is_live: bool | None,
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
            pattern = re.escape(
                normalize(city)
            )

            clauses.append({
                "$or": [
                    {
                        "location.city": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                    {
                        "location.current_location": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                    {
                        "location.prime_city": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                    {
                        "company.city": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                    {
                        "author.city_town": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                ]
            })

        if country:
            country_clauses: list[dict[str, Any]] = []

            for term in SearchDocumentsRepository._country_terms(
                country
            ):
                pattern = re.escape(
                    normalize(term)
                )

                country_clauses.extend([
                    {
                        "location.country.name": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                    {
                        "location.country.ac_name": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                    {
                        "location.country.code": {
                            "$regex": f"^{pattern}$",
                            "$options": "i",
                        }
                    },
                    {
                        "location.countries.name": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                    {
                        "location.countries.code": {
                            "$regex": f"^{pattern}$",
                            "$options": "i",
                        }
                    },
                    {
                        "country.name": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                    {
                        "country.code": {
                            "$regex": f"^{pattern}$",
                            "$options": "i",
                        }
                    },
                    {
                        "company.country.name": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                ])

            clauses.append({
                "$or": country_clauses,
            })

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

        return docs[:candidate_limit]