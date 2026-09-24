from __future__ import annotations

from .normalization import canonical_entity, tokens
from .ranking import score_document
from .repository import SearchDocumentsRepository
from .typo import correct_tokens


class SearchService:
    MAX_RESULTS = 5

    def __init__(self, repository: SearchDocumentsRepository, fuzzy_threshold: int = 82):
        self.repository = repository
        self.fuzzy_threshold = fuzzy_threshold
        self._vocabulary: list[str] | None = None

    def _get_vocabulary(self, query_tokens: list[str] | None = None) -> list[str]:
        # The vocabulary is populated lazily from compact fields. For the first
        # request, start with targeted candidates for the actual query tokens;
        # fall back to the bounded cache only when needed.
        if self._vocabulary is None:
            self._vocabulary = []
            for token in query_tokens or []:
                try:
                    self._vocabulary.extend(self.repository.suggest_vocabulary(token))
                except AttributeError:
                    pass
            self._vocabulary = sorted(set(self._vocabulary))
        return self._vocabulary

    def invalidate_vocabulary(self) -> None:
        self._vocabulary = None

    @staticmethod
    def _nested(doc: dict, *path: str):
        value = doc
        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    @classmethod
    def _description(cls, doc: dict) -> str | None:
        for value in (
            doc.get("description"),
            doc.get("subtitle"),
            doc.get("sub_title"),
            doc.get("question"),
            doc.get("answer"),
            cls._nested(doc, "job", "description"),
            cls._nested(doc, "content", "text"),
        ):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @classmethod
    def _location(cls, doc: dict) -> dict:
        location = doc.get("location") if isinstance(doc.get("location"), dict) else {}
        country = location.get("country") if isinstance(location.get("country"), dict) else {}
        return {
            "city": location.get("city") or location.get("current_location") or location.get("prime_city"),
            "country": country.get("name") or country.get("ac_name") or country.get("code"),
            "address": location.get("address"),
        }

    @classmethod
    def _category(cls, doc: dict) -> str | None:
        category = doc.get("category")
        if isinstance(category, dict):
            return category.get("name")
        if isinstance(category, str):
            return category
        return None

    @classmethod
    def _url(cls, doc: dict) -> str | None:
        for value in (
            doc.get("url"),
            cls._nested(doc, "links", "detail"),
            doc.get("slug"),
        ):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @classmethod
    def _image(cls, doc: dict) -> str | None:
        for value in (
            doc.get("profile_image"),
            doc.get("cover_image"),
            cls._nested(doc, "media", "image"),
            cls._nested(doc, "media", "main_image", "path"),
            cls._nested(doc, "media", "banner"),
            cls._nested(doc, "media", "avatar"),
            cls._nested(doc, "user", "profile_image"),
            cls._nested(doc, "company", "avatar"),
            cls._nested(doc, "author", "avatar"),
        ):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def search(
        self,
        *,
        query: str,
        entity: str | None = None,
        city: str | None = None,
        country: str | None = None,
        status: str | None = None,
        is_live: bool | None = None,
        limit: int = 5,
    ) -> dict:
        original = " ".join(query.strip().split())
        q_tokens = tokens(original)
        if not q_tokens:
            return {"query": original, "corrected_query": None, "total": 0, "results": []}

        corrected_tokens, changes = correct_tokens(
            q_tokens,
            self._get_vocabulary(q_tokens),
            threshold=self.fuzzy_threshold,
        )
        corrected_query = " ".join(corrected_tokens)
        effective_query = corrected_query or original

        canonical_entity_name = canonical_entity(entity)
        if canonical_entity_name in {
            "job", "professional", "company", "product",
            "article", "event", "award", "faq",
        }:
            entity = canonical_entity_name

        limit = min(max(int(limit), 1), self.MAX_RESULTS)
        docs = self.repository.search(
            effective_query,
            entity=entity,
            city=city,
            country=country,
            status=status,
            is_live=is_live,
            limit=limit,
        )

        # A correction must never hide an exact original match. Merge both
        # candidate sets and let deterministic ranking decide the order.
        if changes and corrected_query != original:
            original_docs = self.repository.search(
                original,
                entity=entity,
                city=city,
                country=country,
                status=status,
                is_live=is_live,
                limit=limit,
            )
            seen = {str(d.get("_id")) for d in docs}
            docs.extend(d for d in original_docs if str(d.get("_id")) not in seen)

        ranked = []
        for doc in docs:
            score, matched = score_document(
                doc,
                effective_query,
                float(doc.get("text_score") or 0.0),
            )
            ranked.append((score, doc, matched))

        ranked.sort(key=lambda item: (-item[0], str(item[1].get("_id"))))

        results = []
        for score, doc, matched in ranked[:limit]:
            results.append({
                "entity_type": str(doc.get("entity_type") or ""),
                "entity_id": str(doc.get("source", {}).get("object_id") or doc.get("_id") or ""),
                "title": str(doc.get("title") or doc.get("question") or doc.get("short_title") or "Untitled result"),
                "description": self._description(doc),
                "location": self._location(doc),
                "category": self._category(doc),
                "url": self._url(doc),
                "image": self._image(doc),
                "score": score,
                "matched_by": matched,
                "corrected_query": corrected_query if changes else None,
            })

        return {
            "query": original,
            "corrected_query": corrected_query if changes else None,
            "total": len(results),
            "results": results,
        }
