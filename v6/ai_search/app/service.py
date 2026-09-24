from __future__ import annotations

from .normalization import canonical_entity, tokens
from .ranking import score_document
from .repository import SearchDocumentsRepository
from .typo import correct_tokens


class SearchService:
    def __init__(
        self,
        repository: SearchDocumentsRepository,
        fuzzy_threshold: int = 82,
    ):
        self.repository = repository
        self.fuzzy_threshold = fuzzy_threshold
        self._vocabulary: list[str] | None = None

    def _get_vocabulary(self) -> list[str]:
        if self._vocabulary is None:
            self._vocabulary = self.repository.vocabulary()
        return self._vocabulary

    def invalidate_vocabulary(self) -> None:
        self._vocabulary = None

    def search(
        self,
        *,
        query: str,
        entity: str | None = None,
        city: str | None = None,
        country: str | None = None,
        status: str | None = None,
        is_live: bool | None = True,
        limit: int = 5,
    ) -> dict:
        original = " ".join(query.strip().split())
        q_tokens = tokens(original)

        corrected_tokens, changes = correct_tokens(
            q_tokens,
            self._get_vocabulary(),
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

        docs = self.repository.search(
            effective_query,
            entity=entity,
            city=city,
            country=country,
            status=status,
            is_live=is_live,
            limit=limit,
        )

        # If correction changed the query, also try the original. This protects
        # against a vocabulary false-positive while keeping corrected retrieval.
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
            docs.extend(
                d for d in original_docs
                if str(d.get("_id")) not in seen
            )

        ranked = []
        for doc in docs:
            score, matched = score_document(
                doc,
                effective_query,
                float(doc.get("text_score") or 0.0),
            )
            ranked.append((score, doc, matched))

        ranked.sort(
            key=lambda item: (-item[0], str(item[1].get("_id")))
        )

        results = []
        for score, doc, matched in ranked[: min(max(limit, 1), 5)]:
            results.append(
                {
                    "entity_type": str(doc.get("entity_type") or ""),
                    "entity_id": str(
                        doc.get("entity_id") or doc.get("_id")
                    ),
                    "title": str(
                        doc.get("title")
                        or doc.get("entity_name")
                        or "Untitled result"
                    ),
                    "description": (
                        doc.get("description")
                        or doc.get("ai_summary")
                    ),
                    "location": {
                        "city": doc.get("city") or None,
                        "country": doc.get("country") or None,
                    },
                    "category": doc.get("category") or None,
                    "url": doc.get("url") or None,
                    "image": doc.get("image") or None,
                    "score": score,
                    "matched_by": matched,
                    "corrected_query": (
                        corrected_query if changes else None
                    ),
                }
            )

        return {
            "query": original,
            "corrected_query": corrected_query if changes else None,
            "total": len(results),
            "results": results,
        }
