from __future__ import annotations

from .normalization import tokens
from .ranking import score_document
from .repository import SearchDocumentsRepository
from .typo import correct_tokens


class SearchService:
    def __init__(self, repository: SearchDocumentsRepository, fuzzy_threshold: int = 82):
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
        original = query.strip()
        q_tokens = tokens(original)
        corrected_tokens, changes = correct_tokens(
            q_tokens,
            self._get_vocabulary(),
            threshold=self.fuzzy_threshold,
        )
        corrected_query = " ".join(corrected_tokens)
        effective_query = corrected_query or original

        # First retrieve with the original phrase. If typo correction changed
        # anything, retrieve again with the corrected query and merge.
        docs = self.repository.search(
            original,
            entity=entity,
            city=city,
            country=country,
            status=status,
            is_live=is_live,
            limit=5,
        )
        if changes and corrected_query != original:
            corrected_docs = self.repository.search(
                corrected_query,
                entity=entity,
                city=city,
                country=country,
                status=status,
                is_live=is_live,
                limit=5,
            )
            seen = {str(d.get("_id")) for d in docs}
            docs.extend(d for d in corrected_docs if str(d.get("_id")) not in seen)

        ranked = []
        for doc in docs:
            text_score = float(doc.get("score") or 0.0)
            score, matched = score_document(doc, effective_query, text_score)
            ranked.append((score, doc, matched))

        ranked.sort(key=lambda item: (-item[0], str(item[1].get("_id"))))

        results = []
        for score, doc, matched in ranked[:5]:
            location = doc.get("location") or {}
            results.append(
                {
                    "entity_type": str(doc.get("entity_type") or ""),
                    "entity_id": str(doc.get("entity_id") or doc.get("_id")),
                    "title": str(doc.get("title") or "Untitled result"),
                    "description": doc.get("description"),
                    "location": {
                        "city": location.get("city"),
                        "country": location.get("country"),
                    },
                    "category": doc.get("category"),
                    "url": doc.get("url"),
                    "image": doc.get("image"),
                    "score": score,
                    "matched_by": matched,
                    "corrected_query": corrected_query if changes else None,
                }
            )

        return {
            "query": original,
            "corrected_query": corrected_query if changes else None,
            "total": len(results),
            "results": results,
        }
