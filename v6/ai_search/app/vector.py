from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class SemanticVectorIndex:
    """
    Optional FAISS semantic retrieval layer.

    The lexical MongoDB path remains authoritative. When a local FAISS index
    and sentence-transformers model are configured, this component contributes
    semantic candidates to the same hybrid ranker. It never changes MongoDB data.
    """

    def __init__(self) -> None:
        self.index_path = Path(os.getenv("SEMANTIC_INDEX_PATH", "ai_search/data/search.faiss"))
        self.ids_path = Path(os.getenv("SEMANTIC_IDS_PATH", "ai_search/data/search_ids.json"))
        self.model_name = os.getenv("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.enabled = os.getenv("SEMANTIC_SEARCH_ENABLED", "false").casefold() == "true"
        self._index = None
        self._ids: list[str] | None = None
        self._model = None

    def _load(self) -> bool:
        if not self.enabled or self._index is not None:
            return self._index is not None
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            if not self.index_path.exists() or not self.ids_path.exists():
                return False
            self._index = faiss.read_index(str(self.index_path))
            self._ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
            self._model = SentenceTransformer(self.model_name)
            return True
        except Exception as exc:
            print(f"Semantic vector search disabled: {exc}")
            self.enabled = False
            return False

    def search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        if not self._load():
            return []
        vector = self._model.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(vector, limit)
        result = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            result.append((str(self._ids[idx]), float(score)))
        return result
