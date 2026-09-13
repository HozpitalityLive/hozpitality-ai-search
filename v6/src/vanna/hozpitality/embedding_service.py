"""Optional local embedding service for Hozpitality semantic search."""
from __future__ import annotations

import os
from typing import Optional


class EmbeddingService:
    def __init__(self):
        self.enabled = os.getenv("SEARCH_VECTOR_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        self.model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None
        self.error: Optional[str] = None

    def _load(self):
        if not self.enabled or self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception as exc:
            self.error = str(exc)
            self.enabled = False
            return None

    def encode(self, text: str) -> Optional[list[float]]:
        model = self._load()
        if model is None:
            return None
        vector = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return [float(x) for x in vector]
