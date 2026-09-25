from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .observability import log_event, metrics


class SemanticVectorIndex:
    """
    Optional FAISS semantic retrieval layer.

    The lexical MongoDB path remains authoritative. When a local FAISS index
    and sentence-transformers model are configured, this component contributes
    semantic candidates to the same hybrid ranker. It never changes MongoDB data.

    Document embeddings are produced offline by the background embedding
    worker (ai_search/scripts/embedding_worker.py); a user request only embeds
    its own short query (cached). A newly published index file is picked up
    automatically without a restart.
    """

    QUERY_CACHE_MAX = 2048
    RELOAD_CHECK_SECONDS = 30.0

    def __init__(self) -> None:
        self.index_path = Path(
            os.getenv("SEMANTIC_INDEX_PATH", "ai_search/data/search.faiss")
        )
        self.ids_path = Path(
            os.getenv("SEMANTIC_IDS_PATH", "ai_search/data/search_ids.json")
        )
        self.model_name = os.getenv(
            "SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.enabled = (
            os.getenv("SEMANTIC_SEARCH_ENABLED", "false").casefold() == "true"
        )
        self._index: Any = None
        self._ids: list[str] | None = None
        self._model: Any = None
        self._loaded_mtime: float | None = None
        self._last_reload_check = 0.0
        self._lock = threading.Lock()
        self._query_cache: OrderedDict[str, object] = OrderedDict()

    def _load(self) -> bool:
        if not self.enabled:
            return False
        if self._index is not None:
            self._maybe_reload()
            return True
        with self._lock:
            if self._index is not None:
                return True
            try:
                import faiss
                from sentence_transformers import SentenceTransformer

                if not self.index_path.exists() or not self.ids_path.exists():
                    log_event(
                        "semantic_index_missing",
                        level=logging.WARNING,
                        path=str(self.index_path),
                    )
                    return False
                started = time.perf_counter()
                self._index = faiss.read_index(str(self.index_path))
                self._ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
                self._loaded_mtime = self.index_path.stat().st_mtime
                if self._model is None:
                    self._model = SentenceTransformer(self.model_name)
                log_event(
                    "semantic_index_loaded",
                    vectors=getattr(self._index, "ntotal", None),
                    load_ms=round((time.perf_counter() - started) * 1000, 1),
                )
                return True
            except Exception as exc:
                log_event(
                    "semantic_disabled",
                    level=logging.ERROR,
                    error=type(exc).__name__,
                    detail=str(exc)[:200],
                )
                self.enabled = False
                return False

    def _maybe_reload(self) -> None:
        now = time.monotonic()
        if now - self._last_reload_check < self.RELOAD_CHECK_SECONDS:
            return
        self._last_reload_check = now
        try:
            mtime = self.index_path.stat().st_mtime
        except OSError:
            return
        if self._loaded_mtime is not None and mtime > self._loaded_mtime:
            try:
                import faiss

                index = faiss.read_index(str(self.index_path))
                ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
                with self._lock:
                    self._index, self._ids, self._loaded_mtime = index, ids, mtime
                    self._query_cache.clear()
                log_event(
                    "semantic_index_reloaded", vectors=getattr(index, "ntotal", None)
                )
            except Exception as exc:
                log_event(
                    "semantic_reload_failed",
                    level=logging.WARNING,
                    error=type(exc).__name__,
                )

    def warmup_async(self) -> None:
        threading.Thread(target=self._load, name="semantic-warmup", daemon=True).start()

    def _encode(self, query: str):
        key = query.casefold().strip()
        cached = self._query_cache.get(key)
        if cached is not None:
            self._query_cache.move_to_end(key)
            metrics.incr("semantic_query_cache_hit")
            return cached
        vector = self._model.encode([query], normalize_embeddings=True)
        self._query_cache[key] = vector
        while len(self._query_cache) > self.QUERY_CACHE_MAX:
            self._query_cache.popitem(last=False)
        return vector

    def search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        if not query or not self._load():
            return []
        try:
            vector = self._encode(query)
            scores, indices = self._index.search(vector, limit)
        except Exception as exc:
            metrics.incr("semantic_errors")
            log_event(
                "semantic_search_failed",
                level=logging.WARNING,
                error=type(exc).__name__,
            )
            return []
        result = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            result.append((str(self._ids[idx]), float(score)))
        return result
