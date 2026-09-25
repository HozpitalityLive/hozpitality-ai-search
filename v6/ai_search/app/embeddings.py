"""Background embedding pipeline (never runs inside a user request).

MongoDB search_documents.ai_search_text
    -> incremental, batched sentence-transformer encoding (384-d MiniLM)
    -> ai_search_embeddings collection {_id, text_hash, model, dim, vector}
    -> FAISS HNSW index file published atomically
    -> SemanticVectorIndex hot-reloads it in the API workers

search_documents is only read. Unchanged documents (same text hash + model)
are never re-encoded, so a periodic run is cheap.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from pymongo import UpdateOne
from pymongo.collection import Collection

from .observability import log_event


class Encoder(Protocol):
    def __call__(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class EmbeddingStats:
    scanned: int = 0
    embedded: int = 0
    unchanged: int = 0
    skipped_empty: int = 0
    pruned: int = 0
    batches: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def text_hash(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()


def sentence_transformer_encoder(
    model_name: str, batch_size: int = 64, device: str | None = None
) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str]) -> list[list[float]]:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [list(map(float, v)) for v in vectors]

    return encode


def _bulk_upsert(target: Collection, operations: list[UpdateOne]) -> None:
    try:
        target.bulk_write(operations, ordered=False)
    except TypeError:
        # Test doubles (mongomock) lag behind the pymongo bulk API.
        for op in operations:
            doc = op._doc  # noqa: SLF001 - pymongo keeps the update document here
            target.update_one(op._filter, doc, upsert=True)  # noqa: SLF001


# Range comparisons on _id are type-bracketed in MongoDB: {"$gt": "job:9"}
# never matches an ObjectId. Keyset pagination therefore runs per BSON type,
# otherwise collections with mixed _id types silently lose documents.
_ID_TYPES = ("string", "objectId", "number", "binData", "date")


def _pages(source: Collection, scan_batch: int) -> Iterable[list[dict[str, Any]]]:
    for id_type in _ID_TYPES:
        last_id: Any = None
        while True:
            id_clause: dict[str, Any] = {"$type": id_type}
            if last_id is not None:
                id_clause["$gt"] = last_id
            query = {"ai_search_text": {"$type": "string"}, "_id": id_clause}
            page = list(
                source.find(query, {"_id": 1, "ai_search_text": 1})
                .sort("_id", 1)
                .limit(scan_batch)
            )
            if not page:
                break
            last_id = page[-1]["_id"]
            yield page
            if len(page) < scan_batch:
                break


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sync_embeddings(
    source: Collection,
    target: Collection,
    encoder: Encoder,
    *,
    model: str,
    dim: int = 384,
    batch_size: int = 256,
    scan_batch: int = 2000,
    max_chars: int = 4000,
    prune: bool = False,
    sleep_between_batches: float = 0.0,
) -> EmbeddingStats:
    """Embed new/changed documents in batches; returns statistics."""
    started = time.perf_counter()
    stats = EmbeddingStats()
    seen_ids: set[str] = set()

    for page in _pages(source, scan_batch):
        stats.scanned += len(page)

        candidates: dict[str, tuple[Any, str, str]] = {}
        for doc in page:
            text = " ".join(str(doc.get("ai_search_text") or "").split())[:max_chars]
            key = str(doc["_id"])
            seen_ids.add(key)
            if not text:
                stats.skipped_empty += 1
                continue
            candidates[key] = (doc["_id"], text, text_hash(text, model))

        existing = {
            str(e["_id"]): e.get("text_hash")
            for e in target.find(
                {"_id": {"$in": list(candidates)}}, {"_id": 1, "text_hash": 1}
            )
        }
        todo = [
            (key, value)
            for key, value in candidates.items()
            if existing.get(key) != value[2]
        ]
        stats.unchanged += len(candidates) - len(todo)

        for chunk in _chunks(todo, batch_size):
            texts = [value[1] for _, value in chunk]
            try:
                vectors = encoder(texts)
            except Exception as exc:  # keep going; a later run retries
                stats.errors.append(f"encode: {type(exc).__name__}")
                log_event(
                    "embedding_batch_failed",
                    level=logging.ERROR,
                    error=type(exc).__name__,
                )
                continue
            now = time.time()
            operations = []
            for (key, (source_id, _text, digest)), vector in zip(chunk, vectors):
                if len(vector) != dim:
                    stats.errors.append(f"dimension {len(vector)} != {dim} for {key}")
                    continue
                operations.append(
                    UpdateOne(
                        {"_id": key},
                        {
                            "$set": {
                                "text_hash": digest,
                                "model": model,
                                "dim": dim,
                                "vector": vector,
                                "updated_at": now,
                                "source_id": str(source_id),
                            }
                        },
                        upsert=True,
                    )
                )
            if operations:
                _bulk_upsert(target, operations)
                stats.embedded += len(operations)
            stats.batches += 1
            if sleep_between_batches:
                time.sleep(sleep_between_batches)

    if prune:
        stale = [
            e["_id"]
            for e in target.find({}, {"_id": 1})
            if str(e["_id"]) not in seen_ids
        ]
        for chunk in _chunks(stale, 1000):
            target.delete_many({"_id": {"$in": chunk}})
        stats.pruned = len(stale)

    stats.seconds = round(time.perf_counter() - started, 2)
    log_event(
        "embedding_sync",
        **{k: v for k, v in stats.__dict__.items() if k != "errors"},
        errors=len(stats.errors),
    )
    return stats


def load_vectors(
    target: Collection, *, model: str, dim: int = 384
) -> tuple[list[str], list[list[float]]]:
    ids: list[str] = []
    vectors: list[list[float]] = []
    for doc in target.find({"model": model, "dim": dim}, {"_id": 1, "vector": 1}).sort(
        "_id", 1
    ):
        ids.append(str(doc["_id"]))
        vectors.append(doc["vector"])
    return ids, vectors


def publish_faiss_index(
    ids: list[str],
    vectors: list[list[float]],
    *,
    index_path: Path,
    ids_path: Path,
    dim: int = 384,
    build: Callable[..., Any] | None = None,
) -> int:
    """Build an HNSW inner-product index and publish it atomically."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if build is None:
        import faiss
        import numpy as np

        matrix = (
            np.asarray(vectors, dtype="float32").reshape(-1, dim)
            if vectors
            else np.zeros((0, dim), "float32")
        )
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
        if len(matrix):
            index.add(matrix)
        tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
        faiss.write_index(index, str(tmp_index))
    else:
        tmp_index = build(vectors, index_path)
    tmp_ids = ids_path.with_suffix(ids_path.suffix + ".tmp")
    tmp_ids.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    # ids first, then the index: API workers reload on index mtime change.
    os.replace(tmp_ids, ids_path)
    os.replace(tmp_index, index_path)
    return len(ids)
