#!/usr/bin/env python3
"""Build the optional local semantic index from MongoDB search_documents.

Run after installing ai_search/requirements-phase2-vector.txt:
    python -m ai_search.scripts.build_vector_index
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ai_search.app.db import get_collection


def main() -> None:
    model_name = os.getenv("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    out_index = Path(os.getenv("SEMANTIC_INDEX_PATH", "ai_search/data/search.faiss"))
    out_ids = Path(os.getenv("SEMANTIC_IDS_PATH", "ai_search/data/search_ids.json"))
    batch_size = int(os.getenv("SEMANTIC_BUILD_BATCH_SIZE", "256"))

    out_index.parent.mkdir(parents=True, exist_ok=True)
    collection = get_collection()
    model = SentenceTransformer(model_name)

    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 64

    ids: list[str] = []
    texts: list[str] = []

    def flush() -> None:
        nonlocal texts
        if not texts:
            return
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        index.add(vectors)
        texts = []

    cursor = collection.find(
        {"ai_search_text": {"$type": "string"}},
        {"_id": 1, "ai_search_text": 1},
        batch_size=batch_size,
    )

    for doc in cursor:
        ids.append(str(doc["_id"]))
        texts.append(doc["ai_search_text"])
        if len(texts) >= batch_size:
            flush()
        if len(ids) % 10000 == 0:
            print(f"processed={len(ids):,} vectors={index.ntotal:,}")

    flush()
    faiss.write_index(index, str(out_index))
    out_ids.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    print(f"completed vectors={index.ntotal:,}")
    print(f"index={out_index}")
    print(f"ids={out_ids}")


if __name__ == "__main__":
    main()
