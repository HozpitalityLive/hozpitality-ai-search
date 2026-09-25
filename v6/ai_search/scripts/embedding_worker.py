#!/usr/bin/env python3
"""Background embedding worker for semantic search.

Incrementally embeds search_documents.ai_search_text (384-d
all-MiniLM-L6-v2) in batches, stores vectors in the ai_search_embeddings
collection and publishes a FAISS index that the API hot-reloads.

    # one pass (cron / systemd timer)
    python -m ai_search.scripts.embedding_worker --once

    # long-running worker
    python -m ai_search.scripts.embedding_worker --interval 600

Requires: pip install -r ai_search/requirements-phase2-vector.txt
User search requests never generate document embeddings.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from ai_search.app.config import settings
from ai_search.app.db import get_collection
from ai_search.app.embeddings import (
    load_vectors,
    publish_faiss_index,
    sentence_transformer_encoder,
    sync_embeddings,
)
from ai_search.app.observability import configure_logging, log_event


def run_once(encoder, args) -> None:
    source = get_collection()
    target = get_collection(settings.embeddings_collection)
    target.create_index("model", name="idx_embeddings_model")
    stats = sync_embeddings(
        source,
        target,
        encoder,
        model=settings.semantic_model,
        dim=args.dim,
        batch_size=args.batch_size,
        prune=args.prune,
        sleep_between_batches=args.sleep,
    )
    index_path = Path(os.getenv("SEMANTIC_INDEX_PATH", "ai_search/data/search.faiss"))
    ids_path = Path(os.getenv("SEMANTIC_IDS_PATH", "ai_search/data/search_ids.json"))
    if stats.embedded or stats.pruned or args.force_publish or not index_path.exists():
        ids, vectors = load_vectors(target, model=settings.semantic_model, dim=args.dim)
        count = publish_faiss_index(
            ids, vectors, index_path=index_path, ids_path=ids_path, dim=args.dim
        )
        log_event("faiss_index_published", vectors=count, path=str(index_path))
    print(
        f"scanned={stats.scanned} embedded={stats.embedded} unchanged={stats.unchanged} "
        f"pruned={stats.pruned} errors={len(stats.errors)} seconds={stats.seconds}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--once", action="store_true", help="run one pass and exit")
    parser.add_argument(
        "--interval", type=int, default=600, help="seconds between passes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("SEMANTIC_BUILD_BATCH_SIZE", "256")),
    )
    parser.add_argument(
        "--dim", type=int, default=int(os.getenv("EMBEDDING_DIM", "384"))
    )
    parser.add_argument(
        "--device", default=os.getenv("EMBEDDING_DEVICE"), help="cpu / cuda"
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="pause between batches (throttle)"
    )
    parser.add_argument(
        "--prune", action="store_true", help="delete embeddings of removed documents"
    )
    parser.add_argument("--force-publish", action="store_true")
    args = parser.parse_args()

    configure_logging()
    encoder = sentence_transformer_encoder(
        settings.semantic_model,
        batch_size=min(args.batch_size, 128),
        device=args.device,
    )
    while True:
        try:
            run_once(encoder, args)
        except Exception as exc:
            # Keep the daemon alive; systemd restarts it if it crashes anyway.
            log_event(
                "embedding_worker_error",
                error=type(exc).__name__,
                detail=str(exc)[:200],
            )
            if args.once:
                raise
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
