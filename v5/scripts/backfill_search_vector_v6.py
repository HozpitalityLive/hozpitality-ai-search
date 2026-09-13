"""Backfill master_search_mastersearchindex.search_vector_v6 in small batches.

Safe for 500K+ rows: keyset pagination, bounded batches, optional sleep and
statement timeout. It does not rebuild ChromaDB or embeddings.
"""
from __future__ import annotations

import argparse
import os
import time

import psycopg2
from dotenv import load_dotenv

load_dotenv()

SQL = """
WITH batch AS (
    SELECT id
    FROM public.master_search_mastersearchindex
    WHERE id > %s
      AND search_vector_v6 IS NULL
    ORDER BY id
    LIMIT %s
)
UPDATE public.master_search_mastersearchindex AS si
SET search_vector_v6 =
      setweight(to_tsvector('simple', unaccent(coalesce(si.title, ''))), 'A')
    || setweight(to_tsvector('simple', unaccent(coalesce(si.category_text, ''))), 'A')
    || setweight(to_tsvector('simple', unaccent(coalesce(si.user_name, ''))), 'A')
    || setweight(to_tsvector('simple', unaccent(coalesce(si.location_text, ''))), 'B')
    || setweight(to_tsvector('simple', unaccent(coalesce(si.ai_keywords, ''))), 'B')
    || setweight(to_tsvector('simple', unaccent(coalesce(si.slug, ''))), 'C')
    || setweight(to_tsvector('simple', unaccent(coalesce(si.content, ''))), 'C')
FROM batch
WHERE si.id = batch.id
RETURNING si.id;
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="0 = all missing vectors")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--statement-timeout-ms", type=int, default=30000)
    parser.add_argument("--confirm", action="store_true", help="Required for production backfill")
    args = parser.parse_args()

    if not args.confirm:
        raise SystemExit("Refusing to backfill without --confirm. Start with a small --limit first.")

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DATABASE"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    conn.autocommit = False
    last_id = args.start_id
    processed = 0
    started = time.time()

    try:
        while True:
            batch_size = max(1, min(args.batch_size, 5000))
            if args.limit:
                remaining = args.limit - processed
                if remaining <= 0:
                    break
                batch_size = min(batch_size, remaining)
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = %s", (args.statement_timeout_ms,))
                cur.execute(SQL, (last_id, batch_size))
                ids = [row[0] for row in cur.fetchall()]
            conn.commit()
            if not ids:
                break
            last_id = max(ids)
            processed += len(ids)
            elapsed = max(time.time() - started, 0.001)
            print(f"backfilled={processed} last_id={last_id} rows_per_sec={processed/elapsed:.1f}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    finally:
        conn.close()

    print(f"Done. Backfilled {processed} rows.")


if __name__ == "__main__":
    main()
