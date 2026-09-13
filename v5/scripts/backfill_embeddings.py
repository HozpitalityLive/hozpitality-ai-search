"""Optional, throttled embedding backfill for master_search_mastersearchindex.

Uses sentence-transformers locally. It only fills NULL embeddings and works by
primary-key batches, so it does not load the entire 500K+ dataset into memory.
For the first production run use --limit 1000, inspect DB load, then continue.
"""
from __future__ import annotations

import argparse
import os
import time

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--db-batch", type=int, default=512)
    parser.add_argument("--sleep", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to generate embeddings without --confirm.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Install the embedding extras first: pip install sentence-transformers") from exc

    model = SentenceTransformer(args.model)
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"), port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DATABASE"), user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    conn.autocommit = False
    last_id = 0
    processed = 0

    try:
        while True:
            db_batch = max(1, min(args.db_batch, 2000))
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, concat_ws(' | ', title, category_text, location_text, user_name, ai_keywords, content, slug) AS text
                    FROM public.master_search_mastersearchindex
                    WHERE id > %s AND embedding IS NULL
                    ORDER BY id
                    LIMIT %s
                """, (last_id, db_batch))
                rows = cur.fetchall()
            if not rows:
                break
            texts = [str(r["text"] or "")[:12000] for r in rows]
            embeddings = model.encode(texts, batch_size=max(1, min(args.batch_size, 512)), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            values = []
            for row, vector in zip(rows, embeddings):
                if len(vector) != 384:
                    raise RuntimeError(f"Model produced {len(vector)} dimensions; database expects 384")
                values.append((int(row["id"]), "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"))
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """UPDATE public.master_search_mastersearchindex AS t
                       SET embedding = v.embedding::vector
                       FROM (VALUES %s) AS v(id, embedding)
                       WHERE t.id = v.id""",
                    values,
                    template="(%s, %s)",
                    page_size=256,
                )
            conn.commit()
            last_id = int(rows[-1]["id"])
            processed += len(rows)
            print(f"embedded={processed} last_id={last_id}", flush=True)
            if args.limit and processed >= args.limit:
                break
            if args.sleep:
                time.sleep(args.sleep)
    finally:
        conn.close()

    print(f"Done. Generated {processed} embeddings.")


if __name__ == "__main__":
    main()
