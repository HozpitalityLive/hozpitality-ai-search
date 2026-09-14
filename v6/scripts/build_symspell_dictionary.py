#!/usr/bin/env python3
"""Build the SymSpell vocabulary from live Hozpitality master-search data.

Run after migrations/backfills:
    python scripts/build_symspell_dictionary.py
"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg2


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "symspell_dictionary.txt"


def main() -> None:
    cfg = {
        "host": os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "127.0.0.1")),
        "port": int(os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432"))),
        "dbname": os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "hozpitality")),
        "user": os.getenv("POSTGRES_USER", os.getenv("DB_USER", "postgres")),
        "password": os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "")),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Frequency is derived from the actual indexed Hozpitality corpus, so
    # corrections follow the platform's vocabulary rather than a hardcoded
    # typo/alias table.
    query = """
    WITH words AS (
        SELECT lower(word) AS word
        FROM master_search_mastersearchindex si
        CROSS JOIN LATERAL regexp_split_to_table(
            unaccent(
                concat_ws(' ', si.title, si.category_text, si.location_text,
                          si.user_name, si.ai_keywords, si.slug)
            ),
            '[^[:alnum:]_]+'
        ) AS word
        WHERE si.is_live = TRUE
          AND length(word) >= 2
          AND length(word) <= 40
    )
    SELECT word, COUNT(*)::bigint AS frequency
    FROM words
    GROUP BY word
    HAVING COUNT(*) >= 1
    ORDER BY frequency DESC, word
    """
    with psycopg2.connect(**cfg) as conn, conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    tmp = OUT.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for word, frequency in rows:
            f.write(f"{word}\t{frequency}\n")
    tmp.replace(OUT)
    print(f"Wrote {len(rows):,} vocabulary terms to {OUT}")


if __name__ == "__main__":
    main()
