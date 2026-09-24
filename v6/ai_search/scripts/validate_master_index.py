"""
Read-only validation of the canonical V6 master search index.

This does not write PostgreSQL or MongoDB.

Usage:
    python -m ai_search.scripts.validate_master_index
"""

from __future__ import annotations

import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)


def main():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        connect_timeout=10,
    )

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (
                        WHERE COALESCE(ai_search_text, '') <> ''
                    ) AS ai_search_text_ready,
                    COUNT(*) FILTER (
                        WHERE metadata IS NOT NULL
                    ) AS metadata_ready,
                    COUNT(*) FILTER (
                        WHERE COALESCE(is_live, TRUE) = TRUE
                    ) AS live_rows
                FROM public.master_search_mastersearchindex
            """)
            print(dict(cur.fetchone()))

            cur.execute("""
                SELECT
                    lower(ct.model) AS model,
                    COUNT(*) AS total,
                    COUNT(*) FILTER (
                        WHERE COALESCE(si.ai_search_text, '') <> ''
                    ) AS ai_search_text_ready
                FROM public.master_search_mastersearchindex si
                LEFT JOIN public.django_content_type ct
                  ON ct.id = si.content_type_id
                GROUP BY lower(ct.model)
                ORDER BY lower(ct.model)
            """)

            print("\nPer-model coverage:")
            for row in cur.fetchall():
                print(dict(row))

    finally:
        conn.close()


if __name__ == "__main__":
    main()
