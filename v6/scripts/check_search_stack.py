"""Verify PostgreSQL search extensions, indexes, vectors and master-index health."""
from __future__ import annotations

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DATABASE"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT current_setting('server_version')")
            print("PostgreSQL:", cur.fetchone()[0])

            cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','pg_trgm','unaccent') ORDER BY extname")
            print("Extensions:")
            for row in cur.fetchall():
                print(" ", row)

            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE search_vector_v6 IS NOT NULL) AS fts_ready,
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS vectors_ready,
                    COUNT(DISTINCT content_type_id) AS content_types
                FROM public.master_search_mastersearchindex
            """)
            print("Master index:", cur.fetchone())

            cur.execute("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname='public'
                  AND tablename='master_search_mastersearchindex'
                ORDER BY indexname
            """)
            print("Master indexes:")
            for name, definition in cur.fetchall():
                print(f"  {name}: {definition}")

            cur.execute("""
                SELECT
                    COUNT(*) AS total_tables,
                    (SELECT COUNT(*) FROM information_schema.table_constraints WHERE table_schema='public' AND constraint_type='FOREIGN KEY') AS foreign_keys
                FROM information_schema.tables
                WHERE table_schema='public' AND table_type='BASE TABLE'
            """)
            print("Schema:", cur.fetchone())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
