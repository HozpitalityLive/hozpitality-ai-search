#!/usr/bin/env python3
"""Build the SymSpell vocabulary from live Hozpitality master-search data.

Run after migrations/backfills:

    python scripts/build_symspell_dictionary.py
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
OUT = ROOT / "data" / "symspell_dictionary.txt"


# ---------------------------------------------------------------------------
# Load V6 environment
# ---------------------------------------------------------------------------

load_dotenv(ENV_FILE, override=True)


def get_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def get_db_config() -> dict:
    """Build PostgreSQL configuration from the V6 .env file."""

    return {
        "host": get_env(
            "POSTGRES_HOST",
            "DB_HOST",
            default="127.0.0.1",
        ),
        "port": int(
            get_env(
                "POSTGRES_PORT",
                "DB_PORT",
                default="5432",
            )
        ),
        "dbname": get_env(
            "POSTGRES_DB",
            "POSTGRES_DATABASE",
            "DB_NAME",
            default="hozpitality",
        ),
        "user": get_env(
            "POSTGRES_USER",
            "DB_USER",
            default="postgres",
        ),
        "password": get_env(
            "POSTGRES_PASSWORD",
            "DB_PASSWORD",
            default="",
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = get_db_config()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("==============================================")
    print(" Hozpitality V6 - SymSpell Dictionary Builder")
    print("==============================================")
    print(f"Environment : {ENV_FILE}")
    print(f"DB Host     : {cfg['host']}")
    print(f"DB Port     : {cfg['port']}")
    print(f"DB Name     : {cfg['dbname']}")
    print(f"DB User     : {cfg['user']}")
    print(f"Output      : {OUT}")
    print()

    if not cfg["password"]:
        raise RuntimeError(
            "PostgreSQL password is missing. "
            "Set POSTGRES_PASSWORD or DB_PASSWORD in .env"
        )

    # -----------------------------------------------------------------------
    # Query vocabulary directly from the Hozpitality master search index.
    #
    # This is intentionally data-driven.
    # There is NO hardcoded typo/alias dictionary.
    # -----------------------------------------------------------------------

    query = """
    WITH words AS (
        SELECT
            lower(word) AS word
        FROM master_search_mastersearchindex AS si
        CROSS JOIN LATERAL regexp_split_to_table(
            unaccent(
                concat_ws(
                    ' ',
                    si.title,
                    si.category_text,
                    si.location_text,
                    si.user_name,
                    si.ai_keywords,
                    si.slug
                )
            ),
            '[^[:alnum:]_]+'
        ) AS word
        WHERE si.is_live = TRUE
          AND length(word) >= 2
          AND length(word) <= 40
    )
    SELECT
        word,
        COUNT(*)::bigint AS frequency
    FROM words
    GROUP BY word
    HAVING COUNT(*) >= 1
    ORDER BY frequency DESC, word
    """

    print("Connecting to PostgreSQL...")

    try:
        with psycopg2.connect(**cfg) as conn:
            with conn.cursor() as cur:

                print("Reading live master-search vocabulary...")

                cur.execute(query)
                rows = cur.fetchall()

    except psycopg2.Error as exc:
        raise RuntimeError(
            "Unable to connect to PostgreSQL or read the "
            "master-search vocabulary."
        ) from exc

    print(f"Vocabulary terms: {len(rows):,}")

    # -----------------------------------------------------------------------
    # Atomic-ish file replacement.
    #
    # Write to .tmp first so an interrupted build does not destroy the
    # previous dictionary.
    # -----------------------------------------------------------------------

    tmp = OUT.with_suffix(".tmp")

    try:
        with tmp.open("w", encoding="utf-8") as f:
            for word, frequency in rows:
                f.write(f"{word} {frequency}\n")

        tmp.replace(OUT)

    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    print()
    print("SUCCESS")
    print("----------------------------------------------")
    print(f"Wrote       : {len(rows):,} vocabulary terms")
    print(f"Dictionary  : {OUT}")
    print("----------------------------------------------")


if __name__ == "__main__":
    main()