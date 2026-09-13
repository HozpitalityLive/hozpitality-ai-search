"""Manual smoke test for V6 global retrieval.

Usage: python scripts_test_global_search.py "who is Raj Bhatt"
"""
import asyncio
import os
import sys
from main import HozpitalitySchemaIntelligence, GlobalSearchService


def main():
    message = " ".join(sys.argv[1:]).strip()
    if not message:
        raise SystemExit('Pass a query, e.g. "who is Raj Bhatt"')
    config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("POSTGRES_DATABASE"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }
    schema = HozpitalitySchemaIntelligence(lambda: __import__('psycopg2').connect(**config))
    service = GlobalSearchService(config, schema)
    print(service.search(message, limit=10) or "NO GLOBAL RESULTS")


if __name__ == "__main__":
    main()
