"""
Bootstrap the MongoDB search_documents collection from V6 PostgreSQL's
master_search_mastersearchindex.

This is intentionally a bootstrap/index-build utility. It does not replace
the authoritative application database.

Usage:
    python ai_search/scripts/sync_from_postgres.py --confirm
    python ai_search/scripts/sync_from_postgres.py --confirm --limit 5000
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any

import psycopg2
import psycopg2.extras
from pymongo import MongoClient
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"), override=True)
load_dotenv(os.path.join(ROOT, "ai_search/.env"), override=True)

ENTITY_MAP = {
    "job": "job",
    "professional": "professional",
    "company": "company",
    "product": "product",
    "article": "article",
    "event": "event",
    "award": "award",
    "faq": "faq",
}

SENSITIVE = {
    "password", "password_hash", "token", "access_token", "refresh_token",
    "secret", "secret_key", "api_key", "private_key", "otp", "otp_code",
    "session_data", "email", "phone", "phone_number",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    return re.sub(r"\s+", " ", text).strip()


def normalized(value: Any) -> str:
    return clean(value).casefold()


def row_to_document(row: dict[str, Any]) -> dict[str, Any]:
    model = normalized(row.get("model") or row.get("entity_type"))
    entity = ENTITY_MAP.get(model, model)
    title = clean(row.get("title"))
    content = clean(row.get("content"))
    keywords = [x.strip() for x in clean(row.get("ai_keywords")).split(",") if x.strip()]
    aliases = [x.strip() for x in clean(row.get("aliases")).split(",") if x.strip()]

    city = clean(row.get("city_text") or row.get("location_text"))
    country = clean(row.get("country_text"))

    search_text = " | ".join(
        x for x in [title, clean(row.get("entity_name")), clean(row.get("category_text")),
                    clean(row.get("company_name")), clean(row.get("user_name")),
                    city, country, *keywords, *aliases, clean(row.get("ai_summary")),
                    content]
        if x
    )

    is_live = row.get("is_live")
    if is_live is None:
        is_live = True

    return {
        "entity_type": entity,
        "entity_id": str(row.get("object_id")),
        "title": title,
        "description": clean(row.get("ai_summary") or content)[:4000],
        "keywords": keywords,
        "aliases": aliases,
        "aliases_normalized": [normalized(x) for x in aliases],
        "category": clean(row.get("category_text")),
        "status": clean(row.get("status") or ("active" if is_live else "inactive")),
        "status_normalized": normalized(row.get("status") or ("active" if is_live else "inactive")),
        "is_live": bool(is_live),
        "location": {
            "city": city,
            "country": country,
            "city_normalized": normalized(city),
            "country_normalized": normalized(country),
        },
        "url": clean(row.get("url") or row.get("slug")),
        "image": clean(row.get("image")) or None,
        "search_text": search_text,
        "source": {
            "master_search_id": row.get("id"),
            "content_type_id": row.get("content_type_id"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    if not args.confirm:
        raise SystemExit("Refusing to sync without --confirm.")

    pg = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        connect_timeout=10,
    )
    mongo = MongoClient(
        os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017"),
        maxPoolSize=int(os.getenv("MONGODB_MAX_POOL_SIZE", "100")),
    )
    collection = mongo[
        os.getenv("MONGODB_DATABASE", "hozpitality")
    ][os.getenv("MONGODB_COLLECTION", "search_documents")]

    repo = collection

    with pg, pg.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        query = """
            SELECT
                si.id, si.object_id, si.content_type_id, ct.model AS model, si.title,
                si.entity_name, si.category_text, si.company_name,
                si.user_name, si.subcategory_text, si.country_text,
                si.city_text, si.location_text, si.ai_keywords,
                si.ai_summary, si.content, si.slug, si.is_live,
                si.created_at, si.expires_at
            FROM public.master_search_mastersearchindex si
            LEFT JOIN public.django_content_type ct
              ON ct.id = si.content_type_id
            WHERE COALESCE(si.is_live, TRUE) = TRUE
            ORDER BY si.id
        """
        if args.limit:
            query += " LIMIT %s"
            cur.execute(query, (args.limit,))
        else:
            cur.execute(query)

        batch = []
        total = 0
        for row in cur:
            # V6 master index does not always expose the Django model name
            # directly. Prefer a stored entity_type if one exists.
            document = row_to_document(row)
            if document["entity_type"] not in ENTITY_MAP.values():
                continue
            batch.append(document)
            if len(batch) >= args.batch_size:
                _write(repo, batch)
                total += len(batch)
                batch.clear()
                print(f"Indexed {total} documents")
        if batch:
            _write(repo, batch)
            total += len(batch)

    print(f"MongoDB search_documents bootstrap complete: {total} documents")


def _write(collection, documents: list[dict]) -> None:
    from pymongo import UpdateOne

    operations = [
        UpdateOne(
            {"entity_type": d["entity_type"], "entity_id": d["entity_id"]},
            {"$set": d},
            upsert=True,
        )
        for d in documents
    ]
    collection.bulk_write(operations, ordered=False)
    collection.create_index(
        [
            ("title", "text"), ("aliases", "text"), ("keywords", "text"),
            ("category", "text"), ("location.city", "text"),
            ("location.country", "text"), ("description", "text"),
            ("search_text", "text")
        ],
        name="search_documents_text_v1",
        weights={"title": 10, "aliases": 8, "keywords": 6, "category": 4,
                 "location.city": 3, "location.country": 3,
                 "search_text": 2, "description": 1},
    )


if __name__ == "__main__":
    main()
