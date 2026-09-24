"""
Build MongoDB search_documents from the canonical V6 PostgreSQL
master_search_mastersearchindex.

This is intentionally model-agnostic. The relationship-aware content has
already been materialized into master_search_mastersearchindex.ai_search_text
and metadata by the V6 backfill. We therefore do NOT duplicate the eight
source-model migration queries here.

Usage:
    python -m ai_search.scripts.sync_from_master_index --confirm --limit 1000
    python -m ai_search.scripts.sync_from_master_index --confirm
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / "ai_search" / ".env", override=True)

MODEL_ALIASES = {
    "award": {"award", "awards"},
    "professional": {"professional", "professionals"},
    "company": {"company", "companies"},
    "article": {"article", "articles"},
    "job": {"job", "jobs"},
    "event": {"event", "events"},
    "product": {"product", "products"},
    "faq": {"faq", "faqs"},
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    return re.sub(r"\s+", " ", text).strip()


def norm(value: Any) -> str:
    return clean(value).casefold()


def canonical_model(value: Any) -> str:
    model = norm(value)
    for canonical, aliases in MODEL_ALIASES.items():
        if model in aliases:
            return canonical
    return model


def json_safe(value: Any) -> Any:
    """Convert values that Mongo/Python cannot serialize consistently."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def extract_status(metadata: dict[str, Any]) -> str:
    relationships = metadata.get("relationships") or {}

    # Most V6 entities use relationships.status.
    status = relationships.get("status")
    if isinstance(status, dict):
        status = first_non_empty(
            status.get("name"),
            status.get("status"),
            status.get("is_active"),
        )

    if status not in (None, ""):
        return clean(status)

    # Some source models expose an active flag instead.
    for path in (
        ("relationships", "award", "status"),
        ("relationships", "event", "status"),
        ("relationships", "product", "status"),
    ):
        value = nested_get(metadata, *path)
        if value not in (None, ""):
            return clean(value)

    return "active"


def extract_image(metadata: dict[str, Any]) -> str | None:
    relationships = metadata.get("relationships") or {}

    candidates = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_l = str(key).lower()
                if key_l in {
                    "image", "image_url", "thumbnail", "thumbnail_url",
                    "profile_image", "cover_image",
                } and isinstance(item, str) and item.strip():
                    candidates.append(item.strip())
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(relationships)
    return candidates[0] if candidates else None


def extract_url(master: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    relationships = metadata.get("relationships") or {}

    candidates = [
        master.get("slug"),
        nested_get(relationships, "job", "slug"),
        nested_get(relationships, "article", "slug"),
        nested_get(relationships, "event", "slug"),
        nested_get(relationships, "product", "slug"),
        nested_get(relationships, "award", "slug"),
    ]

    for value in candidates:
        value = clean(value)
        if value:
            return value
    return None


def build_document(row: dict[str, Any]) -> dict[str, Any]:
    model = canonical_model(row.get("model"))
    metadata = json_safe(row.get("metadata") or {})

    title = clean(row.get("title"))
    entity_name = clean(row.get("entity_name"))
    category = clean(row.get("category_text"))
    company_name = clean(row.get("company_name"))
    user_name = clean(row.get("user_name"))
    ai_keywords = clean(row.get("ai_keywords"))
    ai_summary = clean(row.get("ai_summary"))
    content = clean(row.get("content"))
    ai_search_text = clean(row.get("ai_search_text"))

    city = clean(row.get("city_text"))
    country = clean(row.get("country_text"))
    location_text = clean(row.get("location_text"))

    # Canonical V6 ai_search_text is authoritative for broad retrieval.
    # Keep the original structured fields separately for deterministic ranking.
    aliases: list[str] = []

    # Entity aliases are semantic labels, not invented source facts.
    aliases.extend(
        {
            "job": ["jobs", "vacancy", "vacancies", "position", "career"],
            "professional": ["candidate", "candidates", "profile"],
            "company": ["employer", "employers", "hotel", "hotels"],
            "product": ["supplier", "suppliers", "marketplace"],
            "article": ["articles", "news", "story", "stories"],
            "event": ["events"],
            "award": ["awards"],
            "faq": ["faqs", "question", "questions"],
        }.get(model, [])
    )

    # Existing AI keywords are also useful exact alias signals.
    if ai_keywords:
        aliases.extend(
            part.strip()
            for part in re.split(r"[,|]", ai_keywords)
            if part.strip()
        )

    aliases = list(dict.fromkeys(aliases))

    status = extract_status(metadata)
    is_live = row.get("is_live")
    if is_live is None:
        is_live = True

    expires_at = row.get("expires_at")
    if expires_at is not None and not isinstance(expires_at, datetime):
        expires_at = None

    document = {
        "_id": f"{model}:{row.get('object_id')}",
        "schema_version": "phase1-v2",
        "entity_type": model,
        "entity_id": str(row.get("object_id")),
        "content_type_id": row.get("content_type_id"),

        "title": title,
        "title_normalized": norm(title),
        "entity_name": entity_name,
        "category": category,
        "company_name": company_name,
        "user_name": user_name,

        "ai_keywords": ai_keywords,
        "ai_keywords_normalized": norm(ai_keywords),
        "aliases": aliases,
        "aliases_normalized": [norm(x) for x in aliases],

        "ai_summary": ai_summary,
        "description": (ai_summary or content)[:6000],
        "content": content[:12000],

        "ai_search_text": ai_search_text,

        "city": city,
        "city_normalized": norm(city),
        "country": country,
        "country_normalized": norm(country),
        "location_text": location_text,

        "status": status,
        "status_normalized": norm(status),
        "is_live": bool(is_live),
        "expires_at": expires_at,

        "url": extract_url(row, metadata),
        "image": extract_image(metadata),

        # Keep the relationship-aware V6 metadata available for Phase 2+
        # without requiring another source-table join at query time.
        "metadata": metadata,

        "source": {
            "master_search_id": row.get("id"),
            "object_id": row.get("object_id"),
            "content_type_id": row.get("content_type_id"),
            "model": model,
        },

        "created_at": row.get("created_at"),
    }

    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    if not args.confirm:
        raise SystemExit(
            "Refusing production sync without --confirm."
        )

    pg = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        connect_timeout=10,
    )

    mongo = MongoClient(
        os.getenv("MONGODB_URI"),
        maxPoolSize=int(os.getenv("MONGODB_MAX_POOL_SIZE", "100")),
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
    )
    mongo.admin.command("ping")

    collection = mongo[
        os.getenv("MONGODB_DATABASE", "hozpitality")
    ][
        os.getenv("MONGODB_COLLECTION", "search_documents")
    ]

    model_filter = canonical_model(args.model) if args.model else None

    query = """
        SELECT
            si.id,
            si.object_id,
            si.content_type_id,
            ct.model AS model,
            si.title,
            si.entity_name,
            si.category_text,
            si.company_name,
            si.user_name,
            si.subcategory_text,
            si.country_text,
            si.city_text,
            si.location_text,
            si.ai_keywords,
            si.ai_summary,
            si.content,
            si.ai_search_text,
            si.slug,
            si.is_live,
            si.created_at,
            si.expires_at,
            si.metadata
        FROM public.master_search_mastersearchindex si
        LEFT JOIN public.django_content_type ct
          ON ct.id = si.content_type_id
        WHERE COALESCE(si.ai_search_text, '') <> ''
          AND COALESCE(si.is_live, TRUE) = TRUE
    """

    params: list[Any] = []

    if model_filter:
        aliases = sorted(MODEL_ALIASES.get(model_filter, {model_filter}))
        query += " AND lower(ct.model) = ANY(%s)"
        params.append(aliases)

    query += " ORDER BY si.id"

    if args.limit:
        query += " LIMIT %s"
        params.append(args.limit)

    total = 0
    batch: list[dict[str, Any]] = []

    try:
        with pg.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)

            for row in cur:
                document = build_document(dict(row))

                if document["entity_type"] not in MODEL_ALIASES:
                    continue

                batch.append(document)

                if len(batch) >= max(1, args.batch_size):
                    write_batch(collection, batch)
                    total += len(batch)
                    print(f"Indexed {total} documents", flush=True)
                    batch.clear()

            if batch:
                write_batch(collection, batch)
                total += len(batch)

    finally:
        pg.close()
        mongo.close()

    print(
        f"MongoDB search_documents sync complete: {total} documents",
        flush=True,
    )


def write_batch(collection, documents: list[dict[str, Any]]) -> None:
    operations = [
        UpdateOne(
            {
                "entity_type": document["entity_type"],
                "entity_id": document["entity_id"],
            },
            {"$set": document},
            upsert=True,
        )
        for document in documents
    ]

    collection.bulk_write(operations, ordered=False)


if __name__ == "__main__":
    main()
