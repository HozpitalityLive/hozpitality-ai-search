# copy from dev to opt
# sudo cp ~/hozpitality-mongo-migration/migrate_faqs.py /opt/hozpitality-mongo-migration/migrate_faqs.py

# sudo -u postgres env MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_faqs.py \
# --limit=100

# sudo -u postgres env \
# MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_faqs.py


import argparse
import logging
import os
import re
from datetime import date, datetime, timezone

import psycopg2
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

POSTGRES_DB = os.getenv("POSTGRES_DB", "hozpitality")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")

MONGO_URI = os.getenv("MONGO_URI")

MONGO_DB = os.getenv("MONGO_DB", "mongoAdmin")
MONGO_COLLECTION = os.getenv(
    "MONGO_COLLECTION",
    "search_documents",
)

BATCH_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("faq-migration")

def get_postgres_connection():
    logger.info(
        "Connecting to PostgreSQL %s/%s using local peer authentication",
        POSTGRES_USER,
        POSTGRES_DB,
    )

    return psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
    )

def get_mongo_collection():
    if not MONGO_URI:
        raise RuntimeError(
            "MONGO_URI environment variable is required."
        )

    client = MongoClient(MONGO_URI)

    db = client[MONGO_DB]

    return client, db[MONGO_COLLECTION]

def json_safe(value):
    """
    Convert PostgreSQL values into MongoDB-compatible values.

    PostgreSQL DATE is converted to a BSON-compatible datetime.
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, date):
        return datetime(
            value.year,
            value.month,
            value.day,
            tzinfo=timezone.utc,
        )

    return value

def clean_text(value):
    """
    FAQ data currently has no HTML, but normalize whitespace
    and PostgreSQL CR/LF formatting.
    """

    if value is None:
        return ""

    value = str(value)

    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    lines = [
        re.sub(r"[ \t]+$", "", line)
        for line in value.split("\n")
    ]

    value = "\n".join(lines)

    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()

def normalize_question(question):
    """
    Create a clean searchable question.

    Some existing FAQs contain numbering such as:
        1. How do I register...?

    Preserve the original question in `question`, but remove
    the leading number for search aliases/keywords.
    """

    question = clean_text(question)

    normalized = re.sub(
        r"^\s*\d+[\.\)]\s*",
        "",
        question,
    )

    return normalized.strip()

def tokenize_search_text(text):
    """
    Lightweight keyword extraction.

    This is intentionally simple because embeddings will be
    generated separately later.
    """

    text = text.lower()

    words = re.findall(
        r"[a-zA-Z0-9][a-zA-Z0-9_\-]{2,}",
        text,
    )

    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "how",
        "what",
        "when",
        "where",
        "can",
        "you",
        "your",
        "are",
        "from",
        "this",
        "that",
        "have",
        "has",
        "will",
        "into",
        "does",
        "our",
        "their",
        "they",
        "not",
        "all",
        "any",
        "use",
        "using",
    }

    result = []

    seen = set()

    for word in words:
        if word in stop_words:
            continue

        if word not in seen:
            seen.add(word)
            result.append(word)

    return result[:50]

def build_ai_search_text(question, answer):
    return (
        f"FAQ Question: {question}\n"
        f"FAQ Answer: {answer}"
    )

def build_faq_document(row):
    faq_id = row["id"]

    question = clean_text(row["question"])
    answer = clean_text(row["answer"])

    normalized_question = normalize_question(question)

    ai_search_text = build_ai_search_text(
        normalized_question,
        answer,
    )

    search_keywords = tokenize_search_text(
        f"{normalized_question} {answer}"
    )

    search_aliases = [
        question,
        normalized_question,
    ]

    search_aliases = list(
        dict.fromkeys(
            alias
            for alias in search_aliases
            if alias
        )
    )

    document = {
        "_id": f"faq:{faq_id}",

        "entity_type": "faq",

        "question": question,
        "answer": answer,

        "flags": {
            "show_on_landing": bool(
                row["show_on_landing"]
            ),
            "is_popular": bool(
                row["is_popular"]
            ),
        },

        "dates": {
            "created_at": json_safe(
                row["created_at"]
            ),
            "updated_at": json_safe(
                row["updated_at"]
            ),
        },

        "search_keywords": search_keywords,

        "search_aliases": search_aliases,

        "ai_search_text": ai_search_text,

        "embedding": {
            "dimensions": 384,
            "status": "pending",
        },

        "status": "active",

        "source": {
            "model": "FAQ",
            "table": "base_faq",
            "object_id": faq_id,
        },

        "migration": {
            "source": "postgresql",
            "migrated_at": datetime.now(timezone.utc),
        },

        "schema_version": 1,
    }

    return document

def migrate_faqs(
    start_id=0,
    limit=None,
    batch_size=BATCH_SIZE,
):
    client, collection = get_mongo_collection()

    conn = get_postgres_connection()

    processed = 0
    written = 0
    errors = 0
    last_id = start_id

    try:
        with conn.cursor() as cursor:

            query = """
                SELECT
                    id,
                    question,
                    answer,
                    created_at,
                    updated_at,
                    is_popular,
                    show_on_landing
                FROM base_faq
                WHERE id > %s
                ORDER BY id
            """

            params = [start_id]

            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)

            logger.info(
                "Starting FAQ migration: start_id=%s limit=%s batch_size=%s",
                start_id,
                limit,
                batch_size,
            )

            cursor.execute(query, params)

            while True:
                rows = cursor.fetchmany(batch_size)

                if not rows:
                    break

                columns = [
                    "id",
                    "question",
                    "answer",
                    "created_at",
                    "updated_at",
                    "is_popular",
                    "show_on_landing",
                ]

                batch = [
                    dict(zip(columns, row))
                    for row in rows
                ]

                operations = []

                for row in batch:
                    try:
                        document = build_faq_document(row)

                        operations.append(
                            UpdateOne(
                                {
                                    "_id": document["_id"]
                                },
                                {
                                    "$set": document
                                },
                                upsert=True,
                            )
                        )

                        processed += 1
                        last_id = row["id"]

                    except Exception:
                        errors += 1

                        logger.exception(
                            "Failed building FAQ id=%s",
                            row["id"],
                        )

                if operations:

                    try:
                        result = collection.bulk_write(
                            operations,
                            ordered=False,
                        )

                        written += (
                            result.upserted_count
                            + result.modified_count
                        )

                    except BulkWriteError as exc:

                        errors += len(
                            exc.details.get(
                                "writeErrors",
                                [],
                            )
                        )

                        logger.error(
                            "MongoDB bulk write error: %s",
                            exc.details,
                        )

                logger.info(
                    "Batch complete: processed=%s written=%s errors=%s last_id=%s",
                    processed,
                    written,
                    errors,
                    last_id,
                )

    finally:
        conn.close()
        client.close()

    logger.info(
        "FAQ migration complete: processed=%s written=%s errors=%s last_id=%s",
        processed,
        written,
        errors,
        last_id,
    )

def main():
    parser = argparse.ArgumentParser(
        description="Migrate PostgreSQL FAQs to MongoDB search_documents."
    )

    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    args = parser.parse_args()

    migrate_faqs(
        start_id=args.start_id,
        limit=args.limit,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()