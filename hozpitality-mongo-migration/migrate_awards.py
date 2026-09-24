# copy from dev to opt
# sudo cp ~/hozpitality-mongo-migration/migrate_awards.py /opt/hozpitality-mongo-migration/migrate_awards.py

# sudo -u postgres env MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_awards.py \
# --limit=1

# sudo -u postgres env \
# MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_awards.py


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

CLOUDFRONT_BASE = os.getenv(
    "CLOUDFRONT_BASE",
    "https://d2he8nskrbhxwq.cloudfront.net",
).rstrip("/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("award-migration")

def get_postgres_connection():
    """
    PostgreSQL uses local Unix socket + peer authentication.
    Script is executed as postgres user.
    """

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

    PostgreSQL DATE values are converted to BSON datetime at
    midnight UTC because Python datetime.date is not directly
    supported by BSON.
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
    Normalize PostgreSQL text while preserving meaningful
    newlines.
    """

    if value is None:
        return None

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

def clean_optional_text(value):
    value = clean_text(value)

    if value == "":
        return None

    return value

def build_media_url(path):
    """
    Convert Django MediaStorage relative paths into CloudFront URLs.

    If the database already contains an absolute URL, preserve it.
    """

    if not path:
        return None

    path = str(path).strip()

    if not path:
        return None

    if path.startswith("http://") or path.startswith("https://"):
        return path

    return f"{CLOUDFRONT_BASE}/{path.lstrip('/')}"

def build_search_keywords(*values):
    """
    Lightweight keyword extraction.

    Embeddings are intentionally generated separately later.
    """

    text_parts = []

    for value in values:
        if value:
            text_parts.append(str(value))

    text = " ".join(text_parts).lower()

    words = re.findall(
        r"[a-zA-Z0-9][a-zA-Z0-9_\-]{2,}",
        text,
    )

    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "are",
        "was",
        "were",
        "has",
        "have",
        "will",
        "into",
        "your",
        "their",
        "about",
        "best",
        "list",
        "award",
        "awards",
        "2026",
        "2025",
        "2024",
        "2023",
        "2022",
        "2021",
        "2020",
    }

    result = []
    seen = set()

    for word in words:
        if word in stop_words:
            continue

        if word not in seen:
            seen.add(word)
            result.append(word)

    return result[:100]

def build_ai_search_text(
    title,
    short_title,
    subtitle,
    description,
    year,
    country,
    location,
    categories,
):
    """
    Build the semantic text used later for embeddings/vector search.

    URLs are intentionally NOT included here.
    """

    parts = []

    if title:
        parts.append(f"Award: {title}")

    if short_title:
        parts.append(f"Short title: {short_title}")

    if subtitle:
        parts.append(f"Subtitle: {subtitle}")

    if description:
        parts.append(f"Description: {description}")

    if year:
        parts.append(f"Award year: {year}")

    if country:
        country_name = country.get("name")
        country_code = country.get("code")

        country_text = country_name or ""

        if country_code:
            country_text += f" ({country_code})"

        if country_text:
            parts.append(f"Country: {country_text}")

    if location:
        parts.append(f"Location: {location}")

    category_names = [
        category.get("name")
        for category in categories
        if category.get("name")
    ]

    if category_names:
        parts.append(
            "Categories: " + ", ".join(category_names)
        )

    return "\n".join(parts)

def load_countries(cursor):
    """
    Load country metadata used by Awards and AwardCategories.
    """

    cursor.execute(
        """
        SELECT
            id,
            db_id,
            name,
            ac_name,
            country_code,
            sub_domain,
            flag,
            code
        FROM countries
        """
    )

    countries = {}

    for row in cursor.fetchall():
        (
            country_id,
            db_id,
            name,
            ac_name,
            country_code,
            sub_domain,
            flag,
            code,
        ) = row

        countries[country_id] = {
            "id": country_id,
            "external_id": db_id,
            "name": clean_optional_text(name),
            "ac_name": clean_optional_text(ac_name),
            "country_code": clean_optional_text(country_code),
            "sub_domain": clean_optional_text(sub_domain),
            "flag": clean_optional_text(flag),
            "code": clean_optional_text(code),
        }

    logger.info(
        "Loaded %s countries",
        len(countries),
    )

    return countries

def load_award_categories(cursor, countries):
    """
    Load AwardCategory records.

    Returns:
        {
            category_id: {
                id,
                name,
                is_active,
                created_at,
                country
            }
        }
    """

    cursor.execute(
        """
        SELECT
            id,
            category_name,
            category_is_active,
            category_created_at,
            country_id
        FROM base_awardcategory
        ORDER BY id
        """
    )

    categories = {}

    for row in cursor.fetchall():
        (
            category_id,
            category_name,
            category_is_active,
            category_created_at,
            country_id,
        ) = row

        categories[category_id] = {
            "id": category_id,
            "name": clean_optional_text(category_name),
            "is_active": bool(category_is_active),
            "created_at": json_safe(category_created_at),
            "country": countries.get(country_id),
        }

    logger.info(
        "Loaded %s AwardCategories",
        len(categories),
    )

    return categories

def load_award_category_relations(cursor):
    """
    Load Awards ↔ AwardCategory M2M relationships.

    Returns:
        {
            award_id: [category_id, category_id, ...]
        }
    """

    cursor.execute(
        """
        SELECT
            awards_id,
            awardcategory_id
        FROM base_awards_award_category
        ORDER BY awards_id, awardcategory_id
        """
    )

    relations = {}

    for award_id, category_id in cursor.fetchall():
        relations.setdefault(
            award_id,
            [],
        ).append(category_id)

    total_relationships = sum(
        len(value)
        for value in relations.values()
    )

    logger.info(
        "Loaded %s AwardCategory relationships across %s awards",
        total_relationships,
        len(relations),
    )

    return relations

AWARD_COLUMNS = """
    id,
    sequence_number,
    award_title,
    award_description,
    award_image,
    award_is_active,
    award_created_at,
    voting_start_date,
    voting_end_date,
    is_voting_active,
    youtube_link,
    country_id,
    location,
    slug,
    award_bg_image,
    award_bg_personal_image,
    award_bg_corporate_image,
    award_on,
    award_short_title,
    award_subtitle,
    award_video,
    corporate_categories_link,
    nomination_link,
    personal_categories_link,
    show_corporate_categories_button,
    show_nomination_button,
    show_personal_categories_button,
    show_vote_now_button,
    vote_now_link,
    award_avatar_image,
    latitude,
    longitude,
    award_winners_link,
    show_award_winners_button,
    sequence_number,
    award_detail_url,
    code,
    award_year,
    hide_country_from_url
"""

def fetch_awards(cursor, start_id, limit):
    query = f"""
        SELECT
            {AWARD_COLUMNS}
        FROM base_awards
        WHERE id > %s
        ORDER BY id
    """

    params = [start_id]

    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    cursor.execute(query, params)

    columns = [
        "id",
        "sequence_number",
        "award_title",
        "award_description",
        "award_image",
        "award_is_active",
        "award_created_at",
        "voting_start_date",
        "voting_end_date",
        "is_voting_active",
        "youtube_link",
        "country_id",
        "location",
        "slug",
        "award_bg_image",
        "award_bg_personal_image",
        "award_bg_corporate_image",
        "award_on",
        "award_short_title",
        "award_subtitle",
        "award_video",
        "corporate_categories_link",
        "nomination_link",
        "personal_categories_link",
        "show_corporate_categories_button",
        "show_nomination_button",
        "show_personal_categories_button",
        "show_vote_now_button",
        "vote_now_link",
        "award_avatar_image",
        "latitude",
        "longitude",
        "award_winners_link",
        "show_award_winners_button",
        "sequence_number_duplicate",
        "award_detail_url",
        "code",
        "award_year",
        "hide_country_from_url",
    ]

    rows = cursor.fetchall()

    awards = []

    for row in rows:
        data = dict(zip(columns, row))

        data.pop("sequence_number_duplicate", None)

        awards.append(data)

    return awards

def build_award_document(
    award,
    countries,
    categories,
    category_relations,
):
    award_id = award["id"]

    country = countries.get(
        award["country_id"]
    )

    category_ids = category_relations.get(
        award_id,
        [],
    )

    award_categories = []

    for category_id in category_ids:
        category = categories.get(category_id)

        if not category:
            logger.warning(
                "Award %s references missing category %s",
                award_id,
                category_id,
            )
            continue

        award_categories.append({
            "id": category["id"],
            "name": category["name"],
            "is_active": category["is_active"],
            "created_at": category["created_at"],
            "country": category["country"],
        })

    title = clean_optional_text(
        award["award_title"]
    )

    short_title = clean_optional_text(
        award["award_short_title"]
    )

    subtitle = clean_optional_text(
        award["award_subtitle"]
    )

    description = clean_optional_text(
        award["award_description"]
    )

    location = clean_optional_text(
        award["location"]
    )

    ai_search_text = build_ai_search_text(
        title=title,
        short_title=short_title,
        subtitle=subtitle,
        description=description,
        year=award["award_year"],
        country=country,
        location=location,
        categories=award_categories,
    )

    search_keywords = build_search_keywords(
        title,
        short_title,
        subtitle,
        description,
        location,
        award["award_year"],
        country.get("name") if country else None,
        *[
            category["name"]
            for category in award_categories
        ],
    )

    search_aliases = []

    for value in (
        title,
        short_title,
        subtitle,
        award["code"],
        award["slug"],
    ):
        value = clean_optional_text(value)

        if value and value not in search_aliases:
            search_aliases.append(value)

    document = {
        "_id": f"award:{award_id}",

        "entity_type": "award",

        "title": title,
        "short_title": short_title,
        "subtitle": subtitle,
        "description": description,

        "year": award["award_year"],
        "sequence_number": award["sequence_number"],

        "status": (
            "active"
            if award["award_is_active"]
            else "inactive"
        ),

        "is_active": bool(
            award["award_is_active"]
        ),

        "country": country,

        "location": location,

        "coordinates": {
            "latitude": award["latitude"],
            "longitude": award["longitude"],
        },

        "categories": award_categories,

        "media": {
            "image": build_media_url(
                award["award_image"]
            ),
            "background": build_media_url(
                award["award_bg_image"]
            ),
            "personal_background": build_media_url(
                award["award_bg_personal_image"]
            ),
            "corporate_background": build_media_url(
                award["award_bg_corporate_image"]
            ),
            "avatar": build_media_url(
                award["award_avatar_image"]
            ),
            "video": build_media_url(
                award["award_video"]
            ),
        },

        "voting": {
            "is_active": bool(
                award["is_voting_active"]
            ),
            "start_date": json_safe(
                award["voting_start_date"]
            ),
            "end_date": json_safe(
                award["voting_end_date"]
            ),
        },

        "award_date": json_safe(
            award["award_on"]
        ),

        "buttons": {
            "show_nomination": bool(
                award["show_nomination_button"]
            ),
            "show_vote_now": bool(
                award["show_vote_now_button"]
            ),
            "show_personal_categories": bool(
                award[
                    "show_personal_categories_button"
                ]
            ),
            "show_corporate_categories": bool(
                award[
                    "show_corporate_categories_button"
                ]
            ),
            "show_award_winners": bool(
                award[
                    "show_award_winners_button"
                ]
            ),
        },

        "links": {
            "detail": clean_optional_text(
                award["award_detail_url"]
            ),
            "youtube": clean_optional_text(
                award["youtube_link"]
            ),
            "nomination": clean_optional_text(
                award["nomination_link"]
            ),
            "vote_now": clean_optional_text(
                award["vote_now_link"]
            ),
            "personal_categories": clean_optional_text(
                award["personal_categories_link"]
            ),
            "corporate_categories": clean_optional_text(
                award["corporate_categories_link"]
            ),
            "winners": clean_optional_text(
                award["award_winners_link"]
            ),
        },

        "slug": clean_optional_text(
            award["slug"]
        ),

        "code": clean_optional_text(
            award["code"]
        ),

        "hide_country_from_url": bool(
            award["hide_country_from_url"]
        ),

        "dates": {
            "created_at": json_safe(
                award["award_created_at"]
            ),
            "voting_start": json_safe(
                award["voting_start_date"]
            ),
            "voting_end": json_safe(
                award["voting_end_date"]
            ),
            "award_on": json_safe(
                award["award_on"]
            ),
        },

        "search_keywords": search_keywords,

        "search_aliases": search_aliases,

        "ai_search_text": ai_search_text,

        "embedding": {
            "dimensions": 384,
            "status": "pending",
        },

        "source": {
            "model": "Awards",
            "table": "base_awards",
            "object_id": award_id,
        },

        "migration": {
            "source": "postgresql",
            "migrated_at": datetime.now(
                timezone.utc
            ),
        },

        "schema_version": 1,
    }

    return document

def migrate_awards(
    start_id=0,
    limit=None,
    batch_size=BATCH_SIZE,
):
    mongo_client, collection = get_mongo_collection()

    pg_conn = get_postgres_connection()

    processed = 0
    written = 0
    errors = 0
    last_id = start_id

    try:
        with pg_conn.cursor() as cursor:

            logger.info(
                "Loading country data..."
            )

            countries = load_countries(
                cursor
            )

            logger.info(
                "Loading AwardCategory data..."
            )

            categories = load_award_categories(
                cursor,
                countries,
            )

            logger.info(
                "Loading Award ↔ Category relationships..."
            )

            category_relations = (
                load_award_category_relations(
                    cursor
                )
            )

            logger.info(
                "Starting Awards migration..."
            )

            logger.info(
                "start_id=%s limit=%s batch_size=%s",
                start_id,
                limit,
                batch_size,
            )

            remaining = limit

            while True:

                current_limit = batch_size

                if remaining is not None:
                    if remaining <= 0:
                        break

                    current_limit = min(
                        batch_size,
                        remaining,
                    )

                awards = fetch_awards(
                    cursor,
                    last_id,
                    current_limit,
                )

                if not awards:
                    break

                operations = []

                for award in awards:

                    award_id = award["id"]

                    try:
                        document = build_award_document(
                            award=award,
                            countries=countries,
                            categories=categories,
                            category_relations=category_relations,
                        )

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
                        last_id = award_id

                    except Exception:
                        errors += 1

                        logger.exception(
                            "Failed building Award id=%s",
                            award_id,
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

                        write_errors = exc.details.get(
                            "writeErrors",
                            [],
                        )

                        errors += len(
                            write_errors
                        )

                        logger.error(
                            "MongoDB bulk write error: %s",
                            exc.details,
                        )

                logger.info(
                    "Batch complete: "
                    "processed=%s written=%s errors=%s last_id=%s",
                    processed,
                    written,
                    errors,
                    last_id,
                )

                if remaining is not None:
                    remaining -= len(awards)

                if len(awards) < current_limit:
                    break

    finally:
        pg_conn.close()
        mongo_client.close()

    logger.info(
        "Awards migration complete: "
        "processed=%s written=%s errors=%s last_id=%s",
        processed,
        written,
        errors,
        last_id,
    )

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Migrate PostgreSQL Awards to "
            "MongoDB search_documents."
        )
    )

    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Only migrate Awards with id > start-id.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of Awards to migrate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="MongoDB/PostgreSQL batch size.",
    )

    args = parser.parse_args()

    migrate_awards(
        start_id=args.start_id,
        limit=args.limit,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()