# copy from dev to opt
# sudo cp ~/hozpitality-mongo-migration/migrate_articles.py /opt/hozpitality-mongo-migration/migrate_articles.py

# sudo -u postgres env MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_articles.py \
# --limit=100

# sudo -u postgres env \
# MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_articles.py




import os
import re
import sys
import html
import argparse
import logging
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

POSTGRES_DB = os.getenv("POSTGRES_DB", "hozpitality")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

MONGO_URI = os.getenv("MONGO_URI")

MONGO_DB = os.getenv("MONGO_DB", "mongoAdmin")
MONGO_COLLECTION = os.getenv(
    "MONGO_COLLECTION",
    "search_documents",
)

BATCH_SIZE = int(os.getenv("ARTICLE_BATCH_SIZE", "100"))

MAX_CLEAN_CONTENT_CHARS = int(
    os.getenv("MAX_CLEAN_CONTENT_CHARS", "500000")
)

MAX_AI_SEARCH_TEXT_CHARS = int(
    os.getenv("MAX_AI_SEARCH_TEXT_CHARS", "750000")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("migrate_articles")

SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|template)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

COMMENT_RE = re.compile(
    r"<!--.*?-->",
    re.DOTALL,
)

TAG_RE = re.compile(
    r"<[^>]+>",
    re.DOTALL,
)

MULTISPACE_RE = re.compile(
    r"[ \t\r\f\v]+"
)

MULTINEWLINE_RE = re.compile(
    r"\n{3,}"
)

URL_ONLY_RE = re.compile(
    r"https?://\S+"
)

def clean_html_to_text(value):
    """
    Convert HTML/HTML-escaped content into readable text.

    PostgreSQL Article content can be extremely large because of:
      - HTML markup
      - embedded images
      - inline media
      - generated markup
      - tracking markup

    We deliberately do NOT preserve original HTML in MongoDB.
    """

    if not value:
        return ""

    text = str(value)

    text = COMMENT_RE.sub(" ", text)

    text = SCRIPT_STYLE_RE.sub(" ", text)

    text = re.sub(
        r"</(p|div|section|article|h[1-6]|li|ul|ol|blockquote|br|tr)>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    text = html.unescape(text)

    text = TAG_RE.sub(" ", text)

    text = html.unescape(text)

    text = text.replace("\xa0", " ")

    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)

    text = MULTISPACE_RE.sub(" ", text)

    text = MULTINEWLINE_RE.sub("\n\n", text)

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)

    return text.strip()

def limit_text(value, max_chars):
    """
    Safely limit a string by characters.
    """
    if not value:
        return ""

    if len(value) <= max_chars:
        return value

    return value[:max_chars].rstrip() + "\n[content truncated]"

def safe_str(value):
    if value is None:
        return ""

    return str(value).strip()

def build_person_name(first_name, last_name, username=None):
    first = safe_str(first_name)
    last = safe_str(last_name)

    name = " ".join(
        part for part in [first, last]
        if part
    ).strip()

    if name:
        return name

    return safe_str(username)

def unique_strings(values):
    """
    Remove empty values and duplicates while preserving order.
    """
    result = []
    seen = set()

    for value in values:
        value = safe_str(value)

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result

def iso_datetime(value):
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.isoformat()

    return str(value)

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
            "MONGO_URI environment variable is not set."
        )

    logger.info("Connecting to MongoDB")

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=10000,
    )

    client.admin.command("ping")

    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    logger.info(
        "MongoDB connected: db=%s collection=%s",
        MONGO_DB,
        MONGO_COLLECTION,
    )

    return client, collection

def load_categories(pg):
    """
    Load Article categories once.

    base_article.category_id -> base_category.id
    """

    logger.info("Loading categories...")

    sql = """
        SELECT
            id,
            db_id,
            name
        FROM base_category
    """

    categories = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)

        for row in cur.fetchall():
            categories[row["id"]] = {
                "id": row["id"],
                "db_id": row["db_id"],
                "name": safe_str(row["name"]),
            }

    logger.info(
        "Loaded %d categories",
        len(categories),
    )

    return categories

def load_company_names(pg):
    """
    Canonical company display names.

    companies.useraccount_ptr_id -> user_accounts.id
    """

    logger.info("Loading company names...")

    sql = """
        SELECT
            c.useraccount_ptr_id AS user_id,
            c.name AS company_name
        FROM companies c
    """

    companies = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)

        for row in cur.fetchall():
            companies[row["user_id"]] = safe_str(
                row["company_name"]
            )

    logger.info(
        "Loaded %d company names",
        len(companies),
    )

    return companies

def load_article_locations(pg, article_ids):
    """
    Load Article -> Country relationships.

    base_article_location:
        article_id
        country_id

    countries:
        id
        db_id
        name
        code
    """

    if not article_ids:
        return {}

    sql = """
        SELECT
            al.article_id,
            c.id AS country_id,
            c.db_id AS country_db_id,
            c.name AS country_name,
            c.code AS country_code,
            c.country_code AS country_code_alt
        FROM base_article_location al
        JOIN countries c
            ON c.id = al.country_id
        WHERE al.article_id = ANY(%s)
        ORDER BY al.article_id, c.name
    """

    locations = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (article_ids,))

        for row in cur.fetchall():

            article_id = row["article_id"]

            locations.setdefault(article_id, [])

            code = safe_str(
                row["country_code"]
            )

            if not code:
                code = safe_str(
                    row["country_code_alt"]
                )

            locations[article_id].append({
                "id": row["country_id"],
                "db_id": row["country_db_id"],
                "name": safe_str(row["country_name"]),
                "code": code,
            })

    return locations

def fetch_articles(pg, last_id=0, limit=None):
    """
    Fetch Articles incrementally.

    We deliberately migrate every Article regardless of status.
    is_live is determined from status.
    """

    sql = """
        SELECT
            a.id,
            a.my_sql_article_id,
            a.company_id,
            a.category_id,

            a.title,
            a.sub_title,

            a.image,
            a.video,
            a.thumbnail_url,
            a.youtube_link,

            a.content,

            a.latitude,
            a.longitude,

            a.views,
            a.impressions,
            a.share_count,

            a.created_at,

            a.slug,

            a."isFeatured" AS is_featured,

            a.status,
            a.is_auto_renew_enabled

        FROM base_article a
        WHERE a.id > %s
        ORDER BY a.id
    """

    params = [last_id]

    if limit:
        sql += " LIMIT %s"
        params.append(limit)

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return cur.fetchall()

def load_authors(pg, user_ids):
    """
    Load only public author fields.

    IMPORTANT:
    We deliberately do not retrieve:
      password
      email
      phone
      IP
      wallet
      credits
      security/authentication fields
    """

    if not user_ids:
        return {}

    sql = """
        SELECT
            u.id,
            u.first_name,
            u.last_name,
            u.username,
            u.user_type,
            u.verified,
            u.is_pro,
            u.is_active,
            u.avatar,
            u.cover,
            u.about_us,
            u.city_town,
            u.current_country_id,
            u.slug,
            u.no_of_employees,
            u.tagline,
            u.is_featured,
            u.is_working
        FROM user_accounts u
        WHERE u.id = ANY(%s)
    """

    authors = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (user_ids,))

        for row in cur.fetchall():
            authors[row["id"]] = row

    return authors

def build_author(
    author_row,
    company_names,
):
    if not author_row:
        return None

    user_id = author_row["id"]
    user_type = safe_str(
        author_row["user_type"]
    ).lower()

    username = safe_str(
        author_row["username"]
    )

    slug = safe_str(
        author_row["slug"]
    )

    if user_type == "company":

        company_name = safe_str(
            company_names.get(user_id)
        )

        name = (
            company_name
            or username
            or build_person_name(
                author_row["first_name"],
                author_row["last_name"],
                username,
            )
        )

    elif user_type == "professional":

        name = build_person_name(
            author_row["first_name"],
            author_row["last_name"],
            username,
        )

    else:

        name = build_person_name(
            author_row["first_name"],
            author_row["last_name"],
            username,
        )

    about_us = clean_html_to_text(
        author_row["about_us"]
    )

    return {
        "id": user_id,
        "type": user_type or None,
        "name": name or None,
        "username": username or None,
        "slug": slug or None,

        "verified": bool(
            author_row["verified"]
        ),
        "is_pro": bool(
            author_row["is_pro"]
        ),

        "is_active": bool(
            author_row["is_active"]
        ),

        "avatar": safe_str(
            author_row["avatar"]
        ) or None,

        "cover": safe_str(
            author_row["cover"]
        ) or None,

        "about_us": limit_text(
            about_us,
            10000,
        ) or None,

        "city_town": safe_str(
            author_row["city_town"]
        ) or None,

        "current_country_id": author_row[
            "current_country_id"
        ],

        "no_of_employees": author_row[
            "no_of_employees"
        ],

        "tagline": safe_str(
            author_row["tagline"]
        ) or None,

        "is_featured": bool(
            author_row["is_featured"]
        ),

        "is_working": bool(
            author_row["is_working"]
        ),
    }

def build_search_keywords(
    article,
    category,
    author,
    countries,
):
    values = [
        article["title"],
        article["sub_title"],
        article["slug"],
    ]

    if category:
        values.append(category["name"])

    if author:
        values.extend([
            author.get("name"),
            author.get("username"),
            author.get("slug"),
            author.get("city_town"),
            author.get("tagline"),
        ])

    for country in countries:
        values.extend([
            country.get("name"),
            country.get("code"),
        ])

    return unique_strings(values)

def build_ai_search_text(
    article,
    category,
    author,
    countries,
    content_text,
):
    """
    Build one consolidated searchable representation.

    This is intentionally text-only.
    """

    sections = []

    title = safe_str(article["title"])

    if title:
        sections.append(
            f"Title: {title}"
        )

    subtitle = safe_str(
        article["sub_title"]
    )

    if subtitle:
        sections.append(
            f"Subtitle: {subtitle}"
        )

    if category:
        category_name = safe_str(
            category.get("name")
        )

        if category_name:
            sections.append(
                f"Category: {category_name}"
            )

    if author:

        author_name = safe_str(
            author.get("name")
        )

        author_type = safe_str(
            author.get("type")
        )

        if author_name:
            sections.append(
                f"Author: {author_name}"
            )

        if author_type:
            sections.append(
                f"Author Type: {author_type}"
            )

        username = safe_str(
            author.get("username")
        )

        if username:
            sections.append(
                f"Author Username: {username}"
            )

        city = safe_str(
            author.get("city_town")
        )

        if city:
            sections.append(
                f"Author Location: {city}"
            )

        tagline = safe_str(
            author.get("tagline")
        )

        if tagline:
            sections.append(
                f"Author Tagline: {tagline}"
            )

    country_names = unique_strings([
        c.get("name")
        for c in countries
    ])

    if country_names:
        sections.append(
            "Countries: "
            + ", ".join(country_names)
        )

    if article["latitude"] is not None:
        sections.append(
            f"Latitude: {article['latitude']}"
        )

    if article["longitude"] is not None:
        sections.append(
            f"Longitude: {article['longitude']}"
        )

    status = safe_str(
        article["status"]
    )

    if status:
        sections.append(
            f"Status: {status}"
        )

    content_text = safe_str(
        content_text
    )

    if content_text:
        sections.append(
            "Article Content:\n"
            + content_text
        )

    result = "\n\n".join(sections)

    return limit_text(
        result,
        MAX_AI_SEARCH_TEXT_CHARS,
    )

def build_article_document(
    article,
    category,
    author,
    countries,
):
    
    content_text = clean_html_to_text(
        article["content"]
    )

    content_text = limit_text(
        content_text,
        MAX_CLEAN_CONTENT_CHARS,
    )

    keywords = build_search_keywords(
        article,
        category,
        author,
        countries,
    )

    ai_search_text = build_ai_search_text(
        article,
        category,
        author,
        countries,
        content_text,
    )

    status = safe_str(
        article["status"]
    ).lower()

    is_live = status == "publish"

    document = {
        "_id": f"article:{article['id']}",

        "schema_version": 1,
        "entity_type": "article",

        "source": {
            "model": "Article",
            "table": "base_article",
            "object_id": article["id"],
            "my_sql_article_id": article[
                "my_sql_article_id"
            ],
        },

        "title": safe_str(
            article["title"]
        ),

        "sub_title": safe_str(
            article["sub_title"]
        ) or None,

        "slug": safe_str(
            article["slug"]
        ) or None,

        "category": category,

        "author": author,

        "location": {
            "countries": countries,
            "latitude": article["latitude"],
            "longitude": article["longitude"],
        },

        "media": {
            "image": safe_str(
                article["image"]
            ) or None,

            "video": safe_str(
                article["video"]
            ) or None,

            "thumbnail_url": safe_str(
                article["thumbnail_url"]
            ) or None,

            "youtube_link": safe_str(
                article["youtube_link"]
            ) or None,
        },

        "content": {
            "text": content_text or None,
        },

        "metadata": {
            "status": status or None,

            "views": article["views"],
            "impressions": article["impressions"],
            "share_count": article["share_count"],

            "is_featured": bool(
                article["is_featured"]
            ),

            "is_auto_renew_enabled": bool(
                article["is_auto_renew_enabled"]
            ),

            "created_at": iso_datetime(
                article["created_at"]
            ),
        },

        "search_keywords": keywords,
        "search_aliases": [],

        "ai_search_text": ai_search_text,

        "embedding": {
            "model": None,
            "dimensions": 384,
            "vector": None,
            "status": "pending",
        },

        "is_live": is_live,

        "migration": {
            "source": "postgresql",
            "migrated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    }

    return document

def write_batch(
    collection,
    documents,
):
    if not documents:
        return 0

    operations = []

    for document in documents:

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

    try:

        result = collection.bulk_write(
            operations,
            ordered=False,
        )

        return (
            result.upserted_count
            + result.modified_count
        )

    except BulkWriteError as exc:

        details = exc.details or {}

        write_errors = details.get(
            "writeErrors",
            [],
        )

        logger.error(
            "MongoDB bulk write failed: %d errors",
            len(write_errors),
        )

        for error in write_errors[:10]:
            logger.error(
                "Mongo error: %s",
                error,
            )

        raise

def migrate(
    limit=None,
    start_id=0,
    batch_size=BATCH_SIZE,
):
    pg = None
    mongo_client = None

    try:

        pg = get_postgres_connection()

        mongo_client, collection = (
            get_mongo_collection()
        )

        categories = load_categories(
            pg
        )

        company_names = load_company_names(
            pg
        )

        logger.info(
            "Starting Article migration..."
        )

        logger.info(
            "start_id=%s limit=%s batch_size=%s",
            start_id,
            limit,
            batch_size,
        )

        total_processed = 0
        total_written = 0
        total_errors = 0

        last_id = start_id
        remaining = limit

        while True:

            fetch_limit = batch_size

            if remaining is not None:
                fetch_limit = min(
                    fetch_limit,
                    remaining,
                )

            articles = fetch_articles(
                pg,
                last_id=last_id,
                limit=fetch_limit,
            )

            if not articles:
                break

            article_ids = [
                row["id"]
                for row in articles
            ]

            user_ids = list({
                row["company_id"]
                for row in articles
                if row["company_id"] is not None
            })

            authors = load_authors(
                pg,
                user_ids,
            )

            locations = load_article_locations(
                pg,
                article_ids,
            )

            documents = []

            for article in articles:

                try:

                    category = categories.get(
                        article["category_id"]
                    )

                    author_row = authors.get(
                        article["company_id"]
                    )

                    author = build_author(
                        author_row,
                        company_names,
                    )

                    countries = locations.get(
                        article["id"],
                        [],
                    )

                    document = (
                        build_article_document(
                            article,
                            category,
                            author,
                            countries,
                        )
                    )

                    documents.append(
                        document
                    )

                except Exception as exc:

                    total_errors += 1

                    logger.exception(
                        "Failed to build Article %s: %s",
                        article["id"],
                        exc,
                    )

            if documents:

                written = write_batch(
                    collection,
                    documents,
                )

                total_written += written

            processed_count = len(
                articles
            )

            total_processed += (
                processed_count
            )

            last_id = articles[-1]["id"]

            if remaining is not None:
                remaining -= processed_count

                if remaining <= 0:
                    break

            logger.info(
                "Processed=%d | last_id=%s | written=%d | errors=%d",
                total_processed,
                last_id,
                total_written,
                total_errors,
            )

        logger.info(
            "========================================"
        )

        logger.info(
            "ARTICLE MIGRATION COMPLETE"
        )

        logger.info(
            "Processed : %d",
            total_processed,
        )

        logger.info(
            "Written   : %d",
            total_written,
        )

        logger.info(
            "Errors    : %d",
            total_errors,
        )

        logger.info(
            "Last ID   : %s",
            last_id,
        )

        logger.info(
            "========================================"
        )

    finally:

        if pg:
            pg.close()

        if mongo_client:
            mongo_client.close()

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Migrate PostgreSQL Articles "
            "to MongoDB search_documents."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of Articles to migrate.",
    )

    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Only migrate Articles with id greater than this.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of Articles per MongoDB bulk batch.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error(
            "--limit must be greater than 0"
        )

    if args.start_id < 0:
        parser.error(
            "--start-id cannot be negative"
        )

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be greater than 0"
        )

    migrate(
        limit=args.limit,
        start_id=args.start_id,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()