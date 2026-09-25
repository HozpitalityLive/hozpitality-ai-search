# copy from dev to opt
# sudo cp ~/hozpitality-mongo-migration/migrate_events.py /opt/hozpitality-mongo-migration/migrate_events.py

# sudo -u postgres env MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_events.py \
# --limit=100

# sudo -u postgres env \
# MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_events.py



import os
import re
import html
import argparse
import logging
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient, UpdateOne

POSTGRES_DB = "hozpitality"
POSTGRES_USER = "postgres"

MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = "mongoAdmin"
MONGO_COLLECTION = "search_documents"

BATCH_SIZE = 100

MEDIA_BASE_URL = os.getenv(
    "CLOUDFRONT_BASE",
    "https://d2he8nskrbhxwq.cloudfront.net",
).rstrip("/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("event-migration")

def clean_text(value):
    """
    Convert HTML/HTML entities into readable plain text.
    """
    if value is None:
        return ""

    value = str(value)

    for _ in range(3):
        decoded = html.unescape(value)
        if decoded == value:
            break
        value = decoded

    value = re.sub(
        r"<(script|style).*?>.*?</\1>",
        " ",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )

    value = re.sub(
        r"</(p|div|br|li|h[1-6]|tr)>",
        "\n",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(r"<[^>]+>", " ", value)

    value = html.unescape(value)

    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n\s*\n+", "\n", value)

    return value.strip()

def normalize(value):
    if value is None:
        return None

    value = str(value).strip()

    return value if value else None

def build_media_url(path):
    """
    Keep source-relative media paths if MEDIA_BASE_URL is not configured.
    If MEDIA_BASE_URL is configured, convert them to absolute URLs.
    """
    path = normalize(path)

    if not path:
        return None

    if path.startswith("http://") or path.startswith("https://"):
        return path

    if MEDIA_BASE_URL:
        return f"{MEDIA_BASE_URL}/{path.lstrip('/')}"

    return path

def serialize_datetime(value):
    if value is None:
        return None

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value

def is_event_live(status, end_datetime):
    """
    Event is searchable/live only when its lifecycle says it should
    be active AND it has not already ended.
    """
    if status not in {"Upcoming", "Ongoing"}:
        return False

    if end_datetime is None:
        return False

    now = datetime.now(timezone.utc)

    if end_datetime.tzinfo is None:
        end_datetime = end_datetime.replace(tzinfo=timezone.utc)

    return end_datetime >= now

def unique_strings(values):
    result = []
    seen = set()

    for value in values:
        value = normalize(value)

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result

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

    logger.info("Connecting to MongoDB")

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=10000,
    )

    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    client.admin.command("ping")

    logger.info(
        "MongoDB connected: db=%s collection=%s",
        MONGO_DB,
        MONGO_COLLECTION,
    )

    return client, collection

def load_countries(pg):
    logger.info("Loading countries...")

    result = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id,
                db_id,
                name,
                ac_name,
                country_code,
                code
            FROM countries
            """
        )

        for row in cur.fetchall():
            result[row["id"]] = dict(row)

    logger.info("Loaded %s countries", len(result))

    return result

def load_package_types(pg):
    logger.info("Loading package types...")

    result = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id,
                name,
                description,
                "is_CC",
                "is_PAYG",
                "is_POP",
                "is_PREMIUM",
                "is_SP",
                "is_GP",
                "is_EP"
            FROM base_packagetype
            """
        )

        for row in cur.fetchall():
            result[row["id"]] = dict(row)

    logger.info("Loaded %s package types", len(result))

    return result

def load_companies(pg):
    logger.info("Loading company information...")

    result = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                c.useraccount_ptr_id AS company_id,
                c.name AS company_name,
                c.created_by,
                c.current_designation,
                c.website_link,

                u.username,
                u.first_name,
                u.last_name,
                u.slug,
                u.user_type,
                u.city_town,
                u.current_country_id,
                u.nationality_id,
                u.avatar,
                u.cover,
                u.about_us,
                u.verified,
                u.is_pro,
                u.is_active,
                u.is_working,
                u.no_of_employees,
                u.tagline,
                u.is_featured,
                u.created_at

            FROM companies c
            JOIN user_accounts u
              ON u.id = c.useraccount_ptr_id
            """
        )

        for row in cur.fetchall():
            result[row["company_id"]] = dict(row)

    logger.info("Loaded %s companies", len(result))

    return result

def load_event_images(pg, event_ids):
    if not event_ids:
        return {}

    result = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id,
                event_id,
                image
            FROM base_eventimage
            WHERE event_id = ANY(%s)
            ORDER BY event_id, id
            """,
            (event_ids,),
        )

        for row in cur.fetchall():
            event_id = row["event_id"]

            result.setdefault(event_id, []).append(
                {
                    "id": row["id"],
                    "url": build_media_url(row["image"]),
                    "path": normalize(row["image"]),
                }
            )

    return result

def load_interested_counts(pg, event_ids):
    if not event_ids:
        return {}, {}

    normal_counts = {}
    unlocked_counts = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:

        cur.execute(
            """
            SELECT
                event_id,
                COUNT(*) AS interested_count
            FROM base_event_interested
            WHERE event_id = ANY(%s)
            GROUP BY event_id
            """,
            (event_ids,),
        )

        for row in cur.fetchall():
            normal_counts[row["event_id"]] = int(
                row["interested_count"]
            )

        cur.execute(
            """
            SELECT
                event_id,
                COUNT(*) AS interested_user_count,
                COUNT(*) FILTER (
                    WHERE is_unlocked = true
                ) AS unlocked_interested_user_count
            FROM base_eventinteresteduser
            WHERE event_id = ANY(%s)
            GROUP BY event_id
            """,
            (event_ids,),
        )

        for row in cur.fetchall():
            unlocked_counts[row["event_id"]] = {
                "total": int(row["interested_user_count"]),
                "unlocked": int(
                    row["unlocked_interested_user_count"]
                ),
            }

    return normal_counts, unlocked_counts

def build_event_search_text(
    event,
    company,
    country,
    package,
):
    parts = []

    title = clean_text(event.get("title"))
    details = clean_text(event.get("details"))

    if title:
        parts.append(f"Event: {title}")

    if company:
        company_name = normalize(company.get("company_name"))

        if company_name:
            parts.append(f"Company: {company_name}")

        tagline = clean_text(company.get("tagline"))

        if tagline:
            parts.append(f"Company tagline: {tagline}")

        about = clean_text(company.get("about_us"))

        if about:
            parts.append(f"Company description: {about}")

        city = normalize(company.get("city_town"))

        if city:
            parts.append(f"Company city: {city}")

    city = clean_text(event.get("city"))

    if city:
        parts.append(f"Event city: {city}")

    address = clean_text(event.get("address"))

    if address:
        parts.append(f"Address: {address}")

    if country:
        country_name = normalize(country.get("name"))

        if country_name:
            parts.append(f"Country: {country_name}")

    event_type = normalize(event.get("event_type"))

    if event_type:
        parts.append(
            f"Event type: {'Free' if event_type == 'free' else 'Paid'}"
        )

    status = normalize(event.get("status"))

    if status:
        parts.append(f"Status: {status}")

    start_datetime = event.get("start_datetime")
    end_datetime = event.get("end_datetime")

    if start_datetime:
        parts.append(
            f"Start date: {start_datetime.isoformat()}"
        )

    if end_datetime:
        parts.append(
            f"End date: {end_datetime.isoformat()}"
        )

    website = normalize(event.get("website"))

    if website:
        parts.append(f"Website: {website}")

    payment_link = normalize(event.get("payment_link"))

    if payment_link:
        parts.append(f"Payment link: {payment_link}")

    if package:
        package_name = normalize(package.get("name"))

        if package_name:
            parts.append(
                f"Package type: {package_name}"
            )

    flags = []

    if event.get("is_feature"):
        flags.append("featured")

    if event.get("is_premium"):
        flags.append("premium")

    if event.get("is_education"):
        flags.append("education")

    if flags:
        parts.append(
            "Event flags: " + ", ".join(flags)
        )

    if details:
        parts.append(f"Details: {details}")

    text = "\n".join(parts)

    return text[:750000]

def build_event_document(
    event,
    company,
    country,
    package,
    event_images,
    interested_count,
    interested_user_count,
    unlocked_interested_user_count,
):
    event_id = event["id"]

    title = clean_text(event.get("title"))
    details = clean_text(event.get("details"))

    status = normalize(event.get("status"))
    event_type = normalize(event.get("event_type"))

    start_datetime = serialize_datetime(
        event.get("start_datetime")
    )

    end_datetime = serialize_datetime(
        event.get("end_datetime")
    )

    live = is_event_live(
        status,
        end_datetime,
    )

    country_document = None

    if country:
        country_document = {
            "id": country.get("id"),
            "db_id": country.get("db_id"),
            "name": normalize(country.get("name")),
            "ac_name": normalize(country.get("ac_name")),
            "code": normalize(country.get("code")),
            "country_code": normalize(
                country.get("country_code")
            ),
        }

    company_document = None

    if company:
        company_document = {
            "id": company.get("company_id"),
            "name": normalize(company.get("company_name")),
            "username": normalize(company.get("username")),
            "first_name": normalize(company.get("first_name")),
            "last_name": normalize(company.get("last_name")),
            "slug": normalize(company.get("slug")),
            "user_type": normalize(company.get("user_type")),
            "website": normalize(company.get("website_link")),
            "created_by": normalize(company.get("created_by")),
            "current_designation": normalize(
                company.get("current_designation")
            ),
            "city": normalize(company.get("city_town")),
            "avatar": build_media_url(
                company.get("avatar")
            ),
            "cover": build_media_url(
                company.get("cover")
            ),
            "about_us": clean_text(
                company.get("about_us")
            ),
            "verified": bool(company.get("verified")),
            "is_pro": bool(company.get("is_pro")),
            "is_active": bool(company.get("is_active")),
            "is_working": bool(company.get("is_working")),
            "no_of_employees": company.get(
                "no_of_employees"
            ),
            "tagline": normalize(company.get("tagline")),
            "is_featured": bool(
                company.get("is_featured")
            ),
        }

    package_document = None

    if package:
        package_document = {
            "id": package.get("id"),
            "name": normalize(package.get("name")),
            "description": clean_text(
                package.get("description")
            ),
            "is_payg": bool(package.get("is_PAYG")),
            "is_pop": bool(package.get("is_POP")),
            "is_cc": bool(package.get("is_CC")),
            "is_premium": bool(
                package.get("is_PREMIUM")
            ),
            "is_sp": bool(package.get("is_SP")),
            "is_gp": bool(package.get("is_GP")),
            "is_ep": bool(package.get("is_EP")),
        }

    image_documents = []

    for image in event_images or []:
        image_documents.append(
            {
                "id": image["id"],
                "path": image["path"],
                "url": image["url"],
            }
        )

    ai_search_text = build_event_search_text(
        event=event,
        company=company,
        country=country,
        package=package,
    )

    keywords = unique_strings(
        [
            title,
            company.get("company_name") if company else None,
            event.get("city"),
            country.get("name") if country else None,
            event_type,
            status,
            package.get("name") if package else None,
        ]
    )

    aliases = unique_strings(
        [
            title,
            company.get("company_name") if company else None,
            event.get("slug"),
        ]
    )

    document = {
        "_id": f"event:{event_id}",

        "entity_type": "event",
        "schema_version": 1,

        "title": title,
        "slug": normalize(event.get("slug")),

        "description": details,

        "event_type": event_type,
        "status": status,

        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "created_at": serialize_datetime(
            event.get("created_at")
        ),

        "location": {
            "address": normalize(event.get("address")),
            "city": normalize(event.get("city")),
            "country": country_document,
            "latitude": event.get("latitude"),
            "longitude": event.get("longitude"),
        },

        "company": company_document,

        "package": package_document,

        "media": {
            "banner": build_media_url(
                event.get("banner")
            ),
            "images": image_documents,
        },

        "flags": {
            "is_feature": bool(
                event.get("is_feature")
            ),
            "is_premium": bool(
                event.get("is_premium")
            ),
            "is_education": bool(
                event.get("is_education")
            ),
        },

        "engagement": {
            "views": int(event.get("views") or 0),
            "interested_count": interested_count,
            "interested_user_count": interested_user_count,
            "unlocked_interested_user_count":
                unlocked_interested_user_count,
        },

        "website": normalize(event.get("website")),

        "payment_link": normalize(
            event.get("payment_link")
        ),

        "search_keywords": keywords,
        "search_aliases": aliases,

        "ai_search_text": ai_search_text,

        "embedding": {
            "model": None,
            "dimensions": 384,
            "status": "pending",
            "vector": None,
        },

        "source": {
            "model": "Event",
            "table": "base_event",
            "object_id": event_id,
            "my_sql_event_id": event.get(
                "my_sql_event_id"
            ),
        },

        "is_live": live,

        "migration": {
            "source": "postgresql",
            "migrated_at": datetime.now(timezone.utc),
        },
    }

    return document

def migrate_events(
    limit=None,
    start_id=0,
    batch_size=BATCH_SIZE,
):
    pg = get_postgres_connection()

    mongo_client = None

    try:
        mongo_client, collection = get_mongo_collection()

        countries = load_countries(pg)
        packages = load_package_types(pg)
        companies = load_companies(pg)

        processed = 0
        written = 0
        errors = 0
        last_id = start_id

        logger.info("Starting Event migration...")
        logger.info(
            "start_id=%s limit=%s batch_size=%s",
            start_id,
            limit,
            batch_size,
        )

        while True:
            query = """
                SELECT
                    e.id,
                    e.title,
                    e.start_datetime,
                    e.end_datetime,
                    e.address,
                    e.city,
                    e.event_type,
                    e.website,
                    e.details,
                    e.banner,
                    e.is_feature,
                    e.is_premium,
                    e.views,
                    e.created_at,
                    e.company_id,
                    e.country_id,
                    e.slug,
                    e.payment_link,
                    e.status,
                    e.latitude,
                    e.longitude,
                    e.my_sql_event_id,
                    e.is_education,
                    e.package_type_id

                FROM base_event e

                WHERE e.id > %s

                ORDER BY e.id

                LIMIT %s
            """

            fetch_limit = batch_size

            if limit is not None:
                remaining = limit - processed

                if remaining <= 0:
                    break

                fetch_limit = min(
                    batch_size,
                    remaining,
                )

            with pg.cursor(
                cursor_factory=RealDictCursor
            ) as cur:
                cur.execute(
                    query,
                    (
                        last_id,
                        fetch_limit,
                    ),
                )

                rows = cur.fetchall()

            if not rows:
                break

            event_ids = [
                row["id"]
                for row in rows
            ]

            event_images = load_event_images(
                pg,
                event_ids,
            )

            normal_counts, interested_user_counts = (
                load_interested_counts(
                    pg,
                    event_ids,
                )
            )

            operations = []

            for row in rows:
                try:
                    event_id = row["id"]

                    company = companies.get(
                        row["company_id"]
                    )

                    country = countries.get(
                        row["country_id"]
                    )

                    package = packages.get(
                        row["package_type_id"]
                    )

                    interested_count = normal_counts.get(
                        event_id,
                        0,
                    )

                    interested_data = (
                        interested_user_counts.get(
                            event_id,
                            {
                                "total": 0,
                                "unlocked": 0,
                            },
                        )
                    )

                    document = build_event_document(
                        event=row,
                        company=company,
                        country=country,
                        package=package,
                        event_images=event_images.get(
                            event_id,
                            [],
                        ),
                        interested_count=interested_count,
                        interested_user_count=interested_data[
                            "total"
                        ],
                        unlocked_interested_user_count=
                            interested_data[
                                "unlocked"
                            ],
                    )

                    operations.append(
                        UpdateOne(
                            {
                                "_id": f"event:{event_id}"
                            },
                            {
                                "$set": document
                            },
                            upsert=True,
                        )
                    )

                    processed += 1
                    last_id = event_id

                except Exception:
                    errors += 1

                    logger.exception(
                        "Failed to prepare event id=%s",
                        row.get("id"),
                    )

            if operations:
                result = collection.bulk_write(
                    operations,
                    ordered=False,
                )

                written += (
                    result.upserted_count
                    + result.modified_count
                )

            logger.info(
                "Progress: processed=%s written=%s errors=%s last_id=%s",
                processed,
                written,
                errors,
                last_id,
            )

        logger.info("EVENT MIGRATION COMPLETE")
        logger.info("Processed : %s", processed)
        logger.info("Written   : %s", written)
        logger.info("Errors    : %s", errors)
        logger.info("Last ID   : %s", last_id)

    finally:
        try:
            pg.close()
        except Exception:
            pass

        if mongo_client:
            mongo_client.close()

def main():
    parser = argparse.ArgumentParser(
        description="Migrate Hozpitality Events from PostgreSQL to MongoDB"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of events to migrate",
    )

    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Start after this PostgreSQL event ID",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="PostgreSQL/Mongo batch size",
    )

    args = parser.parse_args()

    migrate_events(
        limit=args.limit,
        start_id=args.start_id,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()