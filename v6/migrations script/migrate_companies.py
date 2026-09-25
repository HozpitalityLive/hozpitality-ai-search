# copy from dev to opt
# sudo cp ~/hozpitality-mongo-migration/migrate_companies.py /opt/hozpitality-mongo-migration/migrate_companies.py

# sudo -u postgres env MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_companies.py \
# --limit=5000


import os
import sys
import time
import re
import html
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient, UpdateOne

PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "hozpitality")
PG_USER = os.getenv("PG_USER", "postgres")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "mongoAdmin")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "search_documents")

CLOUDFRONT_BASE = os.getenv(
    "CLOUDFRONT_BASE",
    "https://d2he8nskrbhxwq.cloudfront.net",
).rstrip("/")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return ""

    text = str(value)

    text = html.unescape(text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def html_to_text(value: Any) -> str:
    return clean_text(value)

def media_url(value: Any) -> str | None:
    if not value:
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.startswith("http://") or value.startswith("https://"):
        return value

    return f"{CLOUDFRONT_BASE}/{value.lstrip('/')}"

def dedupe_strings(values):
    result = []
    seen = set()

    for value in values:
        value = clean_text(value)

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result

def normalize_relation(value):
    if not value:
        return []

    if not isinstance(value, list):
        return []

    result = []

    for item in value:
        if not isinstance(item, dict):
            continue

        result.append(item)

    return result

def build_name(first_name, last_name, fallback=""):
    first = clean_text(first_name)
    last = clean_text(last_name)

    name = f"{first} {last}".strip()

    return name or clean_text(fallback)

def object_ref(obj_id, name=None, **extra):
    if obj_id is None:
        return None

    data = {
        "id": obj_id,
    }

    if name:
        data["name"] = clean_text(name)

    for key, value in extra.items():
        if value is not None:
            data[key] = value

    return data

def add_text(parts, label, value):
    value = clean_text(value)

    if value:
        parts.append(f"{label}: {value}.")

def add_list_text(parts, label, values):
    cleaned = dedupe_strings(values)

    if cleaned:
        parts.append(f"{label}: {', '.join(cleaned)}.")

COMPANY_QUERY = """
SELECT
    c.useraccount_ptr_id AS company_id,

    c.name AS company_name,
    c.created_by,
    c.current_designation,
    c.website_link,

    u.first_name,
    u.last_name,
    u.username,
    u.email,

    u.user_type,
    u.slug,

    u.city_town,

    u.current_country_id,
    cc.name AS current_country_name,
    cc.db_id AS current_country_db_id,
    cc.country_code AS current_country_code,
    cc.ac_name AS current_country_ac_name,
    cc.code AS current_country_code_short,

    u.nationality_id,
    nc.name AS nationality_name,
    nc.db_id AS nationality_db_id,
    nc.country_code AS nationality_country_code,
    nc.ac_name AS nationality_ac_name,
    nc.code AS nationality_code,

    u.avatar,
    u.cover,

    u.about_us,
    u.tagline,

    u.no_of_employees,

    u.verified,
    u.is_pro,
    u.is_active,
    u.is_featured,

    u.created_at,

    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'id', i.id,
                    'db_id', i.db_id,
                    'name', i.name,
                    'context', i.context
                )
                ORDER BY i.name
            )
            FROM user_accounts_industry ui
            JOIN industries i
                ON i.id = ui.industry_id
            WHERE ui.useraccount_id = u.id
        ),
        '[]'::jsonb
    ) AS industries,

    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'id', sc.id,
                    'db_id', sc.db_id,
                    'name', sc.name
                )
                ORDER BY sc.name
            )
            FROM user_accounts_supplier_category usc
            JOIN supplier_category sc
                ON sc.id = usc.suppliercategory_id
            WHERE usc.useraccount_id = u.id
        ),
        '[]'::jsonb
    ) AS supplier_categories

FROM companies c

JOIN user_accounts u
    ON u.id = c.useraccount_ptr_id

LEFT JOIN countries cc
    ON cc.id = u.current_country_id

LEFT JOIN countries nc
    ON nc.id = u.nationality_id

WHERE u.is_active = true
  AND u.user_type = 'company'

ORDER BY c.useraccount_ptr_id
"""

def build_document(row):
    company_id = row["company_id"]

    company_name = clean_text(row["company_name"])

    if not company_name:
        company_name = build_name(
            row.get("first_name"),
            row.get("last_name"),
            row.get("username") or f"Company {company_id}",
        )

    industries = normalize_relation(row.get("industries"))
    supplier_categories = normalize_relation(
        row.get("supplier_categories")
    )

    supplier_industries = [
        item
        for item in industries
        if clean_text(item.get("context")).lower() == "supplier"
    ]

    is_supplier = bool(
        supplier_industries or supplier_categories
    )

    normalized_industries = []

    for item in industries:
        industry = {
            "id": item.get("id"),
            "name": clean_text(item.get("name")),
            "context": clean_text(item.get("context")),
        }

        if item.get("db_id") is not None:
            industry["db_id"] = item["db_id"]

        normalized_industries.append(industry)

    normalized_supplier_categories = []

    for item in supplier_categories:
        category = {
            "id": item.get("id"),
            "name": clean_text(item.get("name")),
        }

        if item.get("db_id") is not None:
            category["db_id"] = item["db_id"]

        normalized_supplier_categories.append(category)

    country = None

    if row.get("current_country_id") is not None:
        country = {
            "id": row["current_country_id"],
            "name": clean_text(row.get("current_country_name")),
            "db_id": row.get("current_country_db_id"),
            "country_code": clean_text(
                row.get("current_country_code")
            ),
            "ac_name": clean_text(
                row.get("current_country_ac_name")
            ),
            "code": clean_text(
                row.get("current_country_code_short")
            ),
        }

    if country:
        country = {
            k: v
            for k, v in country.items()
            if v not in ("", None)
        }

    nationality = None

    if row.get("nationality_id") is not None:
        nationality = {
            "id": row["nationality_id"],
            "name": clean_text(row.get("nationality_name")),
            "db_id": row.get("nationality_db_id"),
            "country_code": clean_text(
                row.get("nationality_country_code")
            ),
            "ac_name": clean_text(
                row.get("nationality_ac_name")
            ),
            "code": clean_text(
                row.get("nationality_code")
            ),
        }

        nationality = {
            k: v
            for k, v in nationality.items()
            if v not in ("", None)
        }

    user_name = company_name

    user = {
        "id": company_id,
        "name": user_name,
        "first_name": clean_text(row.get("first_name")),
        "last_name": clean_text(row.get("last_name")),
        "username": clean_text(row.get("username")),
        "slug": clean_text(row.get("slug")),
        "user_type": clean_text(row.get("user_type")),
        "profile_image": media_url(row.get("avatar")),
        "cover_image": media_url(row.get("cover")),
    }

    user = {
        k: v
        for k, v in user.items()
        if v not in ("", None)
    }

    company = {
        "id": company_id,
        "name": company_name,

        "created_by": clean_text(row.get("created_by")),

        "current_designation": clean_text(
            row.get("current_designation")
        ),

        "website": clean_text(row.get("website_link")),

        "is_supplier": is_supplier,

        "industries": normalized_industries,

        "supplier_categories": normalized_supplier_categories,
    }

    company = {
        k: v
        for k, v in company.items()
        if v not in ("", None)
    }

    metadata = {
        "verified": bool(row.get("verified")),
        "is_pro": bool(row.get("is_pro")),
        "is_featured": bool(row.get("is_featured")),
        "is_active": bool(row.get("is_active")),
    }

    if row.get("no_of_employees"):
        metadata["no_of_employees"] = clean_text(
            row["no_of_employees"]
        )

    search_parts = []

    add_text(
        search_parts,
        "Company",
        company_name,
    )

    search_parts.append(
        "Supplier company."
        if is_supplier
        else "Hospitality company."
    )

    add_text(
        search_parts,
        "Current designation",
        row.get("current_designation"),
    )

    add_text(
        search_parts,
        "Company description",
        row.get("about_us"),
    )

    add_text(
        search_parts,
        "Tagline",
        row.get("tagline"),
    )

    add_text(
        search_parts,
        "Company size",
        row.get("no_of_employees"),
    )

    industry_names = [
        item.get("name")
        for item in normalized_industries
    ]

    add_list_text(
        search_parts,
        "Industries",
        industry_names,
    )

    supplier_industry_names = [
        item.get("name")
        for item in normalized_industries
        if item.get("context") == "supplier"
    ]

    add_list_text(
        search_parts,
        "Supplier industries",
        supplier_industry_names,
    )

    category_names = [
        item.get("name")
        for item in normalized_supplier_categories
    ]

    add_list_text(
        search_parts,
        "Supplier categories",
        category_names,
    )

    add_text(
        search_parts,
        "City",
        row.get("city_town"),
    )

    if country:
        add_text(
            search_parts,
            "Country",
            country.get("name"),
        )

    if nationality:
        add_text(
            search_parts,
            "Nationality",
            nationality.get("name"),
        )

    if row.get("verified"):
        search_parts.append("Verified company.")

    if row.get("is_pro"):
        search_parts.append("Pro company.")

    search_keywords = dedupe_strings(
        [
            company_name,
            row.get("current_designation"),
            row.get("city_town"),
            country.get("name") if country else None,
            nationality.get("name") if nationality else None,
            *industry_names,
            *supplier_industry_names,
            *category_names,
        ]
    )

    search_aliases = []

    if is_supplier:
        search_aliases.extend(
            [
                "supplier",
                "hospitality supplier",
                "hospitality vendor",
                "hospitality solutions",
                "supplier company",
            ]
        )

        search_aliases.extend(category_names)
        search_aliases.extend(supplier_industry_names)

    search_aliases = dedupe_strings(search_aliases)

    ai_search_text = "\n\n".join(search_parts)

    document = {
        "_id": f"company:{company_id}",

        "schema_version": 1,

        "entity_type": "company",

        "source": {
            "model": "company",
            "app_label": "base",
            "object_id": company_id,
        },

        "title": company_name,

        "slug": clean_text(row.get("slug")),

        "profile_image": media_url(row.get("avatar")),

        "cover_image": media_url(row.get("cover")),

        "summary": html_to_text(row.get("about_us")),

        "user": user,

        "company": company,

        "location": {
            "city": clean_text(row.get("city_town")),
            "country": country,
            "nationality": nationality,
        },

        "metadata": metadata,

        "search_keywords": search_keywords,

        "search_aliases": search_aliases,

        "ai_search_text": ai_search_text,

        "is_live": True,

        "created_at": row.get("created_at"),

    }

    return document

def migrate(limit=None):
    if not MONGO_URI:
        raise RuntimeError(
            "MONGO_URI environment variable is required"
        )

    print("=" * 70)
    print("HOZPITALITY COMPANY MIGRATION")
    print("=" * 70)

    print(f"PostgreSQL database : {PG_DB}")
    print(f"MongoDB database    : {MONGO_DB}")
    print(f"Collection          : {MONGO_COLLECTION}")
    print(f"Limit               : {limit or 'ALL'}")
    print(f"Batch size          : {BATCH_SIZE}")
    print("=" * 70)

    pg_conn = psycopg2.connect(
        dbname=PG_DB,
        user=PG_USER,
    )

    pg_conn.set_session(
        readonly=True,
        autocommit=False,
    )

    mongo_client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
    )

    mongo_client.admin.command("ping")

    mongo_db = mongo_client[MONGO_DB]
    collection = mongo_db[MONGO_COLLECTION]

    print("MongoDB connection: OK")
    print("PostgreSQL connection: OK")

    query = COMPANY_QUERY

    if limit:
        query += "\nLIMIT %s"
        query_params = (limit,)
    else:
        query_params = ()

    total = 0
    started = time.time()

    operations = []

    try:
        with pg_conn.cursor(
            name="company_migration_cursor",
            cursor_factory=RealDictCursor,
        ) as cursor:

            cursor.itersize = BATCH_SIZE

            print("Executing PostgreSQL query...")

            cursor.execute(
                query,
                query_params,
            )

            while True:
                rows = cursor.fetchmany(BATCH_SIZE)

                if not rows:
                    break

                for row in rows:

                    document = build_document(row)

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

                    if len(operations) >= BATCH_SIZE:

                        result = collection.bulk_write(
                            operations,
                            ordered=False,
                        )

                        total += len(operations)

                        elapsed = time.time() - started

                        print(
                            f"Migrated: {total:,} | "
                            f"matched={result.matched_count:,} | "
                            f"modified={result.modified_count:,} | "
                            f"upserted={result.upserted_count:,} | "
                            f"{total / elapsed:.1f} docs/sec"
                        )

                        operations = []

        if operations:

            result = collection.bulk_write(
                operations,
                ordered=False,
            )

            total += len(operations)

            print(
                f"Migrated final batch: {len(operations):,} | "
                f"matched={result.matched_count:,} | "
                f"modified={result.modified_count:,} | "
                f"upserted={result.upserted_count:,}"
            )

        elapsed = time.time() - started

        print()
        print("=" * 70)
        print("MIGRATION COMPLETE")
        print("=" * 70)
        print(f"Processed : {total:,}")
        print(f"Time      : {elapsed:.2f}s")
        print(
            f"Speed     : "
            f"{total / elapsed:.1f} docs/sec"
            if elapsed
            else "Speed     : N/A"
        )
        print("=" * 70)

    finally:
        pg_conn.close()
        mongo_client.close()

if __name__ == "__main__":

    limit = None

    for arg in sys.argv[1:]:

        if arg.startswith("--limit="):
            value = arg.split("=", 1)[1]

            if value.lower() == "none":
                limit = None
            else:
                limit = int(value)

    migrate(limit=limit)