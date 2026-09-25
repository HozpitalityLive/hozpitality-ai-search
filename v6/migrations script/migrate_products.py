# copy from dev to opt
# sudo cp ~/hozpitality-mongo-migration/migrate_products.py /opt/hozpitality-mongo-migration/migrate_products.py

# sudo -u postgres env MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_products.py \
# --limit=1000

# sudo -u postgres env \
# MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_products.py



import os
import re
import sys
import html
import logging
import argparse
from datetime import datetime, date, timezone
from decimal import Decimal
from urllib.parse import unquote

import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient, UpdateOne

POSTGRES_DB = os.getenv("POSTGRES_DB", "hozpitality")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")

MONGO_URI = os.getenv("MONGO_URI")

MONGO_DB = os.getenv("MONGO_DB", "mongoAdmin")
MONGO_COLLECTION = os.getenv(
    "MONGO_COLLECTION",
    "search_documents",
)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

MEDIA_BASE_URL = os.getenv(
    "CLOUDFRONT_BASE",
    "https://d2he8nskrbhxwq.cloudfront.net",
).rstrip("/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("product-migration")

def clean_string(value):
    """
    Normalize a simple string.
    """
    if value is None:
        return None

    value = str(value)

    value = html.unescape(value)

    value = value.replace("\x00", "")

    value = re.sub(r"\s+", " ", value)

    value = value.strip()

    return value or None

def clean_html(value):
    """
    Convert HTML/custom markup into readable plain text.

    Product descriptions contain normal HTML as well as custom
    [a]...[/a] style markup.
    """
    if not value:
        return None

    text = str(value)

    text = html.unescape(text)

    try:
        text = unquote(text)
    except Exception:
        pass

    text = re.sub(
        r"(?i)<\s*(br|p|div|li|ul|ol|h[1-6]|tr|td|th)\b[^>]*>",
        " ",
        text,
    )

    text = re.sub(
        r"(?is)<script\b[^>]*>.*?</script>",
        " ",
        text,
    )

    text = re.sub(
        r"(?is)<style\b[^>]*>.*?</style>",
        " ",
        text,
    )

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\[/?a(?:=[^\]]+)?\]", " ", text, flags=re.I)

    text = re.sub(
        r"\[/?(?:b|strong|i|em|u|br|p|div|li|ul|ol)[^\]]*\]",
        " ",
        text,
        flags=re.I,
    )

    text = html.unescape(text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text or None

def clean_url(value):
    """
    Preserve URLs while removing surrounding whitespace.
    """
    value = clean_string(value)

    if not value:
        return None

    return value

def build_media_url(path):
    """
    Convert relative media path to absolute media URL when
    MEDIA_BASE_URL is configured.

    Example:

    upload/photos/products/a.jpg

    becomes:

    https://cdn.example.com/upload/photos/products/a.jpg
    """
    path = clean_string(path)

    if not path:
        return None

    if path.startswith(("http://", "https://")):
        return path

    if not MEDIA_BASE_URL:
        return path

    return f"{MEDIA_BASE_URL}/{path.lstrip('/')}"

def media_type_from_path(path):
    """
    Product gallery contains JPG/WEBP/PNG and PDF files.
    """
    if not path:
        return "unknown"

    lower = path.lower()

    if lower.endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".bmp",
            ".svg",
            ".avif",
        )
    ):
        return "image"

    if lower.endswith(".pdf"):
        return "document"

    if lower.endswith(
        (
            ".mp4",
            ".mov",
            ".avi",
            ".webm",
            ".mkv",
        )
    ):
        return "video"

    return "file"

def split_keywords(value):
    """
    Convert product keywords into a clean list.

    Supports comma, semicolon and newline separated values.
    """
    if not value:
        return []

    value = str(value)

    value = html.unescape(value)

    parts = re.split(r"[,;\n\r|]+", value)

    result = []

    seen = set()

    for part in parts:
        part = clean_string(part)

        if not part:
            continue

        key = part.lower()

        if key in seen:
            continue

        seen.add(key)

        result.append(part)

    return result

def unique_strings(values):
    """
    Preserve order while removing duplicate strings.
    """
    result = []

    seen = set()

    for value in values:
        if value is None:
            continue

        value = clean_string(value)

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)

        result.append(value)

    return result

def json_safe(value):
    """
    Convert PostgreSQL values to MongoDB-compatible values.

    PostgreSQL DATE values are converted to midnight UTC
    datetime values because BSON does not support Python
    datetime.date directly.
    """
    if value is None:
        return None

    if isinstance(value, Decimal):
        return float(value)

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

def calculate_is_live(status, expiry_date, today=None):
    """
    Product search lifecycle rule:

        status == live
        AND
        (expiry_date IS NULL OR expiry_date >= today)

    Raw status/date fields remain untouched in the Mongo document.
    """
    if today is None:
        today = date.today()

    status = clean_string(status)

    if status != "live":
        return False

    if expiry_date is None:
        return True

    return expiry_date >= today

def get_postgres_connection():
    """
    PostgreSQL uses local Unix socket + peer authentication.

    The migration is executed as OS user `postgres`, therefore
    no PostgreSQL password is required.
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
    Load all ProductCategory rows.
    """
    logger.info("Loading product categories...")

    sql = """
        SELECT
            id,
            db_id,
            name,
            image_field
        FROM marketplace_productcategory
        ORDER BY id
    """

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    categories = {}

    for row in rows:
        categories[row["id"]] = {
            "id": row["id"],
            "external_id": row["db_id"],
            "name": clean_string(row["name"]),
            "image": build_media_url(row["image_field"]),
        }

    logger.info(
        "Loaded %s product categories",
        len(categories),
    )

    return categories

def load_currencies(pg):
    """
    Load currencies.
    """
    logger.info("Loading currencies...")

    sql = """
        SELECT
            id,
            code,
            name,
            symbol,
            currency_id
        FROM base_currency
        ORDER BY id
    """

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    currencies = {}

    for row in rows:
        currencies[row["id"]] = {
            "id": row["id"],
            "code": clean_string(row["code"]),
            "name": clean_string(row["name"]),
            "symbol": clean_string(row["symbol"]),
            "external_currency_id": row["currency_id"],
        }

    logger.info(
        "Loaded %s currencies",
        len(currencies),
    )

    return currencies

def load_countries(pg):
    """
    Load countries.
    """
    logger.info("Loading countries...")

    sql = """
        SELECT
            id,
            db_id,
            name,
            ac_name,
            country_code,
            sub_domain,
            code
        FROM countries
        ORDER BY id
    """

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    countries = {}

    for row in rows:
        countries[row["id"]] = {
            "id": row["id"],
            "external_id": row["db_id"],
            "name": clean_string(row["name"]),
            "ac_name": clean_string(row["ac_name"]),
            "country_code": clean_string(row["country_code"]),
            "sub_domain": clean_string(row["sub_domain"]),
            "code": clean_string(row["code"]),
        }

    logger.info(
        "Loaded %s countries",
        len(countries),
    )

    return countries

def load_package_types(pg):
    """
    Load PackageType lookup.

    Only use columns confirmed to exist in base_packagetype.
    Product migration does not depend on the package-type
    boolean flags.
    """
    logger.info("Loading package types...")

    sql = """
        SELECT
            id,
            name,
            description
        FROM base_packagetype
        ORDER BY id
    """

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    package_types = {}

    for row in rows:
        package_types[row["id"]] = {
            "id": row["id"],
            "name": clean_string(row["name"]),
            "description": clean_html(row["description"]),
        }

    logger.info(
        "Loaded %s package types",
        len(package_types),
    )

    return package_types

def load_packages(pg, package_types):
    """
    Load Product packages.
    """
    logger.info("Loading packages...")

    sql = """
        SELECT
            bp.id,
            bp.package_id AS external_package_id,
            bp.package_type_id,
            bp.credits,
            bp.price_per_credit,
            bp.validity_days,
            bp.is_customizable
        FROM base_package bp
        ORDER BY bp.id
    """

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    packages = {}

    for row in rows:
        package_type = package_types.get(
            row["package_type_id"]
        )

        packages[row["id"]] = {
            "id": row["id"],
            "external_package_id": row["external_package_id"],
            "package_type_id": row["package_type_id"],
            "package_type": package_type,
            "credits": json_safe(row["credits"]),
            "price_per_credit": json_safe(
                row["price_per_credit"]
            ),
            "validity_days": row["validity_days"],
            "is_customizable": bool(
                row["is_customizable"]
            ),
        }

    logger.info(
        "Loaded %s packages",
        len(packages),
    )

    return packages

def load_product_category_relations(pg, product_ids):
    """
    Load Product -> ProductCategory relationships.
    """
    if not product_ids:
        return {}

    sql = """
        SELECT
            pc.product_id,
            pc.productcategory_id,
            c.db_id,
            c.name,
            c.image_field
        FROM marketplace_product_category pc
        JOIN marketplace_productcategory c
            ON c.id = pc.productcategory_id
        WHERE pc.product_id = ANY(%s)
        ORDER BY pc.product_id, pc.productcategory_id
    """

    result = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (product_ids,))
        rows = cur.fetchall()

    for row in rows:
        product_id = row["product_id"]

        result.setdefault(product_id, [])

        result[product_id].append({
            "id": row["productcategory_id"],
            "external_id": row["db_id"],
            "name": clean_string(row["name"]),
            "image": build_media_url(
                row["image_field"]
            ),
        })

    return result

def load_available_country_relations(
    pg,
    product_ids,
    countries,
):
    """
    Load Product -> available countries relationships.
    """
    if not product_ids:
        return {}

    sql = """
        SELECT
            pc.product_id,
            pc.country_id,
            c.db_id,
            c.name,
            c.ac_name,
            c.country_code,
            c.sub_domain,
            c.code
        FROM marketplace_product_available_in_countries pc
        JOIN countries c
            ON c.id = pc.country_id
        WHERE pc.product_id = ANY(%s)
        ORDER BY pc.product_id, pc.country_id
    """

    result = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (product_ids,))
        rows = cur.fetchall()

    for row in rows:
        product_id = row["product_id"]

        result.setdefault(product_id, [])

        result[product_id].append(
            countries.get(
                row["country_id"],
                {
                    "id": row["country_id"],
                    "external_id": row["db_id"],
                    "name": clean_string(row["name"]),
                    "ac_name": clean_string(row["ac_name"]),
                    "country_code": clean_string(
                        row["country_code"]
                    ),
                    "sub_domain": clean_string(
                        row["sub_domain"]
                    ),
                    "code": clean_string(row["code"]),
                },
            )
        )

    return result

def load_product_gallery(pg, product_ids):
    """
    Load marketplace_productimage records.

    PDFs are intentionally represented as `document`.
    """
    if not product_ids:
        return {}

    sql = """
        SELECT
            id,
            image,
            product_id
        FROM marketplace_productimage
        WHERE product_id = ANY(%s)
        ORDER BY product_id, id
    """

    result = {}

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (product_ids,))
        rows = cur.fetchall()

    for row in rows:
        product_id = row["product_id"]

        path = clean_string(row["image"])

        if not path:
            continue

        result.setdefault(product_id, [])

        result[product_id].append({
            "id": row["id"],
            "path": build_media_url(path),
            "type": media_type_from_path(path),
        })

    return result

def load_sellers(pg, seller_ids):
    """
    Load public seller information.

    Private authentication/security fields are intentionally
    excluded.

    For company accounts, companies.name is preferred as the
    canonical public seller name.
    """
    if not seller_ids:
        return {}

    sql = """
        SELECT
            u.id,
            u.username,
            u.first_name,
            u.last_name,
            u.user_type,
            u.city_town,
            u.slug,
            u.avatar,
            u.cover,
            u.about_us,
            u.tagline,
            u.no_of_employees,
            u.verified,
            u.is_pro,
            u.is_active,
            u.is_featured,
            u.is_working,
            u.current_country_id,
            co.name AS company_name
        FROM user_accounts u
        LEFT JOIN companies co
            ON co.useraccount_ptr_id = u.id
        WHERE u.id = ANY(%s)
    """

    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (seller_ids,))
        rows = cur.fetchall()

    sellers = {}

    for row in rows:
        first_name = clean_string(
            row["first_name"]
        )

        last_name = clean_string(
            row["last_name"]
        )

        full_name = " ".join(
            part
            for part in [first_name, last_name]
            if part
        )

        user_type = clean_string(
            row["user_type"]
        )

        company_name = clean_string(
            row["company_name"]
        )

        username = clean_string(
            row["username"]
        )

        if user_type == "company":
            seller_name = (
                company_name
                or full_name
                or username
            )
        else:
            seller_name = (
                full_name
                or username
                or company_name
            )

        sellers[row["id"]] = {
            "id": row["id"],
            "name": seller_name,
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
            "user_type": user_type,
            "company_name": company_name,
            "city": clean_string(
                row["city_town"]
            ),
            "slug": clean_string(
                row["slug"]
            ),
            "avatar": build_media_url(
                row["avatar"]
            ),
            "cover": build_media_url(
                row["cover"]
            ),
            "about_us": clean_html(
                row["about_us"]
            ),
            "tagline": clean_string(
                row["tagline"]
            ),
            "no_of_employees": clean_string(
                row["no_of_employees"]
            ),
            "verified": bool(
                row["verified"]
            ),
            "is_pro": bool(
                row["is_pro"]
            ),
            "is_active": bool(
                row["is_active"]
            ),
            "is_featured": bool(
                row["is_featured"]
            ),
            "is_working": bool(
                row["is_working"]
            ),
            "current_country_id": row[
                "current_country_id"
            ],
        }

    return sellers

def fetch_products(
    pg,
    start_id=0,
    limit=None,
):
    """
    Fetch Products incrementally by PostgreSQL primary key.

    This avoids OFFSET pagination and remains stable while
    migrating.
    """
    sql = """
        SELECT
            p.id,
            p.title,
            p.p_type,
            p.price,
            p.current_location,
            p.prime_city,
            p.other_location,
            p.youtube_url,
            p.website_link,
            p.keywords,
            p.description,
            p.main_image,
            p.created_at,
            p.status,
            p.expiry_date,
            p.views,
            p.currency_id,
            p.posted_by_id,
            p.price_setting,
            p.p_condition,
            p.country_id,
            p.discount_percentage,
            p.discounted_price,
            p.is_featured,
            p.slug,
            p.start_date,
            p.reposted_at,
            p.package_id,
            p.my_sql_product_id,
            p.is_auto_renew_enabled,
            p.valid_upto
        FROM marketplace_product p
        WHERE p.id > %s
        ORDER BY p.id
    """

    params = [start_id]

    if limit is not None:
        sql += "\nLIMIT %s"
        params.append(limit)

    with pg.cursor(
        cursor_factory=RealDictCursor
    ) as cur:
        cur.execute(sql, params)

        return cur.fetchall()

def build_product_search_text(
    product,
    seller,
    categories,
    country,
    available_countries,
    currency,
    package,
):
    """
    Build rich natural-language search text.

    This is intended for later embedding/vector search,
    semantic retrieval and keyword search.
    """

    parts = []

    title = clean_string(product["title"])

    if title:
        parts.append(
            f"Product: {title}"
        )

    product_type = clean_string(
        product["p_type"]
    )

    if product_type:
        parts.append(
            f"Product type: {product_type}"
        )

    condition = clean_string(
        product["p_condition"]
    )

    if condition:
        parts.append(
            f"Condition: {condition}"
        )

    price_setting = clean_string(
        product["price_setting"]
    )

    if price_setting:
        parts.append(
            f"Price setting: {price_setting}"
        )

    if product["price"] is not None:
        parts.append(
            f"Price: {product['price']}"
        )

    if currency:
        currency_text = " ".join(
            str(x)
            for x in [
                currency.get("code"),
                currency.get("name"),
                currency.get("symbol"),
            ]
            if x
        )

        if currency_text:
            parts.append(
                f"Currency: {currency_text}"
            )

    if product["discount_percentage"] is not None:
        parts.append(
            "Discount percentage: "
            f"{product['discount_percentage']}"
        )

    if product["discounted_price"] is not None:
        parts.append(
            "Discounted price: "
            f"{product['discounted_price']}"
        )

    current_location = clean_string(
        product["current_location"]
    )

    prime_city = clean_string(
        product["prime_city"]
    )

    other_location = clean_string(
        product["other_location"]
    )

    if current_location:
        parts.append(
            f"Current location: {current_location}"
        )

    if prime_city:
        parts.append(
            f"City: {prime_city}"
        )

    if other_location:
        parts.append(
            f"Other location: {other_location}"
        )

    if country:
        parts.append(
            "Country: "
            + " ".join(
                str(x)
                for x in [
                    country.get("name"),
                    country.get("code"),
                    country.get("country_code"),
                ]
                if x
            )
        )

    if available_countries:
        names = [
            x.get("name")
            for x in available_countries
            if x.get("name")
        ]

        if names:
            parts.append(
                "Available in countries: "
                + ", ".join(names)
            )

    category_names = [
        x.get("name")
        for x in categories
        if x.get("name")
    ]

    if category_names:
        parts.append(
            "Categories: "
            + ", ".join(category_names)
        )

    if seller:
        seller_name = seller.get("name")

        if seller_name:
            parts.append(
                f"Seller: {seller_name}"
            )

        seller_type = seller.get(
            "user_type"
        )

        if seller_type:
            parts.append(
                f"Seller type: {seller_type}"
            )

        seller_city = seller.get("city")

        if seller_city:
            parts.append(
                f"Seller city: {seller_city}"
            )

        tagline = seller.get("tagline")

        if tagline:
            parts.append(
                f"Seller tagline: {tagline}"
            )

        about_us = seller.get(
            "about_us"
        )

        if about_us:
            parts.append(
                f"Seller description: {about_us}"
            )

    if package:
        package_type = package.get(
            "package_type"
        )

        if package_type:
            package_name = package_type.get(
                "name"
            )

            if package_name:
                parts.append(
                    f"Package: {package_name}"
                )

    keyword_list = split_keywords(
        product["keywords"]
    )

    if keyword_list:
        parts.append(
            "Keywords: "
            + ", ".join(keyword_list)
        )

    description = clean_html(
        product["description"]
    )

    if description:
        parts.append(
            f"Description: {description}"
        )

    return "\n".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )

def build_search_keywords(
    product,
    categories,
    seller,
    country,
):
    values = []

    values.extend(
        split_keywords(
            product["keywords"]
        )
    )

    if product["title"]:
        values.append(
            product["title"]
        )

    if product["p_type"]:
        values.append(
            product["p_type"]
        )

    if product["p_condition"]:
        values.append(
            product["p_condition"]
        )

    for category in categories:
        values.append(
            category.get("name")
        )

    if seller:
        values.append(
            seller.get("name")
        )

        values.append(
            seller.get("username")
        )

    if country:
        values.append(
            country.get("name")
        )

        values.append(
            country.get("code")
        )

    return unique_strings(values)

def build_search_aliases(
    product,
    categories,
):
    """
    Lightweight aliases useful for search matching.
    """
    aliases = []

    title = clean_string(
        product["title"]
    )

    if title:
        aliases.append(title)

    for category in categories:
        name = category.get("name")

        if name:
            aliases.append(name)

    return unique_strings(aliases)

def build_product_document(
    product,
    categories_map,
    currencies,
    countries,
    packages,
    sellers,
    migrated_at,
):
    product_id = product["id"]

    categories = categories_map.get(
        product_id,
        [],
    )

    country = countries.get(
        product["country_id"]
    )

    available_countries = (
        product.pop(
            "_available_countries",
            None,
        )
        or []
    )

    gallery = (
        product.pop(
            "_gallery",
            None,
        )
        or []
    )

    seller = sellers.get(
        product["posted_by_id"]
    )

    currency = currencies.get(
        product["currency_id"]
    )

    package = packages.get(
        product["package_id"]
    )

    main_image = build_media_url(
        product["main_image"]
    )

    main_image_media = None

    if main_image:
        main_image_media = {
            "path": main_image,
            "type": "image",
        }

    keywords = split_keywords(
        product["keywords"]
    )

    description = clean_html(
        product["description"]
    )

    if description:
        description = description[:500_000]

    search_text = build_product_search_text(
        product=product,
        seller=seller,
        categories=categories,
        country=country,
        available_countries=available_countries,
        currency=currency,
        package=package,
    )

    search_text = search_text[:750_000]

    search_keywords = build_search_keywords(
        product=product,
        categories=categories,
        seller=seller,
        country=country,
    )

    search_aliases = build_search_aliases(
        product=product,
        categories=categories,
    )

    is_live = calculate_is_live(
        product["status"],
        product["expiry_date"],
    )

    document = {
        "_id": f"product:{product_id}",

        "entity_type": "product",

        "schema_version": 1,

        "title": clean_string(
            product["title"]
        ),

        "slug": clean_string(
            product["slug"]
        ),

        "product_type": clean_string(
            product["p_type"]
        ),

        "condition": clean_string(
            product["p_condition"]
        ),

        "pricing": {
            "price": json_safe(
                product["price"]
            ),
            "price_setting": clean_string(
                product["price_setting"]
            ),
            "currency": currency,
            "discount_percentage": json_safe(
                product[
                    "discount_percentage"
                ]
            ),
            "discounted_price": json_safe(
                product[
                    "discounted_price"
                ]
            ),
        },

        "location": {
            "current_location": clean_string(
                product[
                    "current_location"
                ]
            ),
            "prime_city": clean_string(
                product["prime_city"]
            ),
            "other_location": clean_string(
                product["other_location"]
            ),
            "country": country,
            "available_in_countries":
                available_countries,
        },

        "categories": categories,

        "description": description,

        "keywords": keywords,

        "media": {
            "main_image": main_image_media,
            "gallery": gallery,
        },

        "links": {
            "website": clean_url(
                product["website_link"]
            ),
            "youtube": clean_url(
                product["youtube_url"]
            ),
        },

        "seller": seller,

        "package": package,

        "engagement": {
            "views": product["views"] or 0,
        },

        "flags": {
            "is_featured": bool(
                product["is_featured"]
            ),
            "is_auto_renew_enabled": bool(
                product[
                    "is_auto_renew_enabled"
                ]
            ),
        },

        "dates": {
            "created_at": json_safe(
                product["created_at"]
            ),
            "reposted_at": json_safe(
                product["reposted_at"]
            ),
            "start_date": json_safe(
                product["start_date"]
            ),
            "expiry_date": json_safe(
                product["expiry_date"]
            ),
            "valid_upto": json_safe(
                product["valid_upto"]
            ),
        },

        "status": clean_string(
            product["status"]
        ),

        "is_live": is_live,

        "source": {
            "model": "Product",
            "table": "marketplace_product",
            "object_id": product_id,
            "my_sql_product_id": product[
                "my_sql_product_id"
            ],
        },

        "search_keywords": search_keywords,

        "search_aliases": search_aliases,

        "ai_search_text": search_text,

        "embedding": {
            "dimensions": 384,
            "status": "pending",
        },

        "migration": {
            "source": "postgresql",
            "migrated_at": migrated_at,
        },
    }

    return document

def migrate_products(
    limit=None,
    start_id=0,
    batch_size=BATCH_SIZE,
):
    pg = None
    mongo_client = None

    processed = 0
    written = 0
    errors = 0
    last_id = start_id

    try:
        
        pg = get_postgres_connection()

        mongo_client, collection = (
            get_mongo_collection()
        )

        categories = load_categories(pg)

        currencies = load_currencies(pg)

        countries = load_countries(pg)

        package_types = load_package_types(pg)

        packages = load_packages(
            pg,
            package_types,
        )

        logger.info(
            "Starting Product migration..."
        )

        logger.info(
            "start_id=%s limit=%s batch_size=%s",
            start_id,
            limit,
            batch_size,
        )

        remaining = limit

        while True:

            fetch_limit = batch_size

            if remaining is not None:
                fetch_limit = min(
                    fetch_limit,
                    remaining,
                )

            rows = fetch_products(
                pg=pg,
                start_id=last_id,
                limit=fetch_limit,
            )

            if not rows:
                break

            product_ids = [
                row["id"]
                for row in rows
            ]

            seller_ids = list(
                {
                    row["posted_by_id"]
                    for row in rows
                    if row["posted_by_id"]
                    is not None
                }
            )

            category_relations = (
                load_product_category_relations(
                    pg,
                    product_ids,
                )
            )

            available_country_relations = (
                load_available_country_relations(
                    pg,
                    product_ids,
                    countries,
                )
            )

            gallery_relations = (
                load_product_gallery(
                    pg,
                    product_ids,
                )
            )

            sellers = load_sellers(
                pg,
                seller_ids,
            )

            operations = []

            migrated_at = datetime.now(
                timezone.utc
            )

            for row in rows:

                try:
                    product_id = row["id"]

                    row["_available_countries"] = (
                        available_country_relations.get(
                            product_id,
                            [],
                        )
                    )

                    row["_gallery"] = (
                        gallery_relations.get(
                            product_id,
                            [],
                        )
                    )

                    document = (
                        build_product_document(
                            product=row,
                            categories_map=
                                category_relations,
                            currencies=currencies,
                            countries=countries,
                            packages=packages,
                            sellers=sellers,
                            migrated_at=
                                migrated_at,
                        )
                    )

                    operations.append(
                        UpdateOne(
                            {
                                "_id":
                                    document["_id"]
                            },
                            {
                                "$set":
                                    document
                            },
                            upsert=True,
                        )
                    )

                    last_id = product_id

                except Exception as exc:
                    errors += 1

                    logger.exception(
                        "Failed to build Product %s: %s",
                        row.get("id"),
                        exc,
                    )

            if operations:

                try:
                    result = (
                        collection.bulk_write(
                            operations,
                            ordered=False,
                        )
                    )

                    batch_written = (
                        result.upserted_count
                        + result.modified_count
                    )

                    written += batch_written

                except Exception as exc:
                    errors += 1

                    logger.exception(
                        "MongoDB bulk write failed "
                        "for batch ending at product %s: %s",
                        last_id,
                        exc,
                    )

            batch_processed = len(rows)

            processed += batch_processed

            if remaining is not None:
                remaining -= batch_processed

            logger.info(
                "Processed batch: %s | "
                "Total processed: %s | "
                "Written/updated: %s | "
                "Errors: %s | "
                "Last ID: %s",
                batch_processed,
                processed,
                written,
                errors,
                last_id,
            )

            if remaining is not None and remaining <= 0:
                break

            if len(rows) < fetch_limit:
                break

        logger.info(
            "PRODUCT MIGRATION COMPLETE"
        )

        logger.info(
            "Processed : %s",
            processed,
        )

        logger.info(
            "Written   : %s",
            written,
        )

        logger.info(
            "Errors    : %s",
            errors,
        )

        logger.info(
            "Last ID   : %s",
            last_id,
        )

        return {
            "processed": processed,
            "written": written,
            "errors": errors,
            "last_id": last_id,
        }

    finally:

        if pg:
            pg.close()

        if mongo_client:
            mongo_client.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Migrate Hozpitality Products "
            "from PostgreSQL to MongoDB"
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of products to migrate"
        ),
    )

    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help=(
            "Start after PostgreSQL Product ID"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "Number of products processed per batch"
        ),
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit must be greater than 0"
        )

    if args.start_id < 0:
        raise ValueError(
            "--start-id cannot be negative"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than 0"
        )

    migrate_products(
        limit=args.limit,
        start_id=args.start_id,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()