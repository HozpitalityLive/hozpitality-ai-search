# copy from dev to opt
# sudo cp ~/hozpitality-mongo-migration/migrate_jobs.py /opt/hozpitality-mongo-migration/migrate_jobs.py

# sudo -u postgres env MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_jobs.py \
# --limit=100

# sudo -u postgres env \
# MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_jobs.py

import argparse
import html
import os
import re

from datetime import date, datetime, time
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient, UpdateOne

PG_HOST = os.getenv("PG_HOST", "")
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

BATCH_SIZE = 100

SCHEMA_VERSION = 2

def clean_text(v: Any) -> str | None:
    """
    Basic text cleanup.

    Does NOT remove HTML tags.
    Use clean_search_text() when HTML must be converted to readable text.
    """
    if v is None:
        return None

    v = html.unescape(str(v))
    v = v.replace("\x00", " ")
    v = re.sub(r"\s+", " ", v).strip()

    return v or None

def html_to_text(v: Any) -> str | None:
    """
    Convert HTML content into readable plain text.
    """
    if v is None:
        return None

    s = str(v).replace("\x00", " ")

    s = re.sub(
        r"(?i)</(li|p|div|br|h[1-6]|ul|ol|section|article)>",
        "\n",
        s,
    )

    s = re.sub(
        r"(?i)<br\s*/?>",
        "\n",
        s,
    )

    s = re.sub(r"<[^>]+>", " ", s)

    s = html.unescape(s)

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in s.splitlines()
    ]

    lines = [line for line in lines if line]

    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip() or None

def clean_search_text(v: Any) -> str | None:
    """
    Convert rich HTML/text into clean AI/search-friendly text.

    This should be used for:
    - job descriptions
    - company descriptions
    - other rich-text fields
    """
    s = html_to_text(v)

    if not s:
        return None

    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"__(.*?)__", r"\1", s)

    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip() or None

def dedupe(values: List[Any]) -> List[str]:
    """
    Clean and case-insensitively deduplicate values while preserving order.
    """
    out = []
    seen = set()

    for value in values:
        value = clean_text(value)

        if not value:
            continue

        key = value.casefold()

        if key not in seen:
            seen.add(key)
            out.append(value)

    return out

def media_url(v: Any) -> str | None:
    """
    Convert stored media path into an absolute CloudFront URL.
    """
    v = clean_text(v)

    if not v:
        return None

    if v.startswith(("http://", "https://")):
        return v

    return f"{CLOUDFRONT_BASE}/{v.lstrip('/')}"

def iso(v: Any) -> Any:
    """
    Convert PostgreSQL date/datetime/time values to ISO strings.
    """
    if isinstance(v, (date, datetime, time)):
        return v.isoformat()

    return v

def add(parts: List[str], label: str, value: Any) -> None:
    """
    Add a simple clean-text field to AI text.
    """
    value = clean_text(value)

    if value:
        parts.append(f"{label}: {value}")

def add_list(parts: List[str], label: str, values: List[Any]) -> None:
    """
    Add a deduplicated list to AI text.
    """
    values = dedupe(values)

    if values:
        parts.append(f"{label}: {', '.join(values)}")

JOB_QUERY = """
SELECT
    j.id,
    j.my_sql_job_id,
    j.job_status,
    j.company_id,
    j.job_title,
    j.job_desc,
    j.job_country_id,
    j.job_city,
    j.job_address,
    j.latitude,
    j.longitude,
    j.job_start_date,
    j.job_end_date,
    j.posted_by,
    j.posted_by_user_id,
    j.jobtype_id,
    j.employementtype_id,
    j.salary_description,
    j.currency_id,
    j."salaryRange_id",
    j.reference,
    j.job_link,
    j.is_live,
    j.is_premium,
    j.is_spider,
    j.hide_company_details,
    j.slug,
    j.is_featured,
    j.views,
    j.impressions,
    j.matching_views,
    j.matching_profiles,
    j.created_at,
    j.updated_at,
    j.reposted_at,
    j.is_deleted,
    j.receive_response,
    j."walkInFromDate",
    j."walkInToDate",
    j."walkInTime",
    j."walkInVenue",
    j.is_confidential,
    j.is_auto_renew_enabled,
    j.is_spider_job,
    j.spider_url,
    j.package_type_id,
    j.is_migrated,
    j.is_credit_used,
    j.sent_mail_status,
    j.expiry_reminder_status,
    j.tags,
    j.job_avatar,
    j.job_document,

    -- Job country
    jc.id country_id,
    jc.db_id country_db_id,
    jc.name country_name,
    jc.ac_name country_ac_name,
    jc.country_code,
    jc.code country_code_short,

    -- Employment type
    et.id employment_type_id,
    et.db_id employment_type_db_id,
    et.name employment_type_name,

    -- Job type
    jt.id job_type_id,
    jt.db_id job_type_db_id,
    jt.name job_type_name,

    -- Currency
    cur.id currency_id_ref,
    cur.name currency_name,
    cur.code currency_code,
    cur.currency_id currency_external_id,
    cur.symbol currency_symbol,

    -- Salary range
    sr.id salary_range_id,
    sr.db_id salary_range_db_id,
    sr.name salary_range_name,

    -- Package
    pt.id package_type_id_ref,
    pt.name package_type_name,
    pt.description package_type_description,
    pt."is_PAYG" package_is_payg,
    pt."is_POP" package_is_pop,
    pt."is_CC" package_is_cc,
    pt."is_PREMIUM" package_is_premium,
    pt."is_SP" package_is_sp,
    pt."is_GP" package_is_gp,
    pt."is_EP" package_is_ep,

    -- Company
    cu.id company_user_id,
    co.name company_name,
    cu.first_name company_first_name,
    cu.last_name company_last_name,
    cu.username company_username,
    cu.user_type company_user_type,
    cu.slug company_slug,
    cu.city_town company_city,
    cu.about_us company_about,
    cu.tagline company_tagline,
    cu.no_of_employees company_employee_range,
    cu.verified company_verified,
    cu.is_pro company_is_pro,
    cu.is_featured company_is_featured,
    cu.is_active company_is_active,
    cu.avatar company_avatar,
    cu.cover company_cover,

    -- Company country
    cc.id company_country_id,
    cc.db_id company_country_db_id,
    cc.name company_country_name,
    cc.country_code company_country_code,

    -- Posted-by user
    pu.id posted_user_id,
    pu.first_name posted_user_first_name,
    pu.last_name posted_user_last_name,
    pu.username posted_user_username,
    pu.user_type posted_user_type,
    pu.slug posted_user_slug,
    pu.city_town posted_user_city,
    pu.current_country_id posted_user_country_id

FROM base_job j

LEFT JOIN countries jc
    ON jc.id = j.job_country_id

LEFT JOIN employment_type et
    ON et.id = j.employementtype_id

LEFT JOIN job_type jt
    ON jt.id = j.jobtype_id

LEFT JOIN base_currency cur
    ON cur.id = j.currency_id

LEFT JOIN base_salaryrange sr
    ON sr.id = j."salaryRange_id"

LEFT JOIN base_packagetype pt
    ON pt.id = j.package_type_id

LEFT JOIN user_accounts cu
    ON cu.id = j.company_id

LEFT JOIN companies co
    ON co.useraccount_ptr_id = cu.id

LEFT JOIN countries cc
    ON cc.id = cu.current_country_id

LEFT JOIN user_accounts pu
    ON pu.id = j.posted_by_user_id

WHERE
    j.is_live = true
    AND j.is_deleted = false

ORDER BY j.id
"""

REL_QUERIES = {

    "industries": """
        SELECT
            ji.job_id,
            i.id,
            i.db_id,
            i.name,
            i.context
        FROM base_job_job_industry ji
        JOIN industries i
            ON i.id = ji.industry_id
        WHERE ji.job_id = ANY(%s)
        ORDER BY ji.job_id, i.name
    """,

    "levels": """
        SELECT
            x.job_id,
            x2.id,
            x2.db_id,
            x2.name
        FROM base_job_job_level x
        JOIN job_levels x2
            ON x2.id = x.joblevel_id
        WHERE x.job_id = ANY(%s)
        ORDER BY x.job_id, x2.name
    """,

    "roles": """
        SELECT
            x.job_id,
            x2.id,
            x2.db_id,
            x2.name
        FROM base_job_job_role x
        JOIN job_role x2
            ON x2.id = x.jobrole_id
        WHERE x.job_id = ANY(%s)
        ORDER BY x.job_id, x2.name
    """,

    "departments": """
        SELECT
            x.job_id,
            x2.id,
            x2.db_id,
            x2.name
        FROM base_job_job_department x
        JOIN departments x2
            ON x2.id = x.department_id
        WHERE x.job_id = ANY(%s)
        ORDER BY x.job_id, x2.name
    """,

    "received": """
        SELECT
            x.job_id,
            c.id,
            c.db_id,
            c.name,
            c.ac_name,
            c.country_code,
            c.code
        FROM base_job_received_applications x
        JOIN countries c
            ON c.id = x.country_id
        WHERE x.job_id = ANY(%s)
        ORDER BY x.job_id, c.name
    """,

    "questions": """
        SELECT
            x.job_id,
            q.id,
            q.question,
            q.expected_answer,
            q.auto_reject
        FROM base_job_filter_questions x
        JOIN base_jobfilterquestion q
            ON q.id = x.jobfilterquestion_id
        WHERE x.job_id = ANY(%s)
        ORDER BY x.job_id, q.id
    """,
}

def load_relations(conn, ids: List[int]) -> Dict[int, Dict[str, List]]:
    """
    Load all M2M / related job data for a batch.
    """

    relations = {
        job_id: {
            "industries": [],
            "levels": [],
            "roles": [],
            "departments": [],
            "received": [],
            "questions": [],
        }
        for job_id in ids
    }

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        cursor.execute(
            REL_QUERIES["industries"],
            (ids,),
        )

        for row in cursor.fetchall():
            relations[row["job_id"]]["industries"].append(dict(row))

        cursor.execute(
            REL_QUERIES["levels"],
            (ids,),
        )

        for row in cursor.fetchall():
            relations[row["job_id"]]["levels"].append(dict(row))

        cursor.execute(
            REL_QUERIES["roles"],
            (ids,),
        )

        for row in cursor.fetchall():
            relations[row["job_id"]]["roles"].append(dict(row))

        cursor.execute(
            REL_QUERIES["departments"],
            (ids,),
        )

        for row in cursor.fetchall():
            relations[row["job_id"]]["departments"].append(dict(row))

        cursor.execute(
            REL_QUERIES["received"],
            (ids,),
        )

        for row in cursor.fetchall():
            relations[row["job_id"]]["received"].append(dict(row))

        cursor.execute(
            REL_QUERIES["questions"],
            (ids,),
        )

        for row in cursor.fetchall():
            relations[row["job_id"]]["questions"].append(dict(row))

    return relations

def names(items: List[Dict[str, Any]]) -> List[Any]:
    """
    Extract names from relation objects.
    """
    return [
        item.get("name")
        for item in items
        if item.get("name")
    ]

def ai_text(
    row: Dict[str, Any],
    rel: Dict[str, List],
) -> str:

    parts = []

    title = clean_text(row["job_title"])

    company_name = (
        clean_text(row["company_name"])
        or " ".join(
            dedupe([
                row["company_first_name"],
                row["company_last_name"],
            ])
        )
        or clean_text(row["company_username"])
    )

    if title:
        parts.append(f"Job title: {title}")

    parts.append(
        "This is a hospitality employment opportunity."
    )

    if company_name:
        parts.append(
            f"Company: {company_name}"
        )

    add(
        parts,
        "Company tagline",
        row["company_tagline"],
    )

    company_about = clean_search_text(
        row["company_about"]
    )

    if company_about:
        parts.append(
            f"Company description: {company_about}"
        )

    if row["company_verified"]:
        parts.append(
            "Company is verified."
        )

    if row["company_is_pro"]:
        parts.append(
            "Company is a Pro member."
        )

    add(
        parts,
        "Company size",
        row["company_employee_range"],
    )

    company_location = dedupe([
        row["company_city"],
        row["company_country_name"],
    ])

    if company_location:
        parts.append(
            "Company location: "
            + ", ".join(company_location)
        )

    job_location = dedupe([
        row["job_city"],
        row["country_name"],
    ])

    if job_location:
        parts.append(
            "Job location: "
            + ", ".join(job_location)
        )

    add(
        parts,
        "Job address",
        row["job_address"],
    )

    add_list(
        parts,
        "Industries",
        names(rel["industries"]),
    )

    add_list(
        parts,
        "Job levels",
        names(rel["levels"]),
    )

    add_list(
        parts,
        "Job roles",
        names(rel["roles"]),
    )

    add_list(
        parts,
        "Departments",
        names(rel["departments"]),
    )

    add(
        parts,
        "Employment type",
        row["employment_type_name"],
    )

    add(
        parts,
        "Job type",
        row["job_type_name"],
    )

    add(
        parts,
        "Salary description",
        row["salary_description"],
    )

    add(
        parts,
        "Salary range",
        row["salary_range_name"],
    )

    currency = dedupe([
        row["currency_name"],
        row["currency_code"],
        row["currency_symbol"],
    ])

    if currency:
        parts.append(
            "Currency: "
            + ", ".join(currency)
        )

    add(
        parts,
        "Job tags",
        row["tags"],
    )

    description = clean_search_text(
        row["job_desc"]
    )

    if description:
        parts.append(
            "Full job description:\n"
            + description
        )

    walk_in = []

    if row["walkInFromDate"]:
        walk_in.append(
            f"from {row['walkInFromDate']}"
        )

    if row["walkInToDate"]:
        walk_in.append(
            f"to {row['walkInToDate']}"
        )

    if row["walkInTime"]:
        walk_in.append(
            f"at {row['walkInTime']}"
        )

    if row["walkInVenue"]:
        walk_in.append(
            "venue "
            + clean_text(row["walkInVenue"])
        )

    if walk_in:
        parts.append(
            "Walk-in information: "
            + ", ".join(walk_in)
        )

    add(
        parts,
        "Application response method",
        row["receive_response"],
    )

    add_list(
        parts,
        "Countries from which applications are received",
        names(rel["received"]),
    )

    if row["is_premium"]:
        parts.append(
            "Premium job listing."
        )

    if row["is_featured"]:
        parts.append(
            "Featured job listing."
        )

    if row["is_confidential"]:
        parts.append(
            "This is a confidential job listing."
        )

    if row["hide_company_details"]:
        parts.append(
            "Company details may be hidden on the public listing."
        )

    if row["is_spider"] or row["is_spider_job"]:
        parts.append(
            "This job originated from an imported/spider source."
        )

    add(
        parts,
        "Job start date",
        row["job_start_date"],
    )

    add(
        parts,
        "Job end date",
        row["job_end_date"],
    )

    add_list(
        parts,
        "Candidate screening questions",
        [
            q["question"]
            for q in rel["questions"]
            if q.get("question")
        ],
    )

    return "\n\n".join(
        part
        for part in parts
        if part
    ).strip()

def build_doc(
    row: Dict[str, Any],
    rel: Dict[str, List],
) -> Dict[str, Any]:

    def obj(**kwargs):
        return {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }

    company_name = (
        clean_text(row["company_name"])
        or " ".join(
            dedupe([
                row["company_first_name"],
                row["company_last_name"],
            ])
        )
        or clean_text(row["company_username"])
    )

    country = {
        key: value
        for key, value in {
            "id": row["country_id"],
            "db_id": row["country_db_id"],
            "name": clean_text(row["country_name"]),
            "ac_name": clean_text(row["country_ac_name"]),
            "country_code": clean_text(row["country_code"]),
            "code": clean_text(row["country_code_short"]),
        }.items()
        if value is not None
    }

    company_country = {
        key: value
        for key, value in {
            "id": row["company_country_id"],
            "db_id": row["company_country_db_id"],
            "name": clean_text(row["company_country_name"]),
            "country_code": clean_text(row["company_country_code"]),
        }.items()
        if value is not None
    }

    currency = obj(
        id=row["currency_id_ref"],
        name=clean_text(row["currency_name"]),
        code=clean_text(row["currency_code"]),
        currency_id=row["currency_external_id"],
        symbol=clean_text(row["currency_symbol"]),
    )

    salary = obj(
        id=row["salary_range_id"],
        db_id=row["salary_range_db_id"],
        name=clean_text(row["salary_range_name"]),
    )

    employment = obj(
        id=row["employment_type_id"],
        db_id=row["employment_type_db_id"],
        name=clean_text(row["employment_type_name"]),
    )

    job_type = obj(
        id=row["job_type_id"],
        db_id=row["job_type_db_id"],
        name=clean_text(row["job_type_name"]),
    )

    posted = obj(
        id=row["posted_user_id"],
        first_name=clean_text(
            row["posted_user_first_name"]
        ),
        last_name=clean_text(
            row["posted_user_last_name"]
        ),
        username=clean_text(
            row["posted_user_username"]
        ),
        user_type=clean_text(
            row["posted_user_type"]
        ),
        slug=clean_text(
            row["posted_user_slug"]
        ),
        city=clean_text(
            row["posted_user_city"]
        ),
    )

    package = obj(
        id=row["package_type_id_ref"],
        name=clean_text(
            row["package_type_name"]
        ),
        description=clean_text(
            row["package_type_description"]
        ),
        is_payg=row["package_is_payg"],
        is_pop=row["package_is_pop"],
        is_cc=row["package_is_cc"],
        is_premium=row["package_is_premium"],
        is_sp=row["package_is_sp"],
        is_gp=row["package_is_gp"],
        is_ep=row["package_is_ep"],
    )

    tags = dedupe(
        re.split(
            r"[,|;/]+",
            clean_text(row["tags"]) or "",
        )
    )

    keywords = dedupe(
        [
            row["job_title"],
            row["job_city"],
            row["country_name"],
            row["country_code"],
            row["employment_type_name"],
            row["job_type_name"],
            row["salary_range_name"],
            row["currency_code"],
            row["tags"],

            row["company_name"],

            row["company_username"],
            row["company_city"],
            row["company_country_name"],
        ]
        + names(rel["industries"])
        + names(rel["levels"])
        + names(rel["roles"])
        + names(rel["departments"])
    )

    aliases = dedupe(
        [
            "hospitality job",
            "hospitality jobs",
            "hotel job",
            "hotel jobs",
            "hospitality career",
            "hospitality careers",
            row["job_title"],
        ]
        + names(rel["roles"])
        + names(rel["departments"])
    )

    description = clean_search_text(
        row["job_desc"]
    )

    return {

        "_id": f"job:{row['id']}",

        "schema_version": SCHEMA_VERSION,

        "entity_type": "job",

        "source": {
            "model": "Job",
            "table": "base_job",
            "object_id": row["id"],
            "my_sql_job_id": row["my_sql_job_id"],
            "reference": clean_text(
                row["reference"]
            ),
        },

        "title": clean_text(
            row["job_title"]
        ),

        "slug": clean_text(
            row["slug"]
        ),

        "summary": description,

        "company": obj(
            id=row["company_user_id"],

            name=company_name,

            username=clean_text(
                row["company_username"]
            ),

            slug=clean_text(
                row["company_slug"]
            ),

            user_type=clean_text(
                row["company_user_type"]
            ),

            city=clean_text(
                row["company_city"]
            ),

            country=company_country,

            verified=bool(
                row["company_verified"]
            ),

            is_pro=bool(
                row["company_is_pro"]
            ),

            is_featured=bool(
                row["company_is_featured"]
            ),

            is_active=bool(
                row["company_is_active"]
            ),

            profile_url=media_url(
                row["company_avatar"]
            ),

            cover_url=media_url(
                row["company_cover"]
            ),
        ),

        "location": {
            "city": clean_text(
                row["job_city"]
            ),

            "country": country,

            "address": clean_text(
                row["job_address"]
            ),

            "latitude": row["latitude"],

            "longitude": row["longitude"],
        },

        "job": {

            "description": description,

            "industries": rel["industries"],

            "levels": rel["levels"],

            "roles": rel["roles"],

            "departments": rel["departments"],

            "employment_type": employment,

            "job_type": job_type,

            "salary": {
                "description": clean_text(
                    row["salary_description"]
                ),

                "currency": currency,

                "range": salary,
            },

            "tags": tags,
        },

        "walk_in": obj(
            from_date=iso(
                row["walkInFromDate"]
            ),

            to_date=iso(
                row["walkInToDate"]
            ),

            time=iso(
                row["walkInTime"]
            ),

            venue=clean_text(
                row["walkInVenue"]
            ),
        ),

        "posted_by": posted,

        "media": obj(
            avatar=media_url(
                row["job_avatar"]
            ),

            document=media_url(
                row["job_document"]
            ),
        ),

        "package": package,

        "filter_questions": rel["questions"],

        "metadata": obj(

            job_status=clean_text(
                row["job_status"]
            ),

            is_live=bool(
                row["is_live"]
            ),

            is_deleted=bool(
                row["is_deleted"]
            ),

            is_premium=bool(
                row["is_premium"]
            ),

            is_featured=bool(
                row["is_featured"]
            ),

            is_spider=bool(
                row["is_spider"]
            ),

            is_spider_job=bool(
                row["is_spider_job"]
            ),

            is_confidential=bool(
                row["is_confidential"]
            ),

            hide_company_details=bool(
                row["hide_company_details"]
            ),

            is_auto_renew_enabled=bool(
                row["is_auto_renew_enabled"]
            ),

            is_migrated=bool(
                row["is_migrated"]
            ),

            is_credit_used=row[
                "is_credit_used"
            ],

            views=row["views"],

            impressions=row["impressions"],

            matching_views=row[
                "matching_views"
            ],

            matching_profiles=row[
                "matching_profiles"
            ],

            job_start_date=iso(
                row["job_start_date"]
            ),

            job_end_date=iso(
                row["job_end_date"]
            ),

            created_at=iso(
                row["created_at"]
            ),

            updated_at=iso(
                row["updated_at"]
            ),

            reposted_at=iso(
                row["reposted_at"]
            ),

            spider_url=clean_text(
                row["spider_url"]
            ),

            package_type_id=row[
                "package_type_id"
            ],
        ),

        "search_keywords": keywords,

        "search_aliases": aliases,

        "ai_search_text": ai_text(
            row,
            rel,
        ),

        "embedding": {
            "model": None,
            "dimensions": 384,
            "vector": None,
            "status": "pending",
        },

        "is_live": True,
    }

def process_batch(
    conn,
    collection,
    rows: List[Dict[str, Any]],
):

    ids = [
        row["id"]
        for row in rows
    ]

    relations = load_relations(
        conn,
        ids,
    )

    operations = []

    for row in rows:

        document = build_doc(
            row,
            relations[row["id"]],
        )

        operations.append(
            UpdateOne(
                {
                    "_id": f"job:{row['id']}"
                },
                {
                    "$set": document
                },
                upsert=True,
            )
        )

    if not operations:
        return 0, 0, 0

    result = collection.bulk_write(
        operations,
        ordered=False,
    )

    return (
        len(rows),
        result.upserted_count,
        result.modified_count,
    )

def migrate(limit=None):

    if not MONGO_URI:
        raise RuntimeError(
            "MONGO_URI is required"
        )

    print(
        f"Starting job migration | "
        f"limit={limit if limit else 'ALL'} | "
        f"batch={BATCH_SIZE}",
        flush=True,
    )

    pg_kwargs = {
        "dbname": PG_DB,
        "user": PG_USER,
    }

    if PG_HOST:
        pg_kwargs.update(
            host=PG_HOST,
            port=PG_PORT,
        )

    pg = psycopg2.connect(
        **pg_kwargs
    )

    mongo_client = MongoClient(
        MONGO_URI
    )

    collection = (
        mongo_client[MONGO_DB]
        [MONGO_COLLECTION]
    )

    processed = 0
    upserted = 0
    modified = 0

    try:

        with pg.cursor(
            name="hozpitality_job_migration",
            cursor_factory=RealDictCursor,
        ) as cursor:

            query = JOB_QUERY

            if limit:
                query += "\nLIMIT %s"

            cursor.itersize = BATCH_SIZE

            if limit:
                cursor.execute(
                    query,
                    (limit,),
                )
            else:
                cursor.execute(
                    query
                )

            batch = []

            for row in cursor:

                batch.append(row)

                if len(batch) >= BATCH_SIZE:

                    a, b, c = process_batch(
                        pg,
                        collection,
                        batch,
                    )

                    processed += a
                    upserted += b
                    modified += c

                    batch = []

                    print(
                        f"Migrated {processed:,} jobs",
                        flush=True,
                    )

            if batch:

                a, b, c = process_batch(
                    pg,
                    collection,
                    batch,
                )

                processed += a
                upserted += b
                modified += c

        print(
            f"Done | processed={processed:,} | "
            f"upserted={upserted:,} | modified={modified:,}",
            flush=True,
        )

    finally:

        pg.close()
        mongo_client.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Migrate Hozpitality jobs from PostgreSQL to MongoDB."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of jobs to migrate. Omit for all jobs.",
    )

    args = parser.parse_args()

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        parser.error(
            "--limit must be > 0"
        )

    migrate(
        args.limit
    )