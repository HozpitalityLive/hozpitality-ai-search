# copy from dev to opt
# sudo cp ~/hozpitality-mongo-migration/migrate_professionals.py /opt/hozpitality-mongo-migration/migrate_professionals.py

# sudo -u postgres env MONGO_URI='mongodb://mongoAdmin:MongoAdmin2026I@127.0.0.1:27017/?authSource=admin' \
# /opt/hozpitality-mongo-migration/.venv/bin/python \
# /opt/hozpitality-mongo-migration/migrate_professionals.py \
# --limit=400000





import os
import sys
import time
import re
import html

import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient, UpdateOne


# ============================================================
# CONFIG
# ============================================================

CDN_BASE_URL = "https://d2he8nskrbhxwq.cloudfront.net"

PG_DB = "hozpitality"
PG_USER = "postgres"
PG_HOST = "/var/run/postgresql"

MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = "mongoAdmin"
MONGO_COLLECTION = "search_documents"

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1000"))
LIMIT = None


# ============================================================
# CLEANING / NORMALIZATION
# ============================================================

def clean(value):
    """Normalize PostgreSQL values before writing to MongoDB."""
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if value.lower() in {"nan", "none", "null"}:
            return None

        return value

    return value


def clean_text(value):
    value = clean(value)

    if value is None:
        return ""

    return str(value).strip()


def media_url(value):
    """
    Convert a relative media path into the production CloudFront URL.
    Existing absolute URLs are preserved.
    """
    value = clean_text(value)

    if not value:
        return None

    if value.startswith("http://") or value.startswith("https://"):
        return value

    return f"{CDN_BASE_URL}/{value.lstrip('/')}"


def html_to_text(value):
    """
    Convert HTML-rich profile content into clean searchable text.
    """
    value = clean_text(value)

    if not value:
        return ""

    value = re.sub(
        r"<\s*(br|p|/p|div|/div|li|/li|h[1-6]|/h[1-6])[^>]*>",
        " ",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def make_name(first_name, last_name):
    first = clean_text(first_name)
    last = clean_text(last_name)

    return " ".join(x for x in [first, last] if x).strip()


def object_ref(row_id, name):
    row_id = clean(row_id)
    name = clean(name)

    if row_id is None and not name:
        return None

    return {
        "id": row_id,
        "name": name,
    }


def normalize_json_array(value):
    """
    PostgreSQL jsonb comes through psycopg2 as Python lists/dicts.
    """
    if not value:
        return []

    result = []

    for item in value:
        if not item:
            continue

        item_id = clean(item.get("id"))
        name = clean(item.get("name"))

        if item_id is None and not name:
            continue

        result.append({
            "id": item_id,
            "name": name,
        })

    return result


def normalize_relation_array(value):
    """
    Normalize richer JSONB relation arrays while preserving all fields.
    """
    if not value:
        return []

    result = []

    for item in value:
        if not item:
            continue

        cleaned = {}

        for key, item_value in item.items():
            if isinstance(item_value, dict):
                cleaned[key] = normalize_relation_object(item_value)
            else:
                cleaned[key] = clean(item_value)

        # Remove completely empty relation objects.
        if any(
            v is not None and v != "" and v != [] and v != {}
            for v in cleaned.values()
        ):
            result.append(cleaned)

    return result


def normalize_relation_object(value):
    if not value:
        return None

    result = {}

    for key, item_value in value.items():
        if isinstance(item_value, dict):
            result[key] = normalize_relation_object(item_value)
        else:
            result[key] = clean(item_value)

    return result


def join_names(items):
    return ", ".join(
        item["name"]
        for item in items
        if item.get("name")
    )


def add_text(parts, label, value):
    value = clean_text(value)

    if value:
        parts.append(f"{label}: {value}")


def add_relation_text(parts, label, items):
    """
    Add useful searchable text from education, experience,
    certifications, awards and testimonials.
    """
    if not items:
        return

    for item in items:
        values = []

        scalar_keys = (
            "institution_name",
            "course_name",
            "course_type",
            "grade",
            "description",
            "company_name",
            "job_designation",
            "location",
            "position_overview",
            "name",
            "organization",
            "designation",
            "company",
            "content",
        )

        for key in scalar_keys:
            value = item.get(key)

            if value:
                if key in {"description", "position_overview", "content"}:
                    value = html_to_text(value)

                value = clean_text(value)

                if value:
                    values.append(value)

        for nested_key in (
            "education_level",
            "specialization",
            "country",
            "company",
            "job_role",
            "job_level",
            "industry",
            "department",
        ):
            nested = item.get(nested_key)

            if not isinstance(nested, dict):
                continue

            name = clean_text(nested.get("name"))

            if name:
                values.append(name)

            if nested_key == "specialization":
                description = html_to_text(
                    nested.get("description")
                )

                if description:
                    values.append(description)

        if item.get("is_currently_studying"):
            values.append("Currently studying")

        if item.get("is_currently_working"):
            values.append("Currently working")

        text = " ".join(
            value for value in values if value
        ).strip()

        if text:
            parts.append(f"{label}: {text}")


# ============================================================
# POSTGRESQL QUERY
# ============================================================

PROFESSIONAL_QUERY = """
SELECT

    p.useraccount_ptr_id,

    -- ========================================================
    -- USER
    -- ========================================================

    u.first_name,
    u.last_name,
    u.user_type,

    u.avatar,
    u.cover,

    u.city_town,

    u.current_country_id,
    cc.name AS current_country_name,
    cc.country_code AS current_country_code,
    cc.code AS current_country_iso,

    u.nationality_id,
    nc.name AS nationality_name,
    nc.country_code AS nationality_country_code,
    nc.code AS nationality_iso,

    u.about_us,
    u.tagline,
    u.slug,

    u.verified,
    u.is_pro,
    u.is_featured,
    u.is_active,
    u.is_working,

    u.created_at,

    -- ========================================================
    -- PROFESSIONAL
    -- ========================================================

    p.department_id,
    d.name AS department_name,

    p.job_level_id,
    jl.name AS job_level_name,

    p.job_role_id,
    jr.name AS job_role_name,

    p.education_level_id,
    el.name AS education_level_name,

    p.currently_working,
    p.current_company_id,
    p.current_company_text,
    p.resume_title,

    -- ========================================================
    -- CURRENT COMPANY
    -- ========================================================

    c.name AS company_name,
    c.created_by AS company_created_by,
    c.current_designation AS company_current_designation,
    c.website_link AS company_website,

    cu.avatar AS company_avatar,
    cu.cover AS company_cover,

    cu.city_town AS company_city,

    cu.current_country_id AS company_country_id,
    ccountry.name AS company_country_name,

    -- ========================================================
    -- SOCIAL LINKS
    -- ========================================================

    sl.facebook AS social_facebook,
    sl.twitter AS social_twitter,
    sl.linkedin AS social_linkedin,
    sl.instagram AS social_instagram,
    sl.youtube AS social_youtube,
    sl.hozpitality AS social_hozpitality,

    -- ========================================================
    -- SKILLS
    -- ========================================================

    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'id', s.id,
                    'name', s.name
                )
                ORDER BY s.name
            )
            FROM professionals_skills ps
            JOIN skills s
                ON s.id = ps.skills_id
            WHERE ps.professional_id = p.useraccount_ptr_id
        ),
        '[]'::jsonb
    ) AS skills,

    -- ========================================================
    -- LANGUAGES
    -- ========================================================

    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'id', l.id,
                    'name', l.name
                )
                ORDER BY l.name
            )
            FROM professionals_language_know pl
            JOIN languages l
                ON l.id = pl.language_id
            WHERE pl.professional_id = p.useraccount_ptr_id
        ),
        '[]'::jsonb
    ) AS languages,

    -- ========================================================
    -- INDUSTRIES
    -- ========================================================

    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'id', i.id,
                    'name', i.name
                )
                ORDER BY i.name
            )
            FROM user_accounts_industry uai
            JOIN industries i
                ON i.id = uai.industry_id
            WHERE uai.useraccount_id = p.useraccount_ptr_id
        ),
        '[]'::jsonb
    ) AS industries,

    -- ========================================================
    -- EDUCATION
    -- ========================================================

    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'id', e.id,
                    'institution_name', e.institution_name,
                    'course_name', e.course_name,

                    'education_level',
                    CASE
                        WHEN e.education_level_id IS NOT NULL
                        THEN jsonb_build_object(
                            'id', e.education_level_id,
                            'name', el2.name
                        )
                        ELSE NULL
                    END,

                    'specialization',
                    CASE
                        WHEN e.specialization_id IS NOT NULL
                        THEN jsonb_build_object(
                            'id', e.specialization_id,
                            'name', sp.name,
                            'description', sp.description
                        )
                        ELSE NULL
                    END,

                    'course_type', e.course_type,

                    'country',
                    CASE
                        WHEN e.country_id IS NOT NULL
                        THEN jsonb_build_object(
                            'id', e.country_id,
                            'name', ec.name,
                            'country_code', ec.country_code,
                            'code', ec.code
                        )
                        ELSE NULL
                    END,

                    'is_currently_studying', e.is_currently_studying,
                    'start_date', e.start_date,
                    'end_date', e.end_date,
                    'duration', e.duration,
                    'grade', e.grade,
                    'description', e.description,
                    'certificate_upload', e.certificate_upload
                )
                ORDER BY e.start_date DESC NULLS LAST, e.id DESC
            )
            FROM base_education e
            LEFT JOIN education_level el2
                ON el2.id = e.education_level_id
            LEFT JOIN base_specialization sp
                ON sp.id = e.specialization_id
            LEFT JOIN countries ec
                ON ec.id = e.country_id
            WHERE e.user_id = p.useraccount_ptr_id
        ),
        '[]'::jsonb
    ) AS education,

    -- ========================================================
    -- EXPERIENCE
    -- ========================================================

    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'id', ex.id,

                    'company',
                    CASE
                        WHEN ex.company_id IS NOT NULL
                        THEN jsonb_build_object(
                            'id', ex.company_id,
                            'name', ex_company.name
                        )
                        ELSE NULL
                    END,

                    'company_name', ex.company_name,
                    'job_designation', ex.job_designation,

                    'job_role',
                    CASE
                        WHEN ex.job_role_id IS NOT NULL
                        THEN jsonb_build_object(
                            'id', ex.job_role_id,
                            'name', exjr.name
                        )
                        ELSE NULL
                    END,

                    'job_level',
                    CASE
                        WHEN ex.job_level_id IS NOT NULL
                        THEN jsonb_build_object(
                            'id', ex.job_level_id,
                            'name', exjl.name
                        )
                        ELSE NULL
                    END,

                    'industry',
                    CASE
                        WHEN ex.industry_id IS NOT NULL
                        THEN jsonb_build_object(
                            'id', ex.industry_id,
                            'name', exi.name
                        )
                        ELSE NULL
                    END,

                    'department',
                    CASE
                        WHEN ex.department_id IS NOT NULL
                        THEN jsonb_build_object(
                            'id', ex.department_id,
                            'name', exd.name
                        )
                        ELSE NULL
                    END,

                    'location', ex.location,
                    'is_currently_working', ex.is_currently_working,
                    'start_date', ex.start_date,
                    'end_date', ex.end_date,
                    'position_overview', ex.position_overview
                )
                ORDER BY ex.start_date DESC NULLS LAST, ex.id DESC
            )
            FROM base_experience ex

            LEFT JOIN companies ex_company
                ON ex_company.useraccount_ptr_id = ex.company_id

            LEFT JOIN job_role exjr
                ON exjr.id = ex.job_role_id

            LEFT JOIN job_levels exjl
                ON exjl.id = ex.job_level_id

            LEFT JOIN industries exi
                ON exi.id = ex.industry_id

            LEFT JOIN departments exd
                ON exd.id = ex.department_id

            WHERE ex.user_id = p.useraccount_ptr_id
        ),
        '[]'::jsonb
    ) AS experience,

    -- ========================================================
    -- CERTIFICATIONS
    -- ========================================================

    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'id', ca.id,
                    'name', ca.name,
                    'organization', ca.org_name,
                    'certificate_file', ca.c_file
                )
                ORDER BY ca.id DESC
            )
            FROM base_certificationaward ca
            WHERE ca.user_id = p.useraccount_ptr_id
        ),
        '[]'::jsonb
    ) AS certifications,

    -- ========================================================
    -- AWARDS & RECOGNITION
    -- ========================================================

    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'id', ar.id,
                    'name', ar.name,
                    'organization', ar.org_name,
                    'file', ar.c_file
                )
                ORDER BY ar.id DESC
            )
            FROM base_awardandrecognition ar
            WHERE ar.user_id = p.useraccount_ptr_id
        ),
        '[]'::jsonb
    ) AS awards,

    -- ========================================================
    -- TESTIMONIALS
    -- ========================================================

    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'id', t.id,
                    'name', t.name,
                    'designation', t.designation,
                    'company', t.company,
                    'content', t.content,
                    'photo', t.photo
                )
                ORDER BY t.id DESC
            )
            FROM base_testimonial t
            WHERE t.user_id = p.useraccount_ptr_id
        ),
        '[]'::jsonb
    ) AS testimonials

FROM professionals p

JOIN user_accounts u
    ON u.id = p.useraccount_ptr_id

LEFT JOIN countries cc
    ON cc.id = u.current_country_id

LEFT JOIN countries nc
    ON nc.id = u.nationality_id

LEFT JOIN departments d
    ON d.id = p.department_id

LEFT JOIN job_levels jl
    ON jl.id = p.job_level_id

LEFT JOIN job_role jr
    ON jr.id = p.job_role_id

LEFT JOIN education_level el
    ON el.id = p.education_level_id

LEFT JOIN companies c
    ON c.useraccount_ptr_id = p.current_company_id

LEFT JOIN user_accounts cu
    ON cu.id = c.useraccount_ptr_id

LEFT JOIN countries ccountry
    ON ccountry.id = cu.current_country_id

LEFT JOIN base_sociallinks sl
    ON sl.user_id = p.useraccount_ptr_id

WHERE
    u.user_type = 'professional'
    AND u.is_active = TRUE

ORDER BY
    p.useraccount_ptr_id
"""


# ============================================================
# DOCUMENT BUILDER
# ============================================================

def build_document(row):

    professional_id = row["useraccount_ptr_id"]

    name = make_name(
        row["first_name"],
        row["last_name"],
    )

    skills = normalize_json_array(row["skills"])
    languages = normalize_json_array(row["languages"])
    industries = normalize_json_array(row["industries"])

    education = normalize_relation_array(row["education"])
    experience = normalize_relation_array(row["experience"])
    certifications = normalize_relation_array(row["certifications"])
    awards = normalize_relation_array(row["awards"])
    testimonials = normalize_relation_array(row["testimonials"])

    current_company = None

    if row["current_company_id"]:
        current_company = {
            "id": row["current_company_id"],
            "name": clean(row["company_name"]),
            "created_by": clean(row["company_created_by"]),
            "current_designation": clean(
                row["company_current_designation"]
            ),
            "website": clean(row["company_website"]),
            "profile_image": media_url(row["company_avatar"]),
            "cover_image": media_url(row["company_cover"]),
            "city": clean(row["company_city"]),
            "country": (
                {
                    "id": row["company_country_id"],
                    "name": clean(row["company_country_name"]),
                }
                if row["company_country_id"]
                else None
            ),
        }

    current_country = None

    if row["current_country_id"]:
        current_country = {
            "id": row["current_country_id"],
            "name": clean(row["current_country_name"]),
            "country_code": clean(row["current_country_code"]),
            "code": clean(row["current_country_iso"]),
        }

    nationality = None

    if row["nationality_id"]:
        nationality = {
            "id": row["nationality_id"],
            "name": clean(row["nationality_name"]),
            "country_code": clean(row["nationality_country_code"]),
            "code": clean(row["nationality_iso"]),
        }

    department = object_ref(
        row["department_id"],
        row["department_name"],
    )

    job_role = object_ref(
        row["job_role_id"],
        row["job_role_name"],
    )

    job_level = object_ref(
        row["job_level_id"],
        row["job_level_name"],
    )

    education_level = object_ref(
        row["education_level_id"],
        row["education_level_name"],
    )

    social_links = {
        "facebook": clean(row["social_facebook"]),
        "twitter": clean(row["social_twitter"]),
        "linkedin": clean(row["social_linkedin"]),
        "instagram": clean(row["social_instagram"]),
        "youtube": clean(row["social_youtube"]),
        "hozpitality": clean(row["social_hozpitality"]),
    }

    # Remove empty social-link values.
    social_links = {
        key: value
        for key, value in social_links.items()
        if value
    }

    # Convert relation media fields to CloudFront URLs.
    for item in education:
        if item.get("certificate_upload"):
            item["certificate_url"] = media_url(
                item.pop("certificate_upload")
            )
        else:
            item.pop("certificate_upload", None)

    for item in certifications:
        if item.get("certificate_file"):
            item["certificate_url"] = media_url(
                item.pop("certificate_file")
            )
        else:
            item.pop("certificate_file", None)

    for item in awards:
        if item.get("file"):
            item["file_url"] = media_url(
                item.pop("file")
            )
        else:
            item.pop("file", None)

    for item in testimonials:
        if item.get("photo"):
            item["photo"] = media_url(item["photo"])

    # ========================================================
    # AI SEARCH TEXT
    # ========================================================

    text = []

    add_text(text, "Name", name)
    add_text(text, "Professional", name)

    add_text(text, "Department", row["department_name"])
    add_text(text, "Job Role", row["job_role_name"])
    add_text(text, "Job Level", row["job_level_name"])
    add_text(text, "Education", row["education_level_name"])

    add_text(
        text,
        "Current Company",
        row["company_name"],
    )

    add_text(
        text,
        "Current Company Text",
        row["current_company_text"],
    )

    currently_working_text = clean_text(
        row["currently_working"]
    )

    if currently_working_text:
        add_text(
            text,
            "Currently Working",
            currently_working_text,
        )
    elif row["is_working"]:
        text.append("Currently working")

    add_text(
        text,
        "Resume Title",
        row["resume_title"],
    )

    add_text(
        text,
        "Skills",
        join_names(skills),
    )

    add_text(
        text,
        "Languages",
        join_names(languages),
    )

    add_text(
        text,
        "Industries",
        join_names(industries),
    )

    add_text(
        text,
        "City",
        row["city_town"],
    )

    add_text(
        text,
        "Country",
        row["current_country_name"],
    )

    add_text(
        text,
        "Nationality",
        row["nationality_name"],
    )

    about_text = html_to_text(row["about_us"])

    add_text(
        text,
        "About",
        about_text,
    )

    add_text(
        text,
        "Tagline",
        row["tagline"],
    )

    add_relation_text(
        text,
        "Education",
        education,
    )

    add_relation_text(
        text,
        "Experience",
        experience,
    )

    add_relation_text(
        text,
        "Certification",
        certifications,
    )

    add_relation_text(
        text,
        "Award",
        awards,
    )

    add_relation_text(
        text,
        "Testimonial",
        testimonials,
    )

    if row["verified"]:
        text.append("Verified professional")

    if row["is_pro"]:
        text.append("Pro professional")

    if row["is_featured"]:
        text.append("Featured professional")

    ai_search_text = ". ".join(text)

    # ========================================================
    # FINAL MONGO DOCUMENT
    # ========================================================

    document = {
        "_id": f"professional:{professional_id}",

        "entity_type": "professional",

        "slug": clean(row["slug"]),

        "source": {
            "model": "professional",
            "app_label": "base",
            "object_id": professional_id,
        },

        "title": name,

        "profile_image": media_url(row["avatar"]),
        "cover_image": media_url(row["cover"]),

        "user": {
            "id": professional_id,
            "name": name,
            "slug": clean(row["slug"]),
            "profile_image": media_url(row["avatar"]),
            "cover_image": media_url(row["cover"]),
            "user_type": "professional",
        },

        "professional": {
            "department": department,
            "job_role": job_role,
            "job_level": job_level,
            "education_level": education_level,

            "currently_working": clean(
                row["currently_working"]
            ),

            "current_company": current_company,

            "current_company_text": clean(
                row["current_company_text"]
            ),

            "resume_title": clean(
                row["resume_title"]
            ),

            "skills": skills,
            "languages": languages,
            "industries": industries,
        },

        "education": education,
        "experience": experience,
        "certifications": certifications,
        "awards": awards,
        "social_links": social_links,
        "testimonials": testimonials,

        "location": {
            "city": clean(row["city_town"]),
            "country": current_country,
            "nationality": nationality,
        },

        "metadata": {
            "verified": bool(row["verified"]),
            "is_pro": bool(row["is_pro"]),
            "is_featured": bool(row["is_featured"]),
            "is_working": bool(row["is_working"]),
            "is_active": bool(row["is_active"]),
        },

        "ai_search_text": ai_search_text,

        "is_live": bool(row["is_active"]),

        "created_at": row["created_at"],
    }

    return document


# ============================================================
# CONNECTIONS
# ============================================================

def connect_postgres():

    print("Connecting to PostgreSQL...")

    conn = psycopg2.connect(
        dbname=PG_DB,
        user=PG_USER,
        host=PG_HOST,
    )

    conn.set_session(
        readonly=True,
        autocommit=False,
    )

    print("PostgreSQL connected.")

    return conn


def connect_mongo():

    if not MONGO_URI:
        print()
        print("MONGO_URI is not set.")
        print(
            "Example:"
        )
        print(
            "export MONGO_URI='mongodb://mongoAdmin:<PASSWORD>@127.0.0.1:27017/?authSource=admin'"
        )
        print()
        sys.exit(1)

    print("Connecting to MongoDB...")

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=10000,
    )

    client.admin.command("ping")

    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    print(
        f"MongoDB connected: "
        f"{MONGO_DB}.{MONGO_COLLECTION}"
    )

    return client, collection


# ============================================================
# MIGRATION
# ============================================================

def migrate():

    global LIMIT

    args = sys.argv[1:]

    for index, arg in enumerate(args):

        if arg.startswith("--limit="):
            LIMIT = int(
                arg.split("=", 1)[1]
            )

        elif arg == "--limit" and index + 1 < len(args):
            LIMIT = int(args[index + 1])

    pg = connect_postgres()
    mongo_client, collection = connect_mongo()

    cursor = None

    try:

        cursor = pg.cursor(
            name="professional_migration_cursor",
            cursor_factory=RealDictCursor,
        )

        query = PROFESSIONAL_QUERY

        if LIMIT:
            query += f"\nLIMIT {LIMIT}"

        print()
        print("=" * 70)
        print("PROFESSIONAL MIGRATION")
        print("=" * 70)
        print(f"Batch size : {BATCH_SIZE}")
        print(
            f"Limit      : "
            f"{LIMIT if LIMIT else 'ALL ACTIVE PROFESSIONALS'}"
        )
        print("=" * 70)
        print()

        cursor.execute(query)

        total_processed = 0
        total_written = 0
        started = time.time()

        while True:

            rows = cursor.fetchmany(BATCH_SIZE)

            if not rows:
                break

            operations = []

            for row in rows:

                document = build_document(row)

                operations.append(
                    UpdateOne(
                        {"_id": document["_id"]},
                        {"$set": document},
                        upsert=True,
                    )
                )

            if operations:

                result = collection.bulk_write(
                    operations,
                    ordered=False,
                )

                total_written += (
                    result.upserted_count
                    + result.modified_count
                )

            total_processed += len(rows)

            elapsed = time.time() - started

            rate = (
                total_processed / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"Processed: {total_processed:,} | "
                f"Written: {total_written:,} | "
                f"Rate: {rate:,.0f}/sec"
            )

        elapsed = time.time() - started

        print()
        print("=" * 70)
        print("MIGRATION COMPLETE")
        print("=" * 70)
        print(f"Processed : {total_processed:,}")
        print(f"Written   : {total_written:,}")
        print(f"Time      : {elapsed:,.2f} sec")
        print(
            f"Rate      : "
            f"{total_processed / elapsed:,.0f}/sec"
        )
        print("=" * 70)

    finally:

        if cursor:
            cursor.close()

        pg.close()
        mongo_client.close()


if __name__ == "__main__":

    try:
        migrate()

    except KeyboardInterrupt:

        print()
        print("Migration interrupted.")

        sys.exit(130)

    except Exception as exc:

        print()
        print("MIGRATION FAILED")
        print(str(exc))

        raise
