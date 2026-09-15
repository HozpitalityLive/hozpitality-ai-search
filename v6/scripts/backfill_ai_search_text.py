"""
Backfill Hozpitality V6 canonical AI search documents.

Each MasterSearchIndex row becomes one "search document":
- ai_search_text: deterministic, human-readable flattened entity + relationship
  representation used by PostgreSQL FTS and later embeddings.
- metadata: structured relationship/attribute payload for exact filtering and
  future personalization.

The script is deliberately database-driven and does not call an LLM. Search
index content should be deterministic and cheap to rebuild when source data
changes.

Known Hozpitality entities receive relationship-aware enrichment:
professional, company, article, job, event, product, faq, award, category,
post. Other content types receive a safe generic source-row representation.

Usage:
  python scripts/backfill_ai_search_text.py --confirm
  python scripts/backfill_ai_search_text.py --model professional --confirm
  python scripts/backfill_ai_search_text.py --limit 1000 --batch-size 500 --confirm
  python scripts/backfill_ai_search_text.py --force --confirm
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2 import sql
from psycopg2.extras import Json, execute_values
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT, ".env"), override=True)

SENSITIVE = {
    "password", "password_hash", "token", "access_token", "refresh_token",
    "secret", "secret_key", "api_key", "private_key", "otp", "otp_code",
    "session_data", "key", "captcha_entered", "captcha_total", "wallet_balance",
    "total_credits", "messages_sent_this_month", "share_all_read_ids",
    "phone_number", "email",
}

SOURCE_TABLES = {
    "professional": ("professionals", "useraccount_ptr_id"),
    "company": ("companies", "useraccount_ptr_id"),
    "article": ("base_article", "id"),
    "job": ("base_job", "id"),
    "event": ("base_event", "id"),
    "product": ("marketplace_product", "id"),
    "faq": ("base_faq", "id"),
    "award": ("base_awards", "id"),
    "category": ("base_category", "id"),
    "post": ("base_post", "id"),
}


def connect():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        connect_timeout=int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10")),
    )


def clean(value: Any, limit: int = 6000) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        value = value.isoformat()
    elif isinstance(value, Decimal):
        value = str(value)
    elif isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def words(*values: Any, limit: int = 24000) -> str:
    parts = []
    seen = set()
    for value in values:
        text = clean(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(text)
    return " | ".join(parts)[:limit]


def compact_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: compact_metadata(v) for k, v in value.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [compact_metadata(v) for v in value if v not in (None, "", [], {})]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def db_rows(cur, query: str, params: list[Any] | tuple[Any, ...]):
    cur.execute(query, params)
    return [dict(r) for r in cur.fetchall()]


def fetch_master(cur, content_type_id: int, object_ids: list[int], force: bool):
    condition = "" if force else "AND COALESCE(si.ai_search_text, '') = ''"
    cur.execute(
        f"""
        SELECT si.id, si.object_id, si.content_type_id, si.entity_type,
               si.entity_name, si.title, si.subtitle, si.slug, si.content,
               si.ai_keywords, si.ai_summary, si.user_id, si.user_name,
               si.company_id, si.company_name, si.category_id, si.category_text,
               si.subcategory_id, si.subcategory_text, si.country_id,
               si.country_text, si.city_id, si.city_text, si.location_text,
               si.status, si.is_live, si.is_deleted, si.is_searchable,
               si.is_public, si.created_at, si.updated_at, si.published_at,
               si.expires_at
        FROM public.master_search_mastersearchindex si
        WHERE si.content_type_id = %s
          AND si.object_id = ANY(%s)
          {condition}
        ORDER BY si.id
        """,
        (content_type_id, object_ids),
    )
    return [dict(r) for r in cur.fetchall()]


def professional_rows(cur, ids):
    return db_rows(cur, """
        SELECT
            p.useraccount_ptr_id AS object_id,
            concat_ws(' ', u.first_name, u.last_name) AS full_name,
            u.about_us, u.tagline, u.city_town, u.slug AS user_slug,
            u.current_country_id,
            d.name AS department_name,
            jl.name AS job_level_name,
            jr.name AS job_role_name,
            el.name AS education_level_name,
            c.name AS current_company_name,
            COALESCE(sk.skills, '') AS skills,
            COALESCE(la.languages, '') AS languages,
            COALESCE(ind.industries, '') AS industries,
            COALESCE(exp.experience, '') AS experience
        FROM public.professionals p
        JOIN public.user_accounts u
          ON u.id = p.useraccount_ptr_id
        LEFT JOIN public.departments d ON d.id = p.department_id
        LEFT JOIN public.job_levels jl ON jl.id = p.job_level_id
        LEFT JOIN public.job_role jr ON jr.id = p.job_role_id
        LEFT JOIN public.education_level el ON el.id = p.education_level_id
        LEFT JOIN public.companies c ON c.useraccount_ptr_id = p.current_company_id
        LEFT JOIN public.countries co ON co.id = u.current_country_id
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT s.name, ', ' ORDER BY s.name) AS skills
            FROM public.professionals_skills ps
            JOIN public.skills s ON s.id = ps.skills_id
            WHERE ps.professional_id = p.useraccount_ptr_id
        ) sk ON TRUE
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT l.name, ', ' ORDER BY l.name) AS languages
            FROM public.professionals_language_know pl
            JOIN public.languages l ON l.id = pl.language_id
            WHERE pl.professional_id = p.useraccount_ptr_id
        ) la ON TRUE
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT i.name, ', ' ORDER BY i.name) AS industries
            FROM public.user_accounts_industry ui
            JOIN public.industries i ON i.id = ui.industry_id
            WHERE ui.useraccount_id = p.useraccount_ptr_id
        ) ind ON TRUE
        LEFT JOIN LATERAL (
            SELECT string_agg(
                DISTINCT concat_ws(
                    ' — ',
                    e.job_designation,
                    e.company_name,
                    e.location,
                    d2.name,
                    jr2.name,
                    i2.name
                ),
                ' | ' ORDER BY concat_ws(
                    ' — ',
                    e.job_designation,
                    e.company_name,
                    e.location,
                    d2.name,
                    jr2.name,
                    i2.name
                )
            ) AS experience
            FROM public.base_experience e
            LEFT JOIN public.departments d2 ON d2.id = e.department_id
            LEFT JOIN public.job_role jr2 ON jr2.id = e.job_role_id
            LEFT JOIN public.industries i2 ON i2.id = e.industry_id
            WHERE e.user_id = p.useraccount_ptr_id
        ) exp ON TRUE
        WHERE p.useraccount_ptr_id = ANY(%s)
    """, (ids,))


def company_rows(cur, ids):
    return db_rows(cur, """
        SELECT
            c.useraccount_ptr_id AS object_id,
            c.name,
            c.created_by,
            c.current_designation,
            c.website_link,
            u.about_us,
            u.tagline,
            u.city_town,
            u.current_country_id,
            u.slug AS user_slug,
            COALESCE(ind.industries, '') AS industries
        FROM public.companies c
        JOIN public.user_accounts u ON u.id = c.useraccount_ptr_id
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT i.name, ', ' ORDER BY i.name) AS industries
            FROM public.user_accounts_industry ui
            JOIN public.industries i ON i.id = ui.industry_id
            WHERE ui.useraccount_id = c.useraccount_ptr_id
        ) ind ON TRUE
        WHERE c.useraccount_ptr_id = ANY(%s)
    """, (ids,))


def article_rows(cur, ids):
    return db_rows(cur, """
        SELECT
            a.id AS object_id,
            a.title, a.sub_title, a.content, a.slug,
            a.company_id, a.category_id,
            c.name AS category_name,
            COALESCE(co.countries, '') AS countries
        FROM public.base_article a
        LEFT JOIN public.base_category c ON c.id = a.category_id
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT co1.name, ', ' ORDER BY co1.name) AS countries
            FROM public.base_article_location al
            JOIN public.countries co1 ON co1.id = al.country_id
            WHERE al.article_id = a.id
        ) co ON TRUE
        WHERE a.id = ANY(%s)
    """, (ids,))


def job_rows(cur, ids):
    return db_rows(cur, """
        SELECT
            j.id AS object_id,
            j.job_title, j.job_desc, j.job_city, j.job_address, j.job_status,
            j.salary_description, j.reference, j.posted_by,
            j.company_id, c.name AS company_name,
            j.job_country_id, co.name AS country_name,
            et.name AS employment_type,
            jt.name AS job_type,
            sr.name AS salary_range,
            cur.name AS currency_name,
            COALESCE(role.roles, '') AS roles,
            COALESCE(dep.departments, '') AS departments,
            COALESCE(lvl.levels, '') AS levels,
            COALESCE(ind.industries, '') AS industries
        FROM public.base_job j
        LEFT JOIN public.companies c ON c.useraccount_ptr_id = j.company_id
        LEFT JOIN public.countries co ON co.id = j.job_country_id
        LEFT JOIN public.employment_type et ON et.id = j.employementtype_id
        LEFT JOIN public.job_type jt ON jt.id = j.jobtype_id
        LEFT JOIN public.base_salaryrange sr ON sr.id = j."salaryRange_id"
        LEFT JOIN public.base_currency cur ON cur.id = j.currency_id
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT r.name, ', ' ORDER BY r.name) AS roles
            FROM public.base_job_job_role x
            JOIN public.job_role r ON r.id = x.jobrole_id
            WHERE x.job_id = j.id
        ) role ON TRUE
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT d.name, ', ' ORDER BY d.name) AS departments
            FROM public.base_job_job_department x
            JOIN public.departments d ON d.id = x.department_id
            WHERE x.job_id = j.id
        ) dep ON TRUE
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT l.name, ', ' ORDER BY l.name) AS levels
            FROM public.base_job_job_level x
            JOIN public.job_levels l ON l.id = x.joblevel_id
            WHERE x.job_id = j.id
        ) lvl ON TRUE
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT i.name, ', ' ORDER BY i.name) AS industries
            FROM public.base_job_job_industry x
            JOIN public.industries i ON i.id = x.industry_id
            WHERE x.job_id = j.id
        ) ind ON TRUE
        WHERE j.id = ANY(%s)
    """, (ids,))


def event_rows(cur, ids):
    return db_rows(cur, """
        SELECT
            e.id AS object_id, e.title, e.details, e.address, e.city,
            e.event_type, e.website, e.status, e.start_datetime, e.end_datetime,
            e.company_id, c.name AS company_name,
            e.country_id, co.name AS country_name,
            e.slug
        FROM public.base_event e
        LEFT JOIN public.companies c ON c.useraccount_ptr_id = e.company_id
        LEFT JOIN public.countries co ON co.id = e.country_id
        WHERE e.id = ANY(%s)
    """, (ids,))


def product_rows(cur, ids):
    return db_rows(cur, """
        SELECT
            p.id AS object_id, p.title, p.description, p.keywords,
            p.current_location, p.prime_city, p.other_location,
            p.p_condition, p.status, p.price, p.discounted_price,
            p.posted_by_id, u.username AS posted_by,
            p.country_id, co.name AS country_name,
            COALESCE(cat.categories, '') AS categories
        FROM public.marketplace_product p
        LEFT JOIN public.user_accounts u ON u.id = p.posted_by_id
        LEFT JOIN public.countries co ON co.id = p.country_id
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT pc.name, ', ' ORDER BY pc.name) AS categories
            FROM public.marketplace_product_category x
            JOIN public.marketplace_productcategory pc ON pc.id = x.productcategory_id
            WHERE x.product_id = p.id
        ) cat ON TRUE
        WHERE p.id = ANY(%s)
    """, (ids,))


def faq_rows(cur, ids):
    return db_rows(cur, """
        SELECT id AS object_id, question, answer, is_popular, show_on_landing
        FROM public.base_faq
        WHERE id = ANY(%s)
    """, (ids,))


def award_rows(cur, ids):
    return db_rows(cur, """
        SELECT
            a.id AS object_id, a.award_title, a.award_description,
            a.award_short_title, a.award_subtitle, a.location,
            a.country_id, co.name AS country_name, a.award_year,
            a.award_is_active, a.is_voting_active, a.slug,
            COALESCE(cat.categories, '') AS categories
        FROM public.base_awards a
        LEFT JOIN public.countries co ON co.id = a.country_id
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT ac.category_name, ', ' ORDER BY ac.category_name) AS categories
            FROM public.base_awards_award_category x
            JOIN public.base_awardcategory ac ON ac.id = x.awardcategory_id
            WHERE x.awards_id = a.id
        ) cat ON TRUE
        WHERE a.id = ANY(%s)
    """, (ids,))


def category_rows(cur, ids):
    return db_rows(cur, """
        SELECT id AS object_id, name, db_id
        FROM public.base_category
        WHERE id = ANY(%s)
    """, (ids,))


def post_rows(cur, ids):
    return db_rows(cur, """
        SELECT p.id AS object_id, p.post_type, p.content, p.user_id,
               concat_ws(' ', u.first_name, u.last_name) AS author_name
        FROM public.base_post p
        LEFT JOIN public.user_accounts u ON u.id = p.user_id
        WHERE p.id = ANY(%s)
    """, (ids,))


def generic_source(cur, model: str, ids: list[int]):
    """Safely fetch scalar source values for an unknown model."""
    table = None
    pk = "id"
    if model in SOURCE_TABLES:
        table, pk = SOURCE_TABLES[model]
    else:
        # Only accept exact/singular table candidates discovered from
        # information_schema. This is a fallback, not a guessed join.
        compact = re.sub(r"[^a-z0-9]", "", model.lower())
        cur.execute("""
            SELECT table_name, column_name
            FROM information_schema.key_column_usage
            WHERE table_schema='public'
              AND constraint_name IN (
                  SELECT constraint_name
                  FROM information_schema.table_constraints
                  WHERE table_schema='public'
                    AND constraint_type='PRIMARY KEY'
              )
              AND (
                  regexp_replace(lower(table_name), '[^a-z0-9]+', '', 'g') = %s
                  OR regexp_replace(lower(table_name), '[^a-z0-9]+', '', 'g') = %s
              )
            ORDER BY CASE
                WHEN regexp_replace(lower(table_name), '[^a-z0-9]+', '', 'g') = %s THEN 0
                ELSE 1 END
            LIMIT 1
        """, (compact, compact.rstrip("s"), compact))
        found = cur.fetchone()
        if found:
            table, pk = found["table_name"], found["column_name"]

    if not table:
        return {}

    cur.execute(
        sql.SQL("SELECT * FROM public.{} WHERE {} = ANY(%s)").format(
            sql.Identifier(table), sql.Identifier(pk)
        ),
        (ids,),
    )
    rows = {}
    for row in cur.fetchall():
        row = dict(row)
        safe = {}
        for key, value in row.items():
            low = key.lower()
            if low in SENSITIVE or any(x in low for x in ("password", "secret", "token", "session")):
                continue
            if low.endswith("_id") or low == "id":
                continue
            text = clean(value, 3000)
            if text:
                safe[key] = text
        rows[int(row[pk])] = safe
    return rows


FETCHERS = {
    "professional": professional_rows,
    "company": company_rows,
    "article": article_rows,
    "job": job_rows,
    "event": event_rows,
    "product": product_rows,
    "faq": faq_rows,
    "award": award_rows,
    "category": category_rows,
    "post": post_rows,
}


def build_document(master: dict[str, Any], source: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    model = clean(master.get("entity_type") or "").lower() or "record"
    fields = [
        ("Entity", master.get("entity_name") or model),
        ("Title", master.get("title")),
        ("Subtitle", master.get("subtitle")),
        ("Name", master.get("user_name")),
        ("Company", master.get("company_name")),
        ("Category", master.get("category_text")),
        ("Subcategory", master.get("subcategory_text")),
        ("Country", master.get("country_text")),
        ("City", master.get("city_text")),
        ("Location", master.get("location_text")),
        ("Keywords", master.get("ai_keywords")),
        ("Summary", master.get("ai_summary")),
        ("Content", master.get("content")),
    ]

    relationship_meta: dict[str, Any] = {}

    if model == "professional":
        relationship_meta = {
            "job_role": source.get("job_role_name"),
            "department": source.get("department_name"),
            "job_level": source.get("job_level_name"),
            "education_level": source.get("education_level_name"),
            "current_company": source.get("current_company_name"),
            "skills": source.get("skills"),
            "languages": source.get("languages"),
            "industries": source.get("industries"),
            "experience": source.get("experience"),
            "profile": source.get("about_us"),
            "tagline": source.get("tagline"),
            "city": source.get("city_town"),
        }
        fields += [
            ("Job Role", source.get("job_role_name")),
            ("Department", source.get("department_name")),
            ("Job Level", source.get("job_level_name")),
            ("Education", source.get("education_level_name")),
            ("Current Company", source.get("current_company_name")),
            ("Skills", source.get("skills")),
            ("Languages", source.get("languages")),
            ("Industries", source.get("industries")),
            ("Experience", source.get("experience")),
            ("Profile", source.get("about_us")),
            ("Tagline", source.get("tagline")),
            ("City", source.get("city_town")),
        ]
        if source.get("current_country_id"):
            relationship_meta["country_id"] = source["current_country_id"]

    elif model == "company":
        relationship_meta = {
            "company_name": source.get("name"),
            "about": source.get("about_us"),
            "tagline": source.get("tagline"),
            "industries": source.get("industries"),
            "city": source.get("city_town"),
            "website": source.get("website_link"),
        }
        fields += [
            ("Company Name", source.get("name")),
            ("About", source.get("about_us")),
            ("Tagline", source.get("tagline")),
            ("Industries", source.get("industries")),
            ("City", source.get("city_town")),
            ("Website", source.get("website_link")),
            ("Designation", source.get("current_designation")),
        ]

    elif model == "article":
        relationship_meta = {
            "category": source.get("category_name"),
            "countries": source.get("countries"),
            "company_id": source.get("company_id"),
        }
        fields += [
            ("Article Category", source.get("category_name")),
            ("Article Countries", source.get("countries")),
            ("Article Content", source.get("content")),
        ]

    elif model == "job":
        relationship_meta = {
            "company": source.get("company_name"),
            "country": source.get("country_name"),
            "roles": source.get("roles"),
            "departments": source.get("departments"),
            "levels": source.get("levels"),
            "industries": source.get("industries"),
            "employment_type": source.get("employment_type"),
            "job_type": source.get("job_type"),
            "salary_range": source.get("salary_range"),
            "currency": source.get("currency_name"),
        }
        fields += [
            ("Job Role", source.get("roles")),
            ("Department", source.get("departments")),
            ("Job Level", source.get("levels")),
            ("Industry", source.get("industries")),
            ("Employment Type", source.get("employment_type")),
            ("Job Type", source.get("job_type")),
            ("Salary Range", source.get("salary_range")),
            ("Currency", source.get("currency_name")),
            ("Job Description", source.get("job_desc")),
            ("Job City", source.get("job_city")),
            ("Job Address", source.get("job_address")),
            ("Company", source.get("company_name")),
            ("Country", source.get("country_name")),
        ]

    elif model == "event":
        relationship_meta = {
            "company": source.get("company_name"),
            "country": source.get("country_name"),
            "event_type": source.get("event_type"),
            "city": source.get("city"),
        }
        fields += [
            ("Event Type", source.get("event_type")),
            ("Company", source.get("company_name")),
            ("Country", source.get("country_name")),
            ("City", source.get("city")),
            ("Address", source.get("address")),
            ("Details", source.get("details")),
        ]

    elif model == "product":
        relationship_meta = {
            "categories": source.get("categories"),
            "country": source.get("country_name"),
            "posted_by": source.get("posted_by"),
            "city": source.get("prime_city"),
            "condition": source.get("p_condition"),
        }
        fields += [
            ("Product Categories", source.get("categories")),
            ("Description", source.get("description")),
            ("Keywords", source.get("keywords")),
            ("Current Location", source.get("current_location")),
            ("Prime City", source.get("prime_city")),
            ("Other Locations", source.get("other_location")),
            ("Country", source.get("country_name")),
            ("Condition", source.get("p_condition")),
            ("Posted By", source.get("posted_by")),
        ]

    elif model == "faq":
        relationship_meta = {
            "popular": source.get("is_popular"),
            "show_on_landing": source.get("show_on_landing"),
        }
        fields += [
            ("Question", source.get("question")),
            ("Answer", source.get("answer")),
        ]

    elif model == "award":
        relationship_meta = {
            "categories": source.get("categories"),
            "country": source.get("country_name"),
            "year": source.get("award_year"),
        }
        fields += [
            ("Award", source.get("award_title")),
            ("Award Subtitle", source.get("award_subtitle")),
            ("Award Short Title", source.get("award_short_title")),
            ("Description", source.get("award_description")),
            ("Categories", source.get("categories")),
            ("Country", source.get("country_name")),
            ("Location", source.get("location")),
            ("Year", source.get("award_year")),
        ]

    elif model == "category":
        relationship_meta = {"db_id": source.get("db_id")}
        fields += [("Category Name", source.get("name"))]

    elif model == "post":
        relationship_meta = {
            "post_type": source.get("post_type"),
            "author": source.get("author_name"),
        }
        fields += [
            ("Post Type", source.get("post_type")),
            ("Author", source.get("author_name")),
            ("Post Content", source.get("content")),
        ]

    else:
        # Generic source fields are intentionally included only for unknown
        # entities; known entities use curated relationship-aware content.
        for key, value in source.items():
            fields.append((key.replace("_", " ").title(), value))
        relationship_meta = {"source_fields": source}

    document_parts = []
    for label, value in fields:
        text = clean(value, 12000)
        if text:
            document_parts.append(f"{label}: {text}")

    document = "\n".join(document_parts)[:24000]

    metadata = {
        "schema_version": "v6.5",
        "entity_type": model,
        "content_type_id": master.get("content_type_id"),
        "object_id": master.get("object_id"),
        "entity_name": master.get("entity_name"),
        "user_id": master.get("user_id"),
        "company_id": master.get("company_id"),
        "category_id": master.get("category_id"),
        "subcategory_id": master.get("subcategory_id"),
        "country_id": master.get("country_id"),
        "city_id": master.get("city_id"),
        "status": master.get("status"),
        "is_live": master.get("is_live"),
        "is_public": master.get("is_public"),
        "is_searchable": master.get("is_searchable"),
        "relationships": compact_metadata(relationship_meta),
    }
    return document, compact_metadata(metadata)


def resolve_content_types(cur, model_filter: str | None):
    if model_filter:
        cur.execute("""
            SELECT DISTINCT ct.id, ct.app_label, ct.model
            FROM public.django_content_type ct
            JOIN public.master_search_mastersearchindex si
              ON si.content_type_id = ct.id
            WHERE lower(ct.model) = lower(%s)
            ORDER BY CASE WHEN lower(ct.app_label) = 'base' THEN 0 ELSE 1 END, ct.id
        """, (model_filter,))
    else:
        cur.execute("""
            SELECT DISTINCT ct.id, ct.app_label, ct.model
            FROM public.django_content_type ct
            JOIN public.master_search_mastersearchindex si
              ON si.content_type_id = ct.id
            ORDER BY CASE WHEN lower(ct.app_label) = 'base' THEN 0 ELSE 1 END, ct.id
        """)
    return [dict(r) for r in cur.fetchall()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--model", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-fts", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    if not args.confirm:
        raise SystemExit("Refusing production backfill without --confirm.")

    conn = connect()
    conn.autocommit = False
    processed = 0

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema='public'
                  AND table_name='master_search_mastersearchindex'
                  AND column_name IN ('ai_search_text', 'metadata', 'search_vector_v6')
            """)
            columns = {r["column_name"] for r in cur.fetchall()}
            missing = {"ai_search_text", "metadata"} - columns
            if missing:
                raise SystemExit(
                    "Missing columns: " + ", ".join(sorted(missing)) +
                    ". Run sql/013_v6_ai_search_document.sql first."
                )

            content_types = resolve_content_types(cur, args.model or None)

        for ct in content_types:
            ct_id = int(ct["id"])
            model = str(ct["model"] or "").lower()

            last_master_id = 0
            model_processed = 0

            while True:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Select a bounded set of master IDs, then fetch their
                    # source rows in one relationship-aware query.
                    condition = "" if args.force else "AND COALESCE(ai_search_text, '') = ''"
                    cur.execute(
                        f"""
                        SELECT id, object_id
                        FROM public.master_search_mastersearchindex
                        WHERE content_type_id = %s
                          AND id > %s
                          {condition}
                        ORDER BY id
                        LIMIT %s
                        """,
                        (ct_id, last_master_id, max(1, min(args.batch_size, 2000))),
                    )
                    selected = [dict(r) for r in cur.fetchall()]

                if not selected:
                    break

                object_ids = list(dict.fromkeys(int(r["object_id"]) for r in selected))

                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    masters = fetch_master(cur, ct_id, object_ids, args.force)

                    fetcher = FETCHERS.get(model)
                    if fetcher:
                        source_rows = {
                            int(r["object_id"]): r
                            for r in fetcher(cur, object_ids)
                        }
                    else:
                        source_rows = generic_source(cur, model, object_ids)

                values = []
                for master in masters:
                    source = source_rows.get(int(master["object_id"]), {})
                    document, metadata = build_document(master, source)
                    values.append((
                        int(master["id"]),
                        document,
                        Json(metadata),
                    ))

                if values:
                    with conn.cursor() as cur:
                        execute_values(
                            cur,
                            """
                            UPDATE public.master_search_mastersearchindex AS t
                            SET ai_search_text = v.ai_search_text,
                                metadata = v.metadata::jsonb,
                                indexed_at = CURRENT_TIMESTAMP
                            FROM (VALUES %s) AS v(id, ai_search_text, metadata)
                            WHERE t.id = v.id
                            """,
                            values,
                            template="(%s, %s, %s)",
                            page_size=250,
                        )
                    conn.commit()

                last_master_id = int(selected[-1]["id"])
                model_processed += len(values)
                processed += len(values)

                print(
                    f"model={model or 'unknown'} processed={model_processed} "
                    f"total={processed} last_id={last_master_id}",
                    flush=True,
                )

                if args.limit and processed >= args.limit:
                    break
                if args.sleep:
                    time.sleep(args.sleep)

            if args.limit and processed >= args.limit:
                break

    finally:
        conn.close()

    if args.rebuild_fts:
        rebuild_fts()
    print(f"Done. Canonical search documents generated: {processed}")


def rebuild_fts():
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE public.master_search_mastersearchindex
                SET search_vector_v6 =
                      setweight(to_tsvector('simple', unaccent(coalesce(title, ''))), 'A')
                    || setweight(to_tsvector('simple', unaccent(coalesce(entity_name, ''))), 'A')
                    || setweight(to_tsvector('simple', unaccent(coalesce(category_text, ''))), 'A')
                    || setweight(to_tsvector('simple', unaccent(coalesce(company_name, ''))), 'A')
                    || setweight(to_tsvector('simple', unaccent(coalesce(user_name, ''))), 'A')
                    || setweight(to_tsvector('simple', unaccent(coalesce(subcategory_text, ''))), 'B')
                    || setweight(to_tsvector('simple', unaccent(coalesce(country_text, ''))), 'B')
                    || setweight(to_tsvector('simple', unaccent(coalesce(city_text, ''))), 'B')
                    || setweight(to_tsvector('simple', unaccent(coalesce(location_text, ''))), 'B')
                    || setweight(to_tsvector('simple', unaccent(coalesce(ai_keywords, ''))), 'B')
                    || setweight(to_tsvector('simple', unaccent(coalesce(slug, ''))), 'C')
                    || setweight(to_tsvector('simple', unaccent(coalesce(ai_summary, ''))), 'C')
                    || setweight(to_tsvector('simple', unaccent(coalesce(content, ''))), 'C')
                    || setweight(to_tsvector('simple', unaccent(coalesce(ai_search_text, ''))), 'B')
            """)
        conn.commit()
        print("FTS rebuild complete.", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
