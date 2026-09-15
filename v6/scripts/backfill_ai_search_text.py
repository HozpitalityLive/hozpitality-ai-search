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


def fetch_master(
    cur,
    content_type_id: int,
    object_ids: list[int],
    force: bool,
):
    condition = (
        ""
        if force
        else "AND COALESCE(si.ai_search_text, '') = ''"
    )

    cur.execute(
        f"""
        SELECT
            si.id,
            si.object_id,
            si.content_type_id,
            si.title,
            si.location_text,
            si.category_text,
            si.ai_keywords,
            si.user_name,
            si.content,
            si.slug,
            si.is_live,
            si.created_at,
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
    """
    Fetch Professional + UserAccount + all important relationships.

    Includes:
      - Department
      - Job Level
      - Job Role
      - Education
      - Current Company
      - Skills
      - Languages
      - Industries
      - Supplier Industries
      - Supplier Categories
      - Experience
      - Country
      - Nationality
      - Profile information
    """

    return db_rows(
        cur,
        """
        SELECT

            p.useraccount_ptr_id AS object_id,

            -- User
            concat_ws(
                ' ',
                NULLIF(u.first_name, ''),
                NULLIF(u.last_name, '')
            ) AS full_name,

            u.username,
            u.about_us,
            u.tagline,
            u.city_town,
            u.address,
            u.postal_code,
            u.slug AS user_slug,

            u.current_country_id,
            current_country.name AS current_country_name,
            current_country.country_code AS current_country_code,

            u.nationality_id,
            nationality.name AS nationality_name,

            u.user_type,
            u.verified,
            u.is_pro,
            u.is_working,
            u.is_featured,

            u.company_name AS account_company_name,

            -- Professional
            p.department_id,
            d.name AS department_name,

            p.job_level_id,
            jl.name AS job_level_name,

            p.education_level_id,
            el.name AS education_level_name,

            p.job_role_id,
            jr.name AS job_role_name,

            p.currently_working,

            p.current_company_id,
            c.name AS current_company_name,

            p.current_company_text,
            p.resume_title,

            -- Skills
            COALESCE(
                sk.skills,
                ''
            ) AS skills,

            -- Languages
            COALESCE(
                la.languages,
                ''
            ) AS languages,

            -- ALL Industries
            COALESCE(
                ind.industries,
                ''
            ) AS industries,

            -- Supplier Industries
            COALESCE(
                supplier_ind.industries,
                ''
            ) AS supplier_industries,

            -- Supplier Categories
            COALESCE(
                supplier_cat.supplier_categories,
                ''
            ) AS supplier_categories,

            -- Experience
            COALESCE(
                exp.experience,
                ''
            ) AS experience,

            -- Package
            u.package_id,
            package.package_id AS package_reference_id,
            package_type.name AS package_type_name

        FROM public.professionals p

        JOIN public.user_accounts u
            ON u.id = p.useraccount_ptr_id

        -- Professional relationships
        LEFT JOIN public.departments d
            ON d.id = p.department_id

        LEFT JOIN public.job_levels jl
            ON jl.id = p.job_level_id

        LEFT JOIN public.job_role jr
            ON jr.id = p.job_role_id

        LEFT JOIN public.education_level el
            ON el.id = p.education_level_id

        LEFT JOIN public.companies c
            ON c.useraccount_ptr_id = p.current_company_id

        -- Country
        LEFT JOIN public.countries current_country
            ON current_country.id = u.current_country_id

        -- Nationality
        LEFT JOIN public.countries nationality
            ON nationality.id = u.nationality_id

        -- Package
        LEFT JOIN public.base_package package
            ON package.id = u.package_id

        LEFT JOIN public.base_packagetype package_type
            ON package_type.id = package.package_type_id

        -- Skills
        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT s.name,
                    ', '
                    ORDER BY s.name
                ) AS skills
            FROM public.professionals_skills ps
            JOIN public.skills s
                ON s.id = ps.skills_id
            WHERE ps.professional_id = p.useraccount_ptr_id
        ) sk
            ON TRUE

        -- Languages
        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT l.name,
                    ', '
                    ORDER BY l.name
                ) AS languages
            FROM public.professionals_language_know pl
            JOIN public.languages l
                ON l.id = pl.language_id
            WHERE pl.professional_id = p.useraccount_ptr_id
        ) la
            ON TRUE

        -- ALL INDUSTRIES
        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT i.name,
                    ', '
                    ORDER BY i.name
                ) AS industries
            FROM public.user_accounts_industry ui
            JOIN public.industries i
                ON i.id = ui.industry_id
            WHERE ui.useraccount_id = p.useraccount_ptr_id
        ) ind
            ON TRUE

        -- SUPPLIER INDUSTRIES
        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT i.name,
                    ', '
                    ORDER BY i.name
                ) AS industries
            FROM public.user_accounts_industry ui
            JOIN public.industries i
                ON i.id = ui.industry_id
            WHERE ui.useraccount_id = p.useraccount_ptr_id
              AND LOWER(COALESCE(i.context, ''))
                    = 'supplier'
        ) supplier_ind
            ON TRUE

        -- SUPPLIER CATEGORIES
        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT sc.name,
                    ', '
                    ORDER BY sc.name
                ) AS supplier_categories
            FROM public.user_accounts_supplier_category usc
            JOIN public.supplier_category sc
                ON sc.id = usc.suppliercategory_id
            WHERE usc.useraccount_id = p.useraccount_ptr_id
        ) supplier_cat
            ON TRUE

        -- Experience
        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT concat_ws(
                        ' — ',
                        e.job_designation,
                        e.company_name,
                        e.location,
                        d2.name,
                        jr2.name,
                        i2.name
                    ),
                    ' | '
                    ORDER BY concat_ws(
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

            LEFT JOIN public.departments d2
                ON d2.id = e.department_id

            LEFT JOIN public.job_role jr2
                ON jr2.id = e.job_role_id

            LEFT JOIN public.industries i2
                ON i2.id = e.industry_id

            WHERE e.user_id = p.useraccount_ptr_id
        ) exp
            ON TRUE

        WHERE p.useraccount_ptr_id = ANY(%s)
        """,
        (ids,),
    )

def is_supplier_account(source: dict[str, Any]) -> bool:
    return bool(
        clean(source.get("supplier_industries"))
        or clean(source.get("supplier_categories"))
    )


def company_rows(cur, ids):
    """
    Fetch Company + UserAccount information + relationships.

    A Company can also be a Supplier.

    Supplier detection:
        UserAccount.industry -> Industry.context = 'supplier'

    Supplier categories:
        UserAccount.supplier_category -> SupplierCategory
    """

    return db_rows(
        cur,
        """
        SELECT
            c.useraccount_ptr_id AS object_id,

            -- Company
            c.name AS company_name,
            c.created_by,
            c.current_designation,
            c.website_link,

            -- User Account
            concat_ws(
                ' ',
                NULLIF(u.first_name, ''),
                NULLIF(u.last_name, '')
            ) AS full_name,

            u.username,
            u.user_type,
            u.company_name AS account_company_name,
            u.about_us,
            u.tagline,
            u.city_town,
            u.address,
            u.postal_code,
            u.slug AS user_slug,

            u.current_country_id,
            current_country.name AS current_country_name,
            current_country.country_code AS current_country_code,

            u.nationality_id,
            nationality.name AS nationality_name,

            u.verified,
            u.is_pro,
            u.is_working,
            u.is_featured,

            -- General + Supplier Industries
            COALESCE(
                ind.industries,
                ''
            ) AS industries,

            COALESCE(
                supplier_ind.industries,
                ''
            ) AS supplier_industries,

            -- Supplier Categories
            COALESCE(
                supplier_cat.supplier_categories,
                ''
            ) AS supplier_categories,

            -- Package
            u.package_id,
            package.package_id AS package_reference_id,
            package_type.name AS package_type_name

        FROM public.companies c

        JOIN public.user_accounts u
            ON u.id = c.useraccount_ptr_id

        -- Current Country
        LEFT JOIN public.countries current_country
            ON current_country.id = u.current_country_id

        -- Nationality
        LEFT JOIN public.countries nationality
            ON nationality.id = u.nationality_id

        -- Package
        LEFT JOIN public.base_package package
            ON package.id = u.package_id

        LEFT JOIN public.base_packagetype package_type
            ON package_type.id = package.package_type_id

        -- ALL INDUSTRIES
        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT i.name,
                    ', '
                    ORDER BY i.name
                ) AS industries
            FROM public.user_accounts_industry ui
            JOIN public.industries i
                ON i.id = ui.industry_id
            WHERE ui.useraccount_id = c.useraccount_ptr_id
        ) ind
            ON TRUE

        -- ONLY SUPPLIER INDUSTRIES
        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT i.name,
                    ', '
                    ORDER BY i.name
                ) AS industries
            FROM public.user_accounts_industry ui
            JOIN public.industries i
                ON i.id = ui.industry_id
            WHERE ui.useraccount_id = c.useraccount_ptr_id
              AND LOWER(COALESCE(i.context, ''))
                    = 'supplier'
        ) supplier_ind
            ON TRUE

        -- SUPPLIER CATEGORIES
        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT sc.name,
                    ', '
                    ORDER BY sc.name
                ) AS supplier_categories
            FROM public.user_accounts_supplier_category usc
            JOIN public.supplier_category sc
                ON sc.id = usc.suppliercategory_id
            WHERE usc.useraccount_id = c.useraccount_ptr_id
        ) supplier_cat
            ON TRUE

        WHERE c.useraccount_ptr_id = ANY(%s)
        """,
        (ids,),
    )


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
    """
    Fetch Job + all important searchable relationships.

    Includes:
      - Company
      - Company UserAccount information
      - Job Industries
      - Job Roles
      - Job Departments
      - Job Levels
      - Country
      - Job Type
      - Employment Type
      - Salary Range
      - Currency
      - Package Type
      - Posted By User
      - Received Application Countries
      - Job flags/status
      - Walk-in information
      - Job dates
      - Job URL / slug
    """

    return db_rows(
        cur,
        """
        SELECT
            j.id AS object_id,

            /* ============================================================
               JOB CORE
               ============================================================ */

            j.job_title,
            j.job_desc,
            j.job_city,
            j.job_address,
            j.job_status,

            j.job_start_date,
            j.job_end_date,

            j.salary_description,
            j.reference,

            j.posted_by,
            j.posted_by_user_id,

            j.job_link,
            j.slug,

            j.is_live,
            j.is_deleted,
            j.is_featured,
            j.is_premium,

            j.is_spider,
            j.is_spider_job,
            j.is_confidential,
            j.hide_company_details,

            j.is_auto_renew_enabled,
            j.is_credit_used,
            j.is_migrated,

            j.receive_response,

            /* Walk-in */
            j."walkInFromDate" AS walkin_from_date,
            j."walkInToDate" AS walkin_to_date,
            j."walkInTime" AS walkin_time,
            j."walkInVenue" AS walkin_venue,

            /* ============================================================
               COMPANY
               ============================================================ */

            j.company_id,

            c.name AS company_name,

            cu.username AS company_username,

            concat_ws(
                ' ',
                NULLIF(cu.first_name, ''),
                NULLIF(cu.last_name, '')
            ) AS company_account_name,

            cu.company_name AS company_account_company_name,

            cu.city_town AS company_city,

            cu.address AS company_address,

            company_country.name AS company_country_name,

            company_country.country_code AS company_country_code,

            cu.user_type AS company_user_type,

            cu.verified AS company_verified,

            cu.is_pro AS company_is_pro,

            cu.is_featured AS company_is_featured,

            /* ============================================================
               JOB COUNTRY
               ============================================================ */

            j.job_country_id,

            co.name AS country_name,

            co.country_code AS country_code,

            co.code AS country_code_short,

            /* ============================================================
               JOB TYPE
               ============================================================ */

            j.jobtype_id,

            jt.name AS job_type,

            /* ============================================================
               EMPLOYMENT TYPE
               ============================================================ */

            j.employementtype_id,

            et.name AS employment_type,

            /* ============================================================
               SALARY RANGE
               ============================================================ */

            j."salaryRange_id",

            sr.name AS salary_range,

            /* ============================================================
               CURRENCY
               ============================================================ */

            j.currency_id,

            cur.code AS currency_code,

            cur.name AS currency_name,

            cur.symbol AS currency_symbol,

            /* ============================================================
               PACKAGE TYPE
               ============================================================ */

            j.package_type_id,

            package_type.name AS package_type_name,

            package_type.description AS package_type_description,

            package_type.is_PAYG AS package_is_payg,
            package_type.is_POP AS package_is_pop,
            package_type.is_CC AS package_is_cc,
            package_type.is_PREMIUM AS package_is_premium,
            package_type.is_SP AS package_is_sp,
            package_type.is_GP AS package_is_gp,
            package_type.is_EP AS package_is_ep,

            /* ============================================================
               POSTED BY USER
               ============================================================ */

            posted_user.username AS posted_by_username,

            concat_ws(
                ' ',
                NULLIF(posted_user.first_name, ''),
                NULLIF(posted_user.last_name, '')
            ) AS posted_by_name,

            posted_user.user_type AS posted_by_user_type,

            posted_country.name AS posted_by_country_name,

            posted_country.country_code AS posted_by_country_code,

            /* ============================================================
               JOB INDUSTRIES
               ============================================================ */

            COALESCE(
                ind.industries,
                ''
            ) AS industries,

            /* ============================================================
               JOB ROLES
               ============================================================ */

            COALESCE(
                role.roles,
                ''
            ) AS roles,

            /* ============================================================
               JOB DEPARTMENTS
               ============================================================ */

            COALESCE(
                dep.departments,
                ''
            ) AS departments,

            /* ============================================================
               JOB LEVELS
               ============================================================ */

            COALESCE(
                lvl.levels,
                ''
            ) AS levels,

            /* ============================================================
               RECEIVED APPLICATION COUNTRIES
               ============================================================ */

            COALESCE(
                received_countries.countries,
                ''
            ) AS received_application_countries,

            /* ============================================================
               COMPANY INDUSTRIES
               ============================================================ */

            COALESCE(
                company_ind.industries,
                ''
            ) AS company_industries,

            /* ============================================================
               COMPANY SUPPLIER INDUSTRIES
               ============================================================ */

            COALESCE(
                company_supplier_ind.industries,
                ''
            ) AS company_supplier_industries,

            /* ============================================================
               COMPANY SUPPLIER CATEGORIES
               ============================================================ */

            COALESCE(
                company_supplier_cat.categories,
                ''
            ) AS company_supplier_categories

        FROM public.base_job j

        /* ================================================================
           COMPANY
           ================================================================= */

        LEFT JOIN public.companies c
            ON c.useraccount_ptr_id = j.company_id

        LEFT JOIN public.user_accounts cu
            ON cu.id = c.useraccount_ptr_id

        LEFT JOIN public.countries company_country
            ON company_country.id = cu.current_country_id

        /* ================================================================
           JOB COUNTRY
           ================================================================= */

        LEFT JOIN public.countries co
            ON co.id = j.job_country_id

        /* ================================================================
           JOB TYPE
           ================================================================= */

        LEFT JOIN public.job_type jt
            ON jt.id = j.jobtype_id

        /* ================================================================
           EMPLOYMENT TYPE
           ================================================================= */

        LEFT JOIN public.employment_type et
            ON et.id = j.employementtype_id

        /* ================================================================
           SALARY RANGE
           ================================================================= */

        LEFT JOIN public.base_salaryrange sr
            ON sr.id = j."salaryRange_id"

        /* ================================================================
           CURRENCY
           ================================================================= */

        LEFT JOIN public.base_currency cur
            ON cur.id = j.currency_id

        /* ================================================================
           PACKAGE TYPE
           ================================================================= */

        LEFT JOIN public.base_packagetype package_type
            ON package_type.id = j.package_type_id

        /* ================================================================
           POSTED BY USER
           ================================================================= */

        LEFT JOIN public.user_accounts posted_user
            ON posted_user.id = j.posted_by_user_id

        LEFT JOIN public.countries posted_country
            ON posted_country.id = posted_user.current_country_id

        /* ================================================================
           JOB ROLES
           ================================================================= */

        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT r.name,
                    ', '
                    ORDER BY r.name
                ) AS roles
            FROM public.base_job_job_role x
            JOIN public.job_role r
                ON r.id = x.jobrole_id
            WHERE x.job_id = j.id
        ) role
            ON TRUE

        /* ================================================================
           JOB DEPARTMENTS
           ================================================================= */

        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT d.name,
                    ', '
                    ORDER BY d.name
                ) AS departments
            FROM public.base_job_job_department x
            JOIN public.departments d
                ON d.id = x.department_id
            WHERE x.job_id = j.id
        ) dep
            ON TRUE

        /* ================================================================
           JOB LEVELS
           ================================================================= */

        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT l.name,
                    ', '
                    ORDER BY l.name
                ) AS levels
            FROM public.base_job_job_level x
            JOIN public.job_levels l
                ON l.id = x.joblevel_id
            WHERE x.job_id = j.id
        ) lvl
            ON TRUE

        /* ================================================================
           JOB INDUSTRIES
           ================================================================= */

        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT i.name,
                    ', '
                    ORDER BY i.name
                ) AS industries
            FROM public.base_job_job_industry x
            JOIN public.industries i
                ON i.id = x.industry_id
            WHERE x.job_id = j.id
        ) ind
            ON TRUE

        /* ================================================================
           RECEIVED APPLICATION COUNTRIES
           ================================================================= */

        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT rc.name,
                    ', '
                    ORDER BY rc.name
                ) AS countries
            FROM public.base_job_received_applications x
            JOIN public.countries rc
                ON rc.id = x.country_id
            WHERE x.job_id = j.id
        ) received_countries
            ON TRUE

        /* ================================================================
           COMPANY INDUSTRIES
           ================================================================= */

        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT i.name,
                    ', '
                    ORDER BY i.name
                ) AS industries
            FROM public.user_accounts_industry ui
            JOIN public.industries i
                ON i.id = ui.industry_id
            WHERE ui.useraccount_id = j.company_id
        ) company_ind
            ON TRUE

        /* ================================================================
           COMPANY SUPPLIER INDUSTRIES

           A company is considered a supplier when it has an Industry
           whose context is "supplier".
           ================================================================= */

        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT i.name,
                    ', '
                    ORDER BY i.name
                ) AS industries
            FROM public.user_accounts_industry ui
            JOIN public.industries i
                ON i.id = ui.industry_id
            WHERE ui.useraccount_id = j.company_id
              AND LOWER(COALESCE(i.context, '')) = 'supplier'
        ) company_supplier_ind
            ON TRUE

        /* ================================================================
           COMPANY SUPPLIER CATEGORIES
           ================================================================= */

        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT sc.name,
                    ', '
                    ORDER BY sc.name
                ) AS categories
            FROM public.user_accounts_supplier_category usc
            JOIN public.supplier_category sc
                ON sc.id = usc.suppliercategory_id
            WHERE usc.useraccount_id = j.company_id
        ) company_supplier_cat
            ON TRUE

        WHERE j.id = ANY(%s)
    """,
        (ids,),
    )


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
    """
    Fetch Product + all important searchable relationships.

    Relationships included:
      ProductCategory
      Currency
      Country
      Available Countries
      Posted By User
      Package
      Package Type
    """

    return db_rows(
        cur,
        """
        SELECT
            p.id AS object_id,

            -- Product
            p.title,
            p.p_type,
            p.p_condition,
            p.price,
            p.price_setting,
            p.current_location,
            p.prime_city,
            p.other_location,
            p.keywords,
            p.description,
            p.status,
            p.start_date,
            p.expiry_date,
            p.discount_percentage,
            p.discounted_price,
            p.is_featured,
            p.is_auto_renew_enabled,
            p.slug,

            -- Posted By
            p.posted_by_id,
            u.username AS posted_by_username,
            concat_ws(
                ' ',
                NULLIF(u.first_name, ''),
                NULLIF(u.last_name, '')
            ) AS posted_by_name,
            u.user_type AS posted_by_user_type,

            -- Product Country
            p.country_id,
            country.name AS country_name,
            country.country_code AS country_code,

            -- Currency
            p.currency_id,
            currency.code AS currency_code,
            currency.name AS currency_name,
            currency.symbol AS currency_symbol,

            -- Package
            p.package_id,
            package.package_id AS package_reference_id,
            package_type.name AS package_type_name,

            -- Product Categories
            COALESCE(
                categories.categories,
                ''
            ) AS categories,

            -- Available Countries
            COALESCE(
                available_countries.available_countries,
                ''
            ) AS available_countries

        FROM public.marketplace_product p

        LEFT JOIN public.user_accounts u
            ON u.id = p.posted_by_id

        LEFT JOIN public.countries country
            ON country.id = p.country_id

        LEFT JOIN public.base_currency currency
            ON currency.id = p.currency_id

        LEFT JOIN public.base_package package
            ON package.id = p.package_id

        LEFT JOIN public.base_packagetype package_type
            ON package_type.id = package.package_type_id

        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT pc.name,
                    ', '
                    ORDER BY pc.name
                ) AS categories
            FROM public.marketplace_product_category x
            JOIN public.marketplace_productcategory pc
                ON pc.id = x.productcategory_id
            WHERE x.product_id = p.id
        ) categories
            ON TRUE

        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT c.name,
                    ', '
                    ORDER BY c.name
                ) AS available_countries
            FROM public.marketplace_product_available_in_countries x
            JOIN public.countries c
                ON c.id = x.country_id
            WHERE x.product_id = p.id
        ) available_countries
            ON TRUE

        WHERE p.id = ANY(%s)
        """,
        (ids,),
    )


def faq_rows(cur, ids):
    return db_rows(cur, """
        SELECT id AS object_id, question, answer, is_popular, show_on_landing
        FROM public.base_faq
        WHERE id = ANY(%s)
    """, (ids,))


def award_rows(cur, ids):
    """
    Fetch Award + all important searchable relationships.

    Includes:
      - Award title / subtitle / description
      - Award short title
      - Award categories
      - Category countries
      - Award country
      - Award location
      - Award year
      - Voting dates / status
      - Award active status
      - Nomination / voting / category / winner links
      - Award code / slug
      - YouTube link
      - Award date
      - Coordinates
      - Display/action flags
    """

    return db_rows(
        cur,
        """
        SELECT
            a.id AS object_id,

            /* ============================================================
               AWARD CORE
               ============================================================ */

            a.sequence_number,

            a.award_short_title,
            a.award_title,
            a.award_subtitle,
            a.award_description,

            a.award_detail_url,

            a.award_is_active,
            a.award_created_at,

            a.voting_start_date,
            a.voting_end_date,

            a.award_on,

            a.show_nomination_button,
            a.nomination_link,

            a.show_vote_now_button,
            a.vote_now_link,

            a.show_personal_categories_button,
            a.personal_categories_link,

            a.show_corporate_categories_button,
            a.corporate_categories_link,

            a.show_award_winners_button,
            a.award_winners_link,

            a.is_voting_active,

            a.youtube_link,

            a.location,

            a.country_id,
            co.name AS country_name,
            co.country_code AS country_code,
            co.code AS country_code_short,

            a.hide_country_from_url,

            a.code,
            a.slug,

            a.latitude,
            a.longitude,

            a.award_year,

            /* ============================================================
               AWARD CATEGORIES
               ============================================================ */

            COALESCE(
                cat.categories,
                ''
            ) AS categories,

            /* Category countries are useful for queries such as:
               "awards for UAE hospitality categories"
            */

            COALESCE(
                cat.category_countries,
                ''
            ) AS category_countries,

            /* Structured category information for metadata */

            COALESCE(
                cat.category_details::text,
                '[]'
            ) AS category_details

        FROM public.base_awards a

        /* ================================================================
           AWARD COUNTRY
           ================================================================= */

        LEFT JOIN public.countries co
            ON co.id = a.country_id

        /* ================================================================
           AWARD CATEGORIES
           ================================================================= */

        LEFT JOIN LATERAL (
            SELECT
                string_agg(
                    DISTINCT ac.category_name,
                    ', '
                    ORDER BY ac.category_name
                ) AS categories,

                string_agg(
                    DISTINCT category_country.name,
                    ', '
                    ORDER BY category_country.name
                ) AS category_countries,

                jsonb_agg(
                    DISTINCT jsonb_build_object(
                        'id', ac.id,
                        'name', ac.category_name,
                        'is_active', ac.category_is_active,
                        'country_id', ac.country_id,
                        'country', category_country.name,
                        'country_code', category_country.country_code
                    )
                ) AS category_details

            FROM public.base_awards_award_category x

            JOIN public.base_awardcategory ac
                ON ac.id = x.awardcategory_id

            LEFT JOIN public.countries category_country
                ON category_country.id = ac.country_id

            WHERE x.awards_id = a.id
        ) cat
            ON TRUE

        WHERE a.id = ANY(%s)
        """,
        (ids,),
    )


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
    model = clean(
        master.get("entity_type")
        or ""
    ).lower() or "record"

    fields = [
        ("Entity Type", model),
        ("Title", master.get("title")),
        ("Name", master.get("user_name")),
        ("Category", master.get("category_text")),
        ("Location", master.get("location_text")),
        ("Keywords", master.get("ai_keywords")),
        ("Content", master.get("content")),
    ]

    relationship_meta: dict[str, Any] = {}

    if model == "professional":

        supplier_account = is_supplier_account(source)

        relationship_meta = {
            "profile_type": "professional",

            "department": source.get("department_name"),
            "job_role": source.get("job_role_name"),
            "job_level": source.get("job_level_name"),
            "education_level": source.get(
                "education_level_name"
            ),

            "current_company": source.get(
                "current_company_name"
            ),

            "current_company_text": source.get(
                "current_company_text"
            ),

            "currently_working": source.get(
                "currently_working"
            ),

            "skills": source.get("skills"),
            "languages": source.get("languages"),

            "industries": source.get("industries"),

            "supplier": {
                "is_supplier": supplier_account,
                "industries": source.get(
                    "supplier_industries"
                ),
                "categories": source.get(
                    "supplier_categories"
                ),
            },

            "experience": source.get("experience"),

            "country": {
                "id": source.get(
                    "current_country_id"
                ),
                "name": source.get(
                    "current_country_name"
                ),
                "code": source.get(
                    "current_country_code"
                ),
            },

            "nationality": source.get(
                "nationality_name"
            ),

            "profile": source.get("about_us"),
            "tagline": source.get("tagline"),
            "city": source.get("city_town"),

            "verified": source.get("verified"),
            "pro": source.get("is_pro"),
            "featured": source.get("is_featured"),

            "package": {
                "id": source.get("package_id"),
                "reference_id": source.get(
                    "package_reference_id"
                ),
                "type": source.get(
                    "package_type_name"
                ),
            },
        }

        fields += [
            ("Professional Name", source.get("full_name")),
            ("Username", source.get("username")),

            ("Resume Title", source.get("resume_title")),

            ("Job Role", source.get("job_role_name")),
            ("Department", source.get("department_name")),
            ("Job Level", source.get("job_level_name")),
            ("Education", source.get(
                "education_level_name"
            )),

            (
                "Current Company",
                source.get("current_company_name"),
            ),

            (
                "Current Company",
                source.get("current_company_text"),
            ),

            (
                "Currently Working",
                source.get("currently_working"),
            ),

            ("Skills", source.get("skills")),
            ("Languages", source.get("languages")),

            ("Industries", source.get("industries")),

            (
                "Supplier Industries",
                source.get("supplier_industries"),
            ),

            (
                "Supplier Categories",
                source.get("supplier_categories"),
            ),

            ("Experience", source.get("experience")),

            ("Profile", source.get("about_us")),
            ("Tagline", source.get("tagline")),

            ("City", source.get("city_town")),
            (
                "Current Country",
                source.get("current_country_name"),
            ),
            (
                "Country Code",
                source.get("current_country_code"),
            ),

            (
                "Nationality",
                source.get("nationality_name"),
            ),

            (
                "Profile Type",
                "Supplier Professional"
                if supplier_account
                else "Professional",
            ),

            (
                "Verified",
                "Yes"
                if source.get("verified")
                else "No",
            ),

            (
                "Pro Member",
                "Yes"
                if source.get("is_pro")
                else "No",
            ),

            (
                "Featured",
                "Yes"
                if source.get("is_featured")
                else "No",
            ),

            (
                "Package Type",
                source.get("package_type_name"),
            ),
        ]

    elif model == "company":
        supplier_account = is_supplier_account(source)

        relationship_meta = {
            "profile_type": (
                "supplier"
                if supplier_account
                else "company"
            ),

            "company_name": source.get(
                "company_name"
            ),

            "about": source.get("about_us"),

            "tagline": source.get("tagline"),

            "designation": source.get(
                "current_designation"
            ),

            "industries": source.get(
                "industries"
            ),

            "supplier": {
                "is_supplier": supplier_account,
                "industries": source.get(
                    "supplier_industries"
                ),
                "categories": source.get(
                    "supplier_categories"
                ),
            },

            "city": source.get("city_town"),

            "country": {
                "id": source.get(
                    "current_country_id"
                ),
                "name": source.get(
                    "current_country_name"
                ),
                "code": source.get(
                    "current_country_code"
                ),
            },

            "website": source.get(
                "website_link"
            ),

            "verified": source.get("verified"),
            "pro": source.get("is_pro"),
            "featured": source.get("is_featured"),

            "package": {
                "id": source.get("package_id"),
                "reference_id": source.get(
                    "package_reference_id"
                ),
                "type": source.get(
                    "package_type_name"
                ),
            },
        }

        fields += [
            (
                "Company Name",
                source.get("company_name"),
            ),

            (
                "Account Name",
                source.get("full_name"),
            ),

            (
                "Username",
                source.get("username"),
            ),

            (
                "About",
                source.get("about_us"),
            ),

            (
                "Tagline",
                source.get("tagline"),
            ),

            (
                "Designation",
                source.get("current_designation"),
            ),

            (
                "Created By",
                source.get("created_by"),
            ),

            (
                "Industries",
                source.get("industries"),
            ),

            (
                "Supplier Industries",
                source.get("supplier_industries"),
            ),

            (
                "Supplier Categories",
                source.get("supplier_categories"),
            ),

            (
                "City",
                source.get("city_town"),
            ),

            (
                "Country",
                source.get("current_country_name"),
            ),

            (
                "Country Code",
                source.get("current_country_code"),
            ),

            (
                "Website",
                source.get("website_link"),
            ),

            (
                "Profile Type",
                "Supplier"
                if supplier_account
                else "Company",
            ),

            (
                "Verified",
                "Yes"
                if source.get("verified")
                else "No",
            ),

            (
                "Pro Member",
                "Yes"
                if source.get("is_pro")
                else "No",
            ),

            (
                "Featured",
                "Yes"
                if source.get("is_featured")
                else "No",
            ),

            (
                "Package Type",
                source.get("package_type_name"),
            ),
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
            "company": {
                "id": source.get("company_id"),
                "name": source.get("company_name"),
                "username": source.get("company_username"),
                "city": source.get("company_city"),
                "country": source.get("company_country_name"),
                "country_code": source.get("company_country_code"),
                "industries": source.get("company_industries"),
                "supplier_industries": source.get(
                    "company_supplier_industries"
                ),
                "supplier_categories": source.get(
                    "company_supplier_categories"
                ),
                "user_type": source.get("company_user_type"),
                "verified": source.get("company_verified"),
                "pro": source.get("company_is_pro"),
                "featured": source.get("company_is_featured"),
            },

            "country": {
                "id": source.get("job_country_id"),
                "name": source.get("country_name"),
                "code": source.get("country_code"),
                "short_code": source.get("country_code_short"),
            },

            "roles": source.get("roles"),

            "departments": source.get("departments"),

            "levels": source.get("levels"),

            "industries": source.get("industries"),

            "employment_type": source.get(
                "employment_type"
            ),

            "job_type": source.get(
                "job_type"
            ),

            "salary": {
                "range": source.get("salary_range"),
                "description": source.get("salary_description"),
                "currency": source.get("currency_code"),
                "currency_name": source.get("currency_name"),
                "currency_symbol": source.get("currency_symbol"),
            },

            "package_type": {
                "id": source.get("package_type_id"),
                "name": source.get("package_type_name"),
                "description": source.get("package_type_description"),
                "payg": source.get("package_is_payg"),
                "pop": source.get("package_is_pop"),
                "cc": source.get("package_is_cc"),
                "premium": source.get("package_is_premium"),
                "sp": source.get("package_is_sp"),
                "gp": source.get("package_is_gp"),
                "ep": source.get("package_is_ep"),
            },

            "posted_by": {
                "id": source.get("posted_by_user_id"),
                "name": source.get("posted_by_name"),
                "username": source.get("posted_by_username"),
                "user_type": source.get("posted_by_user_type"),
                "country": source.get("posted_by_country_name"),
                "country_code": source.get("posted_by_country_code"),
            },

            "received_application_countries": source.get(
                "received_application_countries"
            ),

            "status": source.get("job_status"),

            "is_live": source.get("is_live"),

            "is_deleted": source.get("is_deleted"),

            "featured": source.get("is_featured"),

            "premium": source.get("is_premium"),

            "confidential": source.get("is_confidential"),

            "walkin": {
                "from": source.get("walkin_from_date"),
                "to": source.get("walkin_to_date"),
                "time": source.get("walkin_time"),
                "venue": source.get("walkin_venue"),
            },
        }

        fields += [

            # ------------------------------------------------------------
            # Core Job
            # ------------------------------------------------------------

            ("Job Title", source.get("job_title")),

            ("Job Description", source.get("job_desc")),

            ("Job Status", source.get("job_status")),

            # ------------------------------------------------------------
            # Company
            # ------------------------------------------------------------

            ("Company", source.get("company_name")),

            ("Company Username", source.get("company_username")),

            ("Company City", source.get("company_city")),

            ("Company Country", source.get("company_country_name")),

            ("Company Country Code", source.get(
                "company_country_code"
            )),

            ("Company Industries", source.get(
                "company_industries"
            )),

            (
                "Company Supplier Industries",
                source.get("company_supplier_industries"),
            ),

            (
                "Company Supplier Categories",
                source.get("company_supplier_categories"),
            ),

            # ------------------------------------------------------------
            # Job Classification
            # ------------------------------------------------------------

            ("Industry", source.get("industries")),

            ("Job Role", source.get("roles")),

            ("Department", source.get("departments")),

            ("Job Level", source.get("levels")),

            ("Job Type", source.get("job_type")),

            ("Employment Type", source.get(
                "employment_type"
            )),

            # ------------------------------------------------------------
            # Location
            # ------------------------------------------------------------

            ("Country", source.get("country_name")),

            ("Country Code", source.get("country_code")),

            ("Job City", source.get("job_city")),

            ("Job Address", source.get("job_address")),

            # ------------------------------------------------------------
            # Salary
            # ------------------------------------------------------------

            ("Salary Range", source.get("salary_range")),

            ("Salary Description", source.get(
                "salary_description"
            )),

            ("Currency", source.get("currency_name")),

            ("Currency Code", source.get(
                "currency_code"
            )),

            ("Currency Symbol", source.get(
                "currency_symbol"
            )),

            # ------------------------------------------------------------
            # Package
            # ------------------------------------------------------------

            ("Package Type", source.get(
                "package_type_name"
            )),

            ("Package Description", source.get(
                "package_type_description"
            )),

            # ------------------------------------------------------------
            # Posting
            # ------------------------------------------------------------

            ("Posted By", source.get(
                "posted_by_name"
            )),

            ("Posted By Username", source.get(
                "posted_by_username"
            )),

            ("Posted By User Type", source.get(
                "posted_by_user_type"
            )),

            # ------------------------------------------------------------
            # Dates
            # ------------------------------------------------------------

            ("Job Start Date", source.get(
                "job_start_date"
            )),

            ("Job End Date", source.get(
                "job_end_date"
            )),

            # ------------------------------------------------------------
            # Walk-in
            # ------------------------------------------------------------

            ("Walk-in From", source.get(
                "walkin_from_date"
            )),

            ("Walk-in To", source.get(
                "walkin_to_date"
            )),

            ("Walk-in Time", source.get(
                "walkin_time"
            )),

            ("Walk-in Venue", source.get(
                "walkin_venue"
            )),

            # ------------------------------------------------------------
            # Application Countries
            # ------------------------------------------------------------

            (
                "Application Countries",
                source.get("received_application_countries"),
            ),

            # ------------------------------------------------------------
            # Job Link / Reference
            # ------------------------------------------------------------

            ("Job Reference", source.get(
                "reference"
            )),

            ("Job Slug", source.get(
                "slug"
            )),

            # ------------------------------------------------------------
            # Flags
            # ------------------------------------------------------------

            (
                "Featured",
                "Yes"
                if source.get("is_featured")
                else "No",
            ),

            (
                "Premium",
                "Yes"
                if source.get("is_premium")
                else "No",
            ),

            (
                "Live",
                "Yes"
                if source.get("is_live")
                else "No",
            ),

            (
                "Confidential",
                "Yes"
                if source.get("is_confidential")
                else "No",
            ),

            (
                "Spider Job",
                "Yes"
                if source.get("is_spider_job")
                else "No",
            ),

            (
                "Auto Renew",
                "Yes"
                if source.get("is_auto_renew_enabled")
                else "No",
            ),
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
            "product_type": source.get("p_type"),
            "condition": source.get("p_condition"),

            "currency": {
                "id": source.get("currency_id"),
                "code": source.get("currency_code"),
                "name": source.get("currency_name"),
                "symbol": source.get("currency_symbol"),
            },

            "country": {
                "id": source.get("country_id"),
                "name": source.get("country_name"),
                "code": source.get("country_code"),
            },

            "available_in_countries": source.get(
                "available_countries"
            ),

            "posted_by": {
                "id": source.get("posted_by_id"),
                "username": source.get("posted_by_username"),
                "name": source.get("posted_by_name"),
                "user_type": source.get("posted_by_user_type"),
            },

            "package": {
                "id": source.get("package_id"),
                "reference_id": source.get("package_reference_id"),
                "type": source.get("package_type_name"),
            },

            "location": {
                "current_location": source.get("current_location"),
                "prime_city": source.get("prime_city"),
                "other_location": source.get("other_location"),
            },

            "pricing": {
                "price": source.get("price"),
                "price_setting": source.get("price_setting"),
                "discount_percentage": source.get(
                    "discount_percentage"
                ),
                "discounted_price": source.get(
                    "discounted_price"
                ),
                "currency_code": source.get("currency_code"),
                "currency_name": source.get("currency_name"),
                "currency_symbol": source.get("currency_symbol"),
            },

            "availability": {
                "start_date": source.get("start_date"),
                "expiry_date": source.get("expiry_date"),
            },

            "featured": source.get("is_featured"),
            "auto_renew": source.get("is_auto_renew_enabled"),
            "status": source.get("status"),
        }

        fields += [
            ("Product Type", source.get("p_type")),
            ("Product Condition", source.get("p_condition")),

            ("Product Categories", source.get("categories")),

            ("Price", source.get("price")),
            ("Price Setting", source.get("price_setting")),

            ("Currency Code", source.get("currency_code")),
            ("Currency Name", source.get("currency_name")),
            ("Currency Symbol", source.get("currency_symbol")),

            ("Discount Percentage", source.get(
                "discount_percentage"
            )),
            ("Discounted Price", source.get(
                "discounted_price"
            )),

            ("Current Location", source.get(
                "current_location"
            )),
            ("Prime City", source.get(
                "prime_city"
            )),
            ("Other Locations", source.get(
                "other_location"
            )),

            ("Country", source.get("country_name")),
            ("Country Code", source.get("country_code")),

            (
                "Available In Countries",
                source.get("available_countries"),
            ),

            ("Keywords", source.get("keywords")),
            ("Description", source.get("description")),

            ("Posted By", source.get("posted_by_name")),
            ("Posted By Username", source.get(
                "posted_by_username"
            )),
            ("Posted By User Type", source.get(
                "posted_by_user_type"
            )),

            ("Package Type", source.get(
                "package_type_name"
            )),

            ("Status", source.get("status")),

            ("Available From", source.get(
                "start_date"
            )),
            ("Available Until", source.get(
                "expiry_date"
            )),

            (
                "Featured",
                "Yes" if source.get("is_featured") else "No",
            ),

            (
                "Auto Renew",
                "Yes"
                if source.get("is_auto_renew_enabled")
                else "No",
            ),
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
            "award": {
                "id": master.get("object_id"),
                "title": source.get("award_title"),
                "short_title": source.get("award_short_title"),
                "subtitle": source.get("award_subtitle"),
                "code": source.get("code"),
                "slug": source.get("slug"),
                "sequence_number": source.get("sequence_number"),
            },

            "categories": {
                "names": source.get("categories"),
                "countries": source.get("category_countries"),
                "details": source.get("category_details"),
            },

            "country": {
                "id": source.get("country_id"),
                "name": source.get("country_name"),
                "code": source.get("country_code"),
                "short_code": source.get("country_code_short"),
            },

            "location": {
                "location": source.get("location"),
                "latitude": source.get("latitude"),
                "longitude": source.get("longitude"),
            },

            "year": source.get("award_year"),

            "dates": {
                "award_on": source.get("award_on"),
                "voting_start": source.get("voting_start_date"),
                "voting_end": source.get("voting_end_date"),
                "created_at": source.get("award_created_at"),
            },

            "status": {
                "is_active": source.get("award_is_active"),
                "is_voting_active": source.get("is_voting_active"),
                "hide_country_from_url": source.get(
                    "hide_country_from_url"
                ),
            },

            "nomination": {
                "enabled": source.get("show_nomination_button"),
                "link": source.get("nomination_link"),
            },

            "voting": {
                "enabled": source.get("show_vote_now_button"),
                "link": source.get("vote_now_link"),
            },

            "personal_categories": {
                "enabled": source.get(
                    "show_personal_categories_button"
                ),
                "link": source.get(
                    "personal_categories_link"
                ),
            },

            "corporate_categories": {
                "enabled": source.get(
                    "show_corporate_categories_button"
                ),
                "link": source.get(
                    "corporate_categories_link"
                ),
            },

            "winners": {
                "enabled": source.get(
                    "show_award_winners_button"
                ),
                "link": source.get(
                    "award_winners_link"
                ),
            },

            "media": {
                "youtube": source.get("youtube_link"),
                "detail_url": source.get("award_detail_url"),
            },
        }

        fields += [

            ("Award Title", source.get("award_title")),

            ("Award Short Title", source.get(
                "award_short_title"
            )),

            ("Award Subtitle", source.get(
                "award_subtitle"
            )),

            ("Award Description", source.get(
                "award_description"
            )),

            ("Award Code", source.get("code")),

            ("Award Slug", source.get("slug")),

            ("Sequence Number", source.get(
                "sequence_number"
            )),


            ("Award Categories", source.get(
                "categories"
            )),

            ("Category Countries", source.get(
                "category_countries"
            )),


            ("Country", source.get("country_name")),

            ("Country Code", source.get(
                "country_code"
            )),

            ("Country Short Code", source.get(
                "country_code_short"
            )),

            ("Location", source.get("location")),

            ("Latitude", source.get("latitude")),

            ("Longitude", source.get("longitude")),


            ("Award Year", source.get(
                "award_year"
            )),

            ("Award Date", source.get(
                "award_on"
            )),

            ("Voting Start Date", source.get(
                "voting_start_date"
            )),

            ("Voting End Date", source.get(
                "voting_end_date"
            )),


            (
                "Voting Active",
                "Yes"
                if source.get("is_voting_active")
                else "No",
            ),

            (
                "Vote Now Available",
                "Yes"
                if source.get("show_vote_now_button")
                else "No",
            ),

            (
                "Vote Now Link",
                source.get("vote_now_link"),
            ),

            (
                "Nomination Available",
                "Yes"
                if source.get("show_nomination_button")
                else "No",
            ),

            (
                "Nomination Link",
                source.get("nomination_link"),
            ),

            (
                "Personal Categories Available",
                "Yes"
                if source.get(
                    "show_personal_categories_button"
                )
                else "No",
            ),

            (
                "Personal Categories Link",
                source.get(
                    "personal_categories_link"
                ),
            ),

            (
                "Corporate Categories Available",
                "Yes"
                if source.get(
                    "show_corporate_categories_button"
                )
                else "No",
            ),

            (
                "Corporate Categories Link",
                source.get(
                    "corporate_categories_link"
                ),
            ),


            (
                "Award Winners Available",
                "Yes"
                if source.get(
                    "show_award_winners_button"
                )
                else "No",
            ),

            (
                "Award Winners Link",
                source.get(
                    "award_winners_link"
                ),
            ),


            (
                "Award Active",
                "Yes"
                if source.get("award_is_active")
                else "No",
            ),

            (
                "Award Detail URL",
                source.get("award_detail_url"),
            ),

            (
                "YouTube",
                source.get("youtube_link"),
            ),
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
        "schema_version": "v6.6",

        "entity_type": model,

        "content_type_id": master.get(
            "content_type_id"
        ),

        "object_id": master.get(
            "object_id"
        ),

        "title": master.get(
            "title"
        ),

        "is_live": master.get(
            "is_live"
        ),

        "created_at": master.get(
            "created_at"
        ),

        "expires_at": master.get(
            "expires_at"
        ),

        "relationships": compact_metadata(
            relationship_meta
        ),
    }
    return document, compact_metadata(metadata)


def resolve_content_types(cur, model_filter: str | None):
    """
    Resolve Django content types that actually exist in the master index.

    When multiple apps contain the same model name, prefer the `base`
    application. The ordering is applied outside the DISTINCT query so
    PostgreSQL does not reject the ORDER BY expression.
    """

    if model_filter:
        cur.execute(
            """
            SELECT
                x.id,
                x.app_label,
                x.model
            FROM (
                SELECT DISTINCT
                    ct.id,
                    ct.app_label,
                    ct.model
                FROM public.django_content_type ct
                JOIN public.master_search_mastersearchindex si
                    ON si.content_type_id = ct.id
                WHERE lower(ct.model) = lower(%s)
            ) x
            ORDER BY
                CASE
                    WHEN lower(x.app_label) = 'base' THEN 0
                    ELSE 1
                END,
                x.id
            """,
            (model_filter,),
        )
    else:
        cur.execute(
            """
            SELECT
                x.id,
                x.app_label,
                x.model
            FROM (
                SELECT DISTINCT
                    ct.id,
                    ct.app_label,
                    ct.model
                FROM public.django_content_type ct
                JOIN public.master_search_mastersearchindex si
                    ON si.content_type_id = ct.id
            ) x
            ORDER BY
                CASE
                    WHEN lower(x.app_label) = 'base' THEN 0
                    ELSE 1
                END,
                x.id
            """
        )

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
                    master["entity_type"] = model
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
                            SET
                                ai_search_text = v.ai_search_text,
                                metadata = v.metadata::jsonb
                            FROM (VALUES %s) AS v(
                                id,
                                ai_search_text,
                                metadata
                            )
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
