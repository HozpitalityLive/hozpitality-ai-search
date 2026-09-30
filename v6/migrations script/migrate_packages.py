from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime, timezone

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "mongoAdmin")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "search_documents")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("package-migration")


PRO_PACKAGE_ID = "pro"
PACKAGE_OVERVIEW_ID = "package-overview"
STARTER_PACKAGE_ID = "starter"
GROWTH_PACKAGE_ID = "growth"
ENTERPRISE_PACKAGE_ID = "enterprise"


PRO_MONTHLY = {
    "period": "monthly",
    "price": 10,
    "currency": "USD",
    "discount": 0,
    "label": "Monthly - USD 10",
    "industry_articles": 5,
    "priority_messages": 20,
    "non_follower_contact_views": 60,
}
PRO_QUARTERLY = {
    "period": "quarterly",
    "price": 27,
    "currency": "USD",
    "discount": 10,
    "label": "Quarterly - USD 27 (Discount: 10%)",
    "industry_articles": 15,
    "priority_messages": 20,
    "non_follower_contact_views": 60,
}
PRO_YEARLY = {
    "period": "yearly",
    "price": 90,
    "currency": "USD",
    "discount": 25,
    "label": "Yearly - USD 90 (Discount: 25%)",
    "industry_articles": 60,
    "priority_messages": 20,
    "non_follower_contact_views": 240,
}

# Company plans exactly as shown in the Packages UI.
COMPANY_COMMON = {
    "tax_note": "Exclusive of 0% tax",
    "buy_cta": "Buy Now",
}

STARTER_MONTHLY = {
    **COMPANY_COMMON,
    "period": "monthly", "price": 120, "currency": "USD", "discount": 0,
    "label": "Monthly - USD 120",
    "cv_searches_downloads": 100,
    "profile_views_messages": 100,
}
STARTER_QUARTERLY = {
    **COMPANY_COMMON,
    "period": "quarterly", "price": 324, "currency": "USD", "discount": 10,
    "label": "Quarterly - USD 324 (Discount: 10%)",
    "cv_searches_downloads": 300,
    "profile_views_messages": 300,
}
STARTER_YEARLY = {
    **COMPANY_COMMON,
    "period": "yearly", "price": 1080, "currency": "USD", "discount": 25,
    "label": "Yearly - USD 1080 (Discount: 25%)",
    "cv_searches_downloads": 1200,
    "profile_views_messages": 1200,
}

GROWTH_MONTHLY = {
    **COMPANY_COMMON,
    "period": "monthly", "price": 250, "currency": "USD", "discount": 0,
    "label": "Monthly - USD 250",
    "cv_searches_downloads": 200,
    "profile_views_messages": 200,
}
GROWTH_QUARTERLY = {
    **COMPANY_COMMON,
    "period": "quarterly", "price": 675, "currency": "USD", "discount": 10,
    "label": "Quarterly - USD 675 (Discount: 10%)",
    "cv_searches_downloads": 600,
    "profile_views_messages": 600,
}
GROWTH_YEARLY = {
    **COMPANY_COMMON,
    "period": "yearly", "price": 2250, "currency": "USD", "discount": 25,
    "label": "Yearly - USD 2250 (Discount: 25%)",
    "cv_searches_downloads": 2400,
    "profile_views_messages": 2400,
}

ENTERPRISE_MONTHLY = {
    **COMPANY_COMMON,
    "period": "monthly", "price": 400, "currency": "USD", "discount": 0,
    "label": "Monthly - USD 400",
}
ENTERPRISE_QUARTERLY = {
    **COMPANY_COMMON,
    "period": "quarterly", "price": 1080, "currency": "USD", "discount": 10,
    "label": "Quarterly - USD 1080 (Discount: 10%)",
}
ENTERPRISE_YEARLY = {
    **COMPANY_COMMON,
    "period": "yearly", "price": 3600, "currency": "USD", "discount": 25,
    "label": "Yearly - USD 3600 (Discount: 25%)",
}

FREE_COMPANY_PACKAGE_ID = "free-company"
def _search_keywords(*values: str) -> list[str]:
    """Create stable, searchable keywords while preserving useful phrases."""
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if not value:
            continue
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+\-]{1,}", value.lower()):
            if token not in seen:
                seen.add(token)
                result.append(token)

    return result


def _faq_document(
    *,
    document_id: str,
    question: str,
    aliases: list[str],
    answer: str,
    package: dict,
) -> dict:
    now = datetime.now(timezone.utc)

    normalized_aliases = list(dict.fromkeys([question, *aliases]))
    keywords = _search_keywords(
        question,
        " ".join(normalized_aliases),
        answer,
        package.get("name", ""),
        package.get("audience", ""),
    )

    return {
        "_id": f"faq:package:{document_id}",
        "entity_type": "faq",
        "question": question,
        "answer": answer,
        "search_aliases": normalized_aliases,
        "search_keywords": keywords,
        "ai_search_text": f"FAQ Question: {question}\nFAQ Answer: {answer}",
        "status": "active",
        "flags": {
            "show_on_landing": False,
            "is_popular": False,
        },
        "package_details": package,
        "dates": {
            "created_at": now,
            "updated_at": now,
        },
        "embedding": {
            "dimensions": 384,
            "status": "pending",
        },
        "source": {
            "model": "Package",
            "table": "package_seed",
            "object_id": document_id,
        },
        "migration": {
            "source": "package_seed",
            "migrated_at": now,
        },
        "schema_version": 1,
    }


def _company_package(
    *,
    package_id: str,
    name: str,
    slug: str,
    subtitle: str,
    best_for: str,
    plans: list[dict],
    base_included_heading: str,
    base_included: list[str],
    boost_heading: str,
    boost_features: list[str],
    aliases: list[str],
) -> dict:
    return {
        "name": name,
        "slug": slug,
        "audience": "companies",
        "audience_label": "For Companies",
        "category": "company",
        "status": "active",
        "subtitle": subtitle,
        "best_for": best_for,
        "plans": plans,
        "included": {
            "heading": base_included_heading,
            "items": base_included,
        },
        "hiring_boost": {
            "heading": boost_heading,
            "items": boost_features,
        },
        "aliases": aliases,
    }


def _plan_line(plan: dict, *items: str) -> str:
    details = [
        f"{plan['label']}.",
        *items,
    ]
    return " ".join(details)


def build_package_documents() -> list[dict]:
    """Return all professional and company package FAQ records for MongoDB."""

    overview_answer = (
        "Hozpitality Packages has options for hospitality professionals and companies. "
        "For professionals, Hozpitality Pro is available. For companies, the available options are Free Account, Starter Plan, Growth Plan, and Enterprise Plan. "
        "Starter, Growth, and Enterprise paid plans are shown with monthly, quarterly, and yearly billing. "
        "Company paid-plan prices are exclusive of 0% tax."
    )

    pro_package = {
        "name": "Hozpitality Pro",
        "slug": "pro",
        "audience": "professionals",
        "audience_label": "For Professionals",
        "category": "professional",
        "status": "active",
        "plans": [PRO_MONTHLY, PRO_QUARTERLY, PRO_YEARLY],
        "features": [
            {
                "name": "Get Seen Before Other Candidates",
                "details": [
                    "Appear at the top of recruiter searches before free profiles.",
                    "Highlighted PRO profile attracts more profile views.",
                    "Free profiles appear after PRO members.",
                ],
            },
            {
                "name": "Verified Hospitality Professional",
                "details": [
                    "Blue Tick builds instant trust with recruiters.",
                    "Higher chance of shortlisting and recruiter responses.",
                    "Verification is available only for PRO users.",
                ],
            },
            {
                "name": "Message Employers Directly",
                "details": [
                    "Send 20 priority messages to employers.",
                    "Messages are marked as PRO for faster attention.",
                    "View contact details of non-followers according to the billing-period limit.",
                    "Free users cannot message employers.",
                ],
            },
            {
                "name": "Apply as a Featured Candidate",
                "details": [
                    "PRO applications are shown before free candidates.",
                    "Stand out in high-competition jobs.",
                    "Featured applications are PRO-only.",
                ],
            },
            {
                "name": "Build Your Professional Brand",
                "details": [
                    "Publish industry articles according to the billing-period limit.",
                    "Articles are linked to the verified professional profile.",
                    "Personal branding tools are PRO-only.",
                ],
            },
            {
                "name": "Recruiters Are Viewing Profiles",
                "details": [
                    "See who viewed your profile.",
                    "Know when employers show interest.",
                    "Respond faster to opportunities.",
                ],
            },
        ],
        "free_users_miss": [
            "Priority visibility in recruiter searches",
            "Direct employer messaging",
            "Verified Blue Tick trust badge",
            "Recruiter interest and profile-view insights",
        ],
    }

    free_company = {
        "name": "Free Account",
        "slug": "free-company",
        "audience": "companies",
        "audience_label": "For Companies",
        "category": "company",
        "status": "active",
        "best_for": "First-time employers & visibility",
        "plans": [],
        "included": [
            "Blue Tick Verification",
            "Branded Mini Site",
            "Unlimited Job Postings & Renewals",
            "Receive & Contact Applications (Dashboard)",
            "Unlimited Marketplace Listings",
            "Unlimited Articles, News & Events",
            "SEO / Web Page Visibility",
        ],
        "limitations": [
            "No Email Delivery of Applications",
            "No WhatsApp Outreach",
            "No CV Downloads",
            "No Featured Exposure",
            "No White-Label Career Page",
            "No Social Media Sharing",
            "No Dedicated Account Manager",
        ],
    }

    starter = _company_package(
        package_id=STARTER_PACKAGE_ID,
        name="Starter Plan",
        slug="starter",
        subtitle="Best for: cafes, restaurants & boutique hotels",
        best_for="cafes, restaurants & boutique hotels",
        plans=[STARTER_MONTHLY, STARTER_QUARTERLY, STARTER_YEARLY],
        base_included_heading="Everything in Free, plus",
        base_included=[
            "Blue Tick Verification",
            "Branded Mini Site",
            "Unlimited Job Postings & Renewals",
            "Receive & Contact Applications (Dashboard)",
            "Unlimited Marketplace Listings",
            "Unlimited Articles, News & Events",
            "White-Label Career Page",
        ],
        boost_heading="Starter Hiring Boost",
        boost_features=[
            "Monthly: 100 CV Searches & Downloads; Contact 100 Profiles via Messages.",
            "Quarterly: 300 CV Searches & Downloads; Contact 300 Profiles via Messages.",
            "Yearly: 1200 CV Searches & Downloads; Contact 1200 Profiles via Messages.",
            "Job Posts shown to Non-Followers",
            "Receive Job Applications via Email",
            "Accept Walk-In Interviews",
            "Download Applications (Excel / CSV)",
        ],
        aliases=[
            "Starter company package",
            "Starter Plan for companies",
            "What is the Starter Plan?",
            "How much is Starter?",
            "Starter package price",
            "Starter hiring boost",
        ],
    )

    growth = _company_package(
        package_id=GROWTH_PACKAGE_ID,
        name="Growth Plan",
        slug="growth",
        subtitle="Best for: multi-location brands and aggressive hiring",
        best_for="multi-location brands and aggressive hiring",
        plans=[GROWTH_MONTHLY, GROWTH_QUARTERLY, GROWTH_YEARLY],
        base_included_heading="Everything in Starter, plus",
        base_included=[
            "Blue Tick Verification",
            "Branded Mini Site",
            "Unlimited Job Postings & Renewals",
            "Receive & Contact Applications",
            "Unlimited Marketplace Listings",
            "Unlimited Articles, News & Events",
            "White-Label Career Page",
        ],
        boost_heading="Growth Hiring Accelerator",
        boost_features=[
            "Monthly: 200 CV Searches & Downloads; Contact 200 Profiles via Messages.",
            "Quarterly: 600 CV Searches & Downloads; Contact 600 Profiles via Messages.",
            "Yearly: 2400 CV Searches & Downloads; Contact 2400 Profiles via Messages.",
            "Higher Job Visibility to Non-Followers",
            "Faster Hiring Cycles",
        ],
        aliases=[
            "Growth company package",
            "Growth Plan for companies",
            "What is the Growth Plan?",
            "How much is Growth?",
            "Growth package price",
            "Growth hiring accelerator",
        ],
    )

    enterprise = _company_package(
        package_id=ENTERPRISE_PACKAGE_ID,
        name="Enterprise Plan",
        slug="enterprise",
        subtitle="Best for: Hotel groups, chains & leadership brands",
        best_for="Hotel groups, chains & leadership brands",
        plans=[ENTERPRISE_MONTHLY, ENTERPRISE_QUARTERLY, ENTERPRISE_YEARLY],
        base_included_heading="Everything in Growth, plus",
        base_included=[
            "Unlimited Job Postings & Renewals",
            "Unlimited CV Searches & Downloads",
            "Jobs Shown to All Non-Followers",
            "Maximum Hiring Visibility",
            "Branded Career Pages (White-Label)",
            "We Post and Share your News/Articles/Announcements",
        ],
        boost_heading="Enterprise Brand Boost (Annual Accounts only)",
        boost_features=[
            "WhatsApp Outreach",
            "5 Featured Post Credits per Month",
            "Featured in Daily, Weekly & Monthly Newsletters",
            "Top Management Interviews",
            "Social Media & WhatsApp Group Promotion",
            "API Integrations & Spider Services",
            "Premium Priority Support",
            "Event Sponsorship Discounts",
            "Dedicated Account Manager",
        ],
        aliases=[
            "Enterprise company package",
            "Enterprise Plan for companies",
            "What is the Enterprise Plan?",
            "How much is Enterprise?",
            "Enterprise package price",
            "Enterprise Brand Boost",
        ],
    )

    company_packages = [starter, growth, enterprise]

    company_answers = {
        "free-company": (
            "The Hozpitality Free Account is a company option for first-time employers and visibility. "
            "It includes Blue Tick Verification, a Branded Mini Site, Unlimited Job Postings & Renewals, "
            "Receive & Contact Applications through the Dashboard, Unlimited Marketplace Listings, "
            "Unlimited Articles, News & Events, and SEO / Web Page Visibility. "
            "Limitations are no Email Delivery of Applications, no WhatsApp Outreach, no CV Downloads, "
            "no Featured Exposure, no White-Label Career Page, no Social Media Sharing, and no Dedicated Account Manager."
        ),
        "starter": (
            "The Hozpitality Starter Plan is for companies, especially cafes, restaurants and boutique hotels. "
            "Pricing is Monthly USD 120, Quarterly USD 324 with a 10% discount, and Yearly USD 1080 with a 25% discount. "
            "All company prices are exclusive of 0% tax. It includes Blue Tick Verification, a Branded Mini Site, "
            "Unlimited Job Postings & Renewals, Receive & Contact Applications through the Dashboard, Unlimited Marketplace Listings, "
            "Unlimited Articles, News & Events, and a White-Label Career Page. "
            "Starter Hiring Boost limits are 100 CV Searches & Downloads and contact with 100 profiles via messages monthly; "
            "300 and 300 quarterly; and 1200 and 1200 yearly. It also includes job posts shown to non-followers, job applications via email, "
            "walk-in interview acceptance, and application downloads in Excel/CSV."
        ),
        "growth": (
            "The Hozpitality Growth Plan is for companies, especially multi-location brands and businesses with aggressive hiring needs. "
            "Pricing is Monthly USD 250, Quarterly USD 675 with a 10% discount, and Yearly USD 2250 with a 25% discount. "
            "All company prices are exclusive of 0% tax. It includes the Starter-level company features: Blue Tick Verification, Branded Mini Site, "
            "Unlimited Job Postings & Renewals, Receive & Contact Applications, Unlimited Marketplace Listings, Unlimited Articles, News & Events, "
            "and a White-Label Career Page. Growth Hiring Accelerator limits are 200 CV Searches & Downloads and contact with 200 profiles via messages monthly; "
            "600 and 600 quarterly; and 2400 and 2400 yearly. It also provides higher job visibility to non-followers and faster hiring cycles."
        ),
        "enterprise": (
            "The Hozpitality Enterprise Plan is for companies, especially hotel groups, chains and leadership brands. "
            "Pricing is Monthly USD 400, Quarterly USD 1080 with a 10% discount, and Yearly USD 3600 with a 25% discount. "
            "All company prices are exclusive of 0% tax. Enterprise includes everything in Growth plus Unlimited Job Postings & Renewals, "
            "Unlimited CV Searches & Downloads, Jobs Shown to All Non-Followers, Maximum Hiring Visibility, Branded Career Pages (White-Label), "
            "and posting/sharing of company News, Articles and Announcements. The Enterprise Brand Boost is marked 'Annual Accounts only' "
            "and includes WhatsApp Outreach, 5 Featured Post Credits per Month, inclusion in Daily/Weekly/Monthly Newsletters, Top Management Interviews, "
            "Social Media & WhatsApp Group Promotion, API Integrations & Spider Services, Premium Priority Support, Event Sponsorship Discounts, "
            "and a Dedicated Account Manager."
        ),
    }

    records = [
        _faq_document(
            document_id=PACKAGE_OVERVIEW_ID,
            question="What packages does Hozpitality offer?",
            aliases=[
                "What are the Hozpitality packages?",
                "Tell me about Hozpitality packages",
                "What package plans are available on Hozpitality?",
                "Does Hozpitality have packages for professionals and companies?",
                "What company packages are available?",
                "What professional package is available?",
                "Hozpitality pricing plans",
            ],
            answer=overview_answer,
            package={
                "type": "package_overview",
                "audiences": ["professionals", "companies"],
                "professional_packages": ["Hozpitality Pro"],
                "company_packages": ["Free Account", "Starter Plan", "Growth Plan", "Enterprise Plan"],
            },
        ),
        _faq_document(
            document_id=PRO_PACKAGE_ID,
            question="What is the Hozpitality Pro package for professionals?",
            aliases=[
                "What is Hozpitality Pro?",
                "Tell me about the Pro package",
                "What does the Pro package include?",
                "How much does Hozpitality Pro cost?",
                "What are the benefits of Hozpitality Pro?",
                "Is the Pro package for professionals?",
                "What do I get with Hozpitality Pro?",
            ],
            answer=(
                "The Hozpitality Pro package is for hospitality professionals only. "
                "It includes priority visibility in recruiter searches, a Verified Hospitality Professional Blue Tick, "
                "direct messaging to employers, Featured Candidate applications, professional branding through industry articles, "
                "and profile-view insights. Pricing: Monthly USD 10; Quarterly USD 27 with a 10% discount; Yearly USD 90 with a 25% discount. "
                "All Pro billing periods include 20 priority messages to employers. Non-follower contact details can be viewed for 60 monthly/quarterly "
                "and 240 yearly. Industry article publishing is up to 5 monthly, 15 quarterly, and 60 yearly. "
                "Free users do not receive priority recruiter-search visibility, direct employer messaging, the Verified Blue Tick trust badge, "
                "or recruiter interest and profile-view insights."
            ),
            package=pro_package,
        ),
    ]

    records.append(
        _faq_document(
            document_id=FREE_COMPANY_PACKAGE_ID,
            question="What is the Hozpitality Free Account for companies?",
            aliases=[
                "What does the Free Account include?",
                "What is the free company package?",
                "Is there a free package for companies?",
                "Hozpitality free employer account",
                "What are the limitations of the Free Account?",
            ],
            answer=company_answers["free-company"],
            package=free_company,
        )
    )

    for pkg, answer in [
        (starter, company_answers["starter"]),
        (growth, company_answers["growth"]),
        (enterprise, company_answers["enterprise"]),
    ]:
        records.append(
            _faq_document(
                document_id=pkg["slug"],
                question=f"What is the Hozpitality {pkg['name']} for companies?",
                aliases=pkg["aliases"],
                answer=answer,
                package=pkg,
            )
        )

    return records

def get_collection(client):
    return client[MONGO_DB][MONGO_COLLECTION]


def migrate(*, dry_run: bool = False) -> int:
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI environment variable is required.")

    documents = build_package_documents()

    logger.info("MongoDB database: %s", MONGO_DB)
    logger.info("MongoDB collection: %s", MONGO_COLLECTION)
    logger.info("Package records to seed: %s", len(documents))

    if dry_run:
        for document in documents:
            logger.info(
                "DRY RUN: would upsert %s (%s)",
                document["_id"],
                document["question"],
            )
        return len(documents)

    from pymongo import MongoClient

    client = MongoClient(MONGO_URI)
    try:
        client.admin.command("ping")
        collection = get_collection(client)

        for document in documents:
            result = collection.replace_one(
                {"_id": document["_id"]},
                document,
                upsert=True,
            )
            action = "inserted" if result.upserted_id is not None else "updated"
            logger.info("%s %s", action.capitalize(), document["_id"])

        logger.info("Package migration completed successfully.")
        return len(documents)
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed Hozpitality package knowledge into MongoDB search_documents."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show records without connecting to MongoDB.",
    )
    args = parser.parse_args()

    migrate(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
