"""Normalized search result contract (same shape for every module).

{
  "id": "job:101",                      # MongoDB _id (always a real record)
  "entity_type": "job", "entity_id": "101", "doc_id": "job:101",
  "title": "...", "slug": "...",
  "url": "https://..." | null,          # clickable page for the record
  "url_source": "record" | "template" | null,
  "description": "...", "snippet": "...",
  "location": {"city": ..., "country": ..., "address": ...},
  "company": "ABC Hospitality",         # display name (kept for compatibility)
  "company_ref": {"id": 1001, "name": ..., "slug": ..., "url": ... | null},
  "category": "...",
  "external_url": ..., "external_label": ...,   # e.g. company website, event website
  "image": ...,
  "metadata": {...},                    # real, module-specific facts
  "match_type": "exact" | "lexical" | "fuzzy" | "semantic" | "browse" | "related",
  "score": 0, "matched_by": [...]
}

URLs
----
The migration stores a page URL for awards only (links.detail =
base_awards.award_detail_url). Every other module stores a slug. The site's
route patterns are not part of the migration, so they are NOT guessed: an
operator configures them (PUBLIC_URL_TEMPLATES / URL_TEMPLATE_<ENTITY>) and
the URL is built only when every placeholder has a real value. Without a
template a result has `url: null` and the UI shows it as plain text.
"""

from __future__ import annotations

import re
from typing import Any

from . import evidence as ev
from .config import settings
from .schema_map import schema
from .security import clean_untrusted_text, safe_url

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def fill_template(template: str | None, values: dict[str, Any]) -> str | None:
    """Fill {slug}/{id}/... only when every placeholder has a real value."""
    if not template:
        return None
    names = _PLACEHOLDER_RE.findall(template)
    filled = template
    for name in names:
        value = values.get(name)
        if value is None or str(value).strip() == "":
            return None
        filled = filled.replace(
            "{" + name + "}", re.sub(r"[^A-Za-z0-9._~-]", "-", str(value).strip())
        )
    return safe_url(filled)


def record_url(doc: dict[str, Any]) -> tuple[str | None, str | None]:
    """(url, source) for a document."""
    entity = str(doc.get("entity_type") or "")
    item = schema(entity)
    for path in item.record_url if item else ():
        for value in ev.all_values(doc, path):
            url = safe_url(value)
            if url:
                return url, "record"
    values = {"slug": doc.get("slug"), "id": (doc.get("source") or {}).get("object_id")}
    url = fill_template(settings.url_templates.get(entity), values)
    return (url, "template") if url else (None, None)


def company_ref(doc: dict[str, Any]) -> dict[str, Any] | None:
    """The owning organisation with its own (template) URL when available."""
    entity = str(doc.get("entity_type") or "")
    item = schema(entity)
    if not item or not item.company_id:
        return None
    ids = ev.all_values(doc, item.company_id)
    name = ev.company_name(doc)
    if not ids and not name:
        return None
    ref: dict[str, Any] = {"id": ids[0] if ids else None, "name": name}
    slugs = ev.all_values(doc, item.company_slug) if item.company_slug else []
    slug = slugs[0] if slugs else None
    if slug:
        ref["slug"] = slug
    # Sellers/authors are user accounts that may be companies or people; only
    # link company accounts to the company route.
    user_type = ev.first_text(
        ev.nested(doc, "seller", "user_type"), ev.nested(doc, "author", "type")
    )
    template_entity = (
        "company" if entity in {"job", "event"} or user_type == "company" else None
    )
    ref["url"] = fill_template(
        settings.url_templates.get(template_entity or ""),
        {"slug": slug, "id": ref["id"]},
    )
    return ref


def external_link(doc: dict[str, Any]) -> tuple[str | None, str | None]:
    item = schema(doc.get("entity_type"))
    for path, label in item.external_urls if item else ():
        for value in ev.all_values(doc, path):
            url = safe_url(value, base_url="")
            if url and url.startswith(("http://", "https://")):
                return url, label
    return None, None


def display_title(doc: dict[str, Any]) -> str:
    item = schema(doc.get("entity_type"))
    for path in item.title if item else ("title", "question"):
        text = ev.first_text(*ev.all_values(doc, path))
        if text:
            if doc.get("entity_type") == "faq":
                # migrate_faqs keeps numbering ("1. How do I ...") in question
                # and strips it in search_aliases; display the clean question.
                text = re.sub(r"^\s*\d+[.)]\s*", "", text)
            return text
    return "Untitled result"


def description(doc: dict[str, Any]) -> str | None:
    item = schema(doc.get("entity_type"))
    for path in item.description if item else ("description", "summary"):
        text = ev.first_text(*ev.all_values(doc, path))
        if text:
            return text
    return None


def image(doc: dict[str, Any]) -> str | None:
    item = schema(doc.get("entity_type"))
    for path in item.images if item else ():
        for value in ev.all_values(doc, path):
            url = safe_url(value, base_url="")
            if url and url.startswith(("http://", "https://")):
                return url
    return None


def metadata(doc: dict[str, Any]) -> dict[str, Any]:
    """Compact, real, module-specific facts for display and comparison."""
    entity = doc.get("entity_type")
    data: dict[str, Any] = {}

    def put(key: str, value: Any) -> None:
        if value not in (None, "", [], {}):
            data[key] = value

    if entity == "job":
        put("level", ev.level_text(doc))
        put("employment_type", ev.employment_type(doc))
        put("salary", ev.salary_value(doc))
        put("roles", ev.all_texts(doc, "job.roles.name"))
        put("posted", ev.posted_date(doc))
    elif entity == "professional":
        put("role", ev.first_text(ev.nested(doc, "professional", "job_role")))
        put("level", ev.level_text(doc))
        put("skills", ev.all_texts(doc, "professional.skills.name")[:5])
    elif entity == "company":
        put("industries", ev.all_texts(doc, "company.industries.name"))
        put("is_supplier", bool(ev.nested(doc, "company", "is_supplier")))
        put(
            "supplier_categories", ev.all_texts(doc, "company.supplier_categories.name")
        )
        put("company_size", ev.nested(doc, "metadata", "no_of_employees"))
    elif entity == "product":
        put("categories", ev.all_texts(doc, "categories.name"))
        put("price", ev.salary_value(doc))
        put("seller", ev.first_text(ev.nested(doc, "seller", "name")))
    elif entity == "article":
        put("category", ev.first_text(ev.nested(doc, "category")))
        put("published", ev.posted_date(doc))
    elif entity == "event":
        put("starts", ev.posted_date(doc))
        put("status", doc.get("status"))
        put("event_type", doc.get("event_type"))
    elif entity == "award":
        put("year", doc.get("year"))
        put("categories", ev.all_texts(doc, "categories.name"))
    elif entity == "faq":
        put("answer", clean_untrusted_text(doc.get("answer"), 2000))
    if ev.nested(doc, "metadata", "verified") is True:
        data["verified"] = True
    return data


def match_type(matched: list[str]) -> str:
    labels = set(matched)
    if "related_result" in labels:
        return "related"
    if labels & {
        "exact_title",
        "exact_alias",
        "exact_keyword",
        "exact_category",
        "exact_company",
        "exact_person",
    }:
        return "exact"
    if "browse" in labels:
        return "browse"
    if labels & {
        "phrase_title",
        "alias",
        "keyword",
        "category",
        "company",
        "all_tokens",
        "partial_tokens",
        "canonical_text",
    }:
        return "lexical"
    if "semantic" in labels:
        return "semantic"
    if "fuzzy" in labels:
        return "fuzzy"
    return "lexical"


def result_payload(
    doc: dict[str, Any],
    score: float,
    matched: list[str],
    corrected_query: str | None = None,
) -> dict:
    entity_id = str((doc.get("source") or {}).get("object_id") or doc.get("_id") or "")
    url, url_source = record_url(doc)
    external_url, external_label = external_link(doc)
    text = description(doc)
    matched = list(dict.fromkeys(matched))
    return {
        "id": str(doc.get("_id") or ""),
        "entity_type": str(doc.get("entity_type") or ""),
        "entity_id": entity_id,
        "doc_id": str(doc.get("_id") or ""),
        "title": display_title(doc),
        "slug": doc.get("slug") if isinstance(doc.get("slug"), str) else None,
        "url": url,
        "url_source": url_source,
        "description": text,
        "snippet": clean_untrusted_text(text, max_chars=240),
        "company": ev.company_name(doc),
        "company_ref": company_ref(doc),
        "location": ev.location_parts(doc),
        "category": ev.category_name(doc),
        "external_url": external_url,
        "external_label": external_label,
        "image": image(doc),
        "metadata": metadata(doc),
        "match_type": match_type(matched),
        "score": round(score, 4),
        "matched_by": matched,
        "corrected_query": corrected_query,
    }
