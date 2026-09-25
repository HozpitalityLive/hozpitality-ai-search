"""Field extraction and constraint evidence over real search_documents records.

Every path here is written by the migration scripts (see schema_map.py).
Nothing is inferred or invented: a missing field is reported as missing.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .normalization import normalize
from .schema_map import schema


def nested(doc: Any, *path: str) -> Any:
    value = doc
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, dict):
            value = value.get("name") or value.get("title")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def all_values(doc: Any, path: str) -> list[Any]:
    current = [doc]
    for part in path.split("."):
        nxt: list[Any] = []
        for item in current:
            if isinstance(item, dict):
                value = item.get(part)
                if isinstance(value, list):
                    nxt.extend(value)
                elif value is not None:
                    nxt.append(value)
        current = nxt
    return current


def all_texts(doc: Any, *paths: str) -> list[str]:
    out: list[str] = []
    for path in paths:
        for value in all_values(doc, path):
            text = first_text(value)
            if text and text not in out:
                out.append(text)
    return out


# ---------------------------------------------------------------------------
# Level evidence
# ---------------------------------------------------------------------------

# Terms that evidence a level in a title, level or role name. "manager"
# (management) includes more senior leadership: an Executive Chef or Chef de
# Cuisine is a management position.
LEVEL_EVIDENCE: dict[str, tuple[str, ...]] = {
    "senior": ("senior", "sr", "lead", "principal", "head"),
    "junior": ("junior", "jr", "entry level", "entry-level", "commis", "assistant"),
    "mid": ("mid level", "mid-level", "midlevel", "associate"),
    "manager": (
        "manager",
        "management",
        "managerial",
        "head",
        "supervisor",
        "director",
        "executive",
        "chief",
        "chef de cuisine",
        "general manager",
        "gm",
    ),
    "executive": ("executive", "director", "vp", "vice president", "chief", "c-suite"),
    "intern": ("intern", "internship", "trainee", "graduate", "apprentice"),
}

# migrate_jobs: job.levels[] / job.roles[]; migrate_professionals:
# professional.job_level / professional.job_role.
LEVEL_FIELDS = ("job.levels.name", "professional.job_level.name")
ROLE_FIELDS = ("job.roles.name", "professional.job_role.name")


def _has_term(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))


def level_values(doc: dict[str, Any]) -> list[str]:
    return all_texts(doc, *LEVEL_FIELDS)


def level_evidence(doc: dict[str, Any], level: str) -> bool:
    terms = LEVEL_EVIDENCE.get(level, (level,))
    # Professionals' titles are person names, so only jobs use the title.
    candidates = [*level_values(doc), *all_texts(doc, *ROLE_FIELDS)]
    if doc.get("entity_type") != "professional":
        candidates.append(str(doc.get("title") or ""))
    return any(
        _has_term(normalize(value), term) for value in candidates for term in terms
    )


# ---------------------------------------------------------------------------
# Accommodation evidence
# ---------------------------------------------------------------------------

# The migration has no structured accommodation field: evidence can only come
# from the job's own text (title, description) and tags.
_ACCOMMODATION_POSITIVE = re.compile(
    r"\b(?:accommodation|housing|lodging|staff\s+quarters|living\s+quarters)\b"
    r"(?!\s+(?:is\s+)?not\s+(?:provided|available|included|offered))",
    re.I,
)
_ACCOMMODATION_NEGATIVE = re.compile(
    r"\b(?:no|without|excluding|excludes?)\s+(?:staff\s+)?(?:accommodation|housing|lodging)\b"
    r"|\b(?:accommodation|housing|lodging)\s+(?:is\s+)?not\s+(?:provided|available|included|offered)\b",
    re.I,
)


def accommodation_status(doc: dict[str, Any]) -> bool | None:
    """True/False when the record's own text says so, None when unknown.

    ai_search_text is deliberately not used: for jobs it also contains the
    COMPANY description, which is not evidence about the job.
    """
    text = " ".join(
        str(v)
        for v in (
            doc.get("title"),
            doc.get("summary"),
            nested(doc, "job", "description"),
            *all_values(doc, "job.tags"),
        )
        if isinstance(v, str)
    )
    if _ACCOMMODATION_NEGATIVE.search(text):
        return False
    if _ACCOMMODATION_POSITIVE.search(text):
        return True
    return None


# ---------------------------------------------------------------------------
# Strict filter evidence
# ---------------------------------------------------------------------------

EVIDENCE_FILTERS = {"level", "accommodation"}


def satisfies_strict(
    doc: dict[str, Any], filters: dict[str, Any], strict: set[str]
) -> bool:
    """True when the document has positive evidence for every strict filter."""
    if filters.get("accommodation") is True and accommodation_status(doc) is not True:
        return False
    if "level" in strict and filters.get("level"):
        if not level_evidence(doc, str(filters["level"])):
            return False
    return True


# ---------------------------------------------------------------------------
# Record fields (results, comparison and detail views)
# ---------------------------------------------------------------------------

_YEARS_RE = re.compile(
    r"\b(\d{1,2})\s*(?:\+|plus)?\s*(?:-|–|to)?\s*(\d{1,2})?\s*\+?\s*(?:years?|yrs?)\b",
    re.I,
)


def experience_value(doc: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (value, source). There is no numeric experience field in the
    migration; years are only mentioned in the record's own text."""
    text = " ".join(
        str(v)
        for v in (
            doc.get("summary"),
            nested(doc, "job", "description"),
            doc.get("description"),
            nested(doc, "professional", "resume_title"),
        )
        if isinstance(v, str)
    )
    m = _YEARS_RE.search(text)
    if m:
        return m.group(0).strip(), "description"
    return None, None


def experience_years(doc: dict[str, Any]) -> float | None:
    value, _ = experience_value(doc)
    if not value:
        return None
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", value)]
    return max(numbers) if numbers else None


def salary_value(doc: dict[str, Any]) -> str | None:
    """job.salary.{description, range.name, currency.code}; product pricing."""
    description = first_text(nested(doc, "job", "salary", "description"))
    if description:
        return description
    range_name = first_text(nested(doc, "job", "salary", "range", "name"))
    currency = first_text(nested(doc, "job", "salary", "currency", "code")) or ""
    if range_name:
        return f"{currency} {range_name}".strip()
    price = nested(doc, "pricing", "price")
    if isinstance(price, (int, float)) and not isinstance(price, bool):
        code = first_text(nested(doc, "pricing", "currency", "code")) or ""
        return f"{code} {price:g}".strip()
    return None


def company_name(doc: dict[str, Any]) -> str | None:
    """The organisation a record belongs to (never the record itself)."""
    entity = doc.get("entity_type")
    if entity == "company":
        return None
    if entity == "professional":
        return first_text(
            nested(doc, "professional", "current_company", "name"),
            nested(doc, "professional", "current_company_text"),
        )
    if entity == "product":
        return first_text(
            nested(doc, "seller", "company_name"), nested(doc, "seller", "name")
        )
    if entity == "article":
        return first_text(nested(doc, "author", "name"))
    return first_text(nested(doc, "company", "name"))


def location_parts(doc: dict[str, Any]) -> dict[str, str | None]:
    """City/country from each module's own location fields."""
    raw = doc.get("location")
    if isinstance(raw, str):  # awards: location string + top-level country
        country = doc.get("country") if isinstance(doc.get("country"), dict) else {}
        return {
            "city": first_text(raw),
            "country": first_text(country.get("name")),
            "address": None,
        }
    location = raw if isinstance(raw, dict) else {}
    country = (
        location.get("country") if isinstance(location.get("country"), dict) else {}
    )
    country_name = country.get("name") or country.get("ac_name") or country.get("code")
    if not country_name:
        for key in ("countries", "available_in_countries"):  # articles, products
            names = [
                c.get("name")
                for c in location.get(key) or []
                if isinstance(c, dict) and c.get("name")
            ]
            if names:
                country_name = ", ".join(names[:3])
                break
    return {
        "city": first_text(
            location.get("city"),
            location.get("prime_city"),
            location.get("current_location"),
        ),
        "country": first_text(country_name),
        "address": first_text(location.get("address")),
    }


def posted_date(doc: dict[str, Any]) -> str | None:
    item = schema(doc.get("entity_type"))
    for path in item.created if item else ("created_at",):
        for value in all_values(doc, path):
            if isinstance(value, datetime):
                return value.date().isoformat()
            if isinstance(value, str) and value.strip():
                return value.strip()[:10]
    return None


def employment_type(doc: dict[str, Any]) -> str | None:
    return first_text(
        nested(doc, "job", "employment_type", "name"),
        nested(doc, "job", "job_type", "name"),
    )


def level_text(doc: dict[str, Any]) -> str | None:
    values = level_values(doc)
    return ", ".join(values) if values else None


def category_name(doc: dict[str, Any]) -> str | None:
    item = schema(doc.get("entity_type"))
    names = all_texts(
        doc, *(item.category if item else ("category.name", "categories.name"))
    )
    return ", ".join(names[:3]) if names else None
