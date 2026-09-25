"""Field extraction and constraint evidence over real search_documents records.

Everything here reads values that exist on the MongoDB document. Nothing is
inferred or invented: a missing field is reported as missing (None).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .normalization import normalize


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


# ---------------------------------------------------------------------------
# Level evidence
# ---------------------------------------------------------------------------

# Terms that evidence a level in a title or a structured level field.
# "manager" (management) deliberately includes more senior leadership roles:
# an Executive Chef or Chef de Cuisine is a management position.
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

LEVEL_FIELDS = (
    "metadata.job_level",
    "job.level.name",
    "job.job_level.name",
    "job_level.name",
    "professional.job_level.name",
    "level",
)


def _has_term(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))


def level_values(doc: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for path in LEVEL_FIELDS:
        for value in all_values(doc, path):
            text = first_text(value)
            if text:
                values.append(text)
    return values


def level_evidence(doc: dict[str, Any], level: str) -> bool:
    terms = LEVEL_EVIDENCE.get(level, (level,))
    title = normalize(" ".join(str(doc.get(k) or "") for k in ("title", "short_title")))
    if any(_has_term(title, term) for term in terms):
        return True
    return any(
        any(_has_term(normalize(value), term) for term in terms)
        for value in level_values(doc)
    )


# ---------------------------------------------------------------------------
# Accommodation evidence
# ---------------------------------------------------------------------------

ACCOMMODATION_FIELDS = (
    "job.accommodation",
    "job.provides_accommodation",
    "accommodation",
    "provides_accommodation",
    "metadata.accommodation",
    "metadata.provides_accommodation",
    "benefits",
    "job.benefits",
    "metadata.benefits",
)

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
_TRUE_STRINGS = {
    "true",
    "yes",
    "y",
    "1",
    "provided",
    "available",
    "included",
    "offered",
}
_FALSE_STRINGS = {"false", "no", "n", "0", "not provided", "not available", "none"}


def accommodation_status(doc: dict[str, Any]) -> bool | None:
    """True/False from structured data or explicit text, None when unknown."""
    for path in ACCOMMODATION_FIELDS:
        for value in all_values(doc, path):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                norm = normalize(value)
                if norm in _TRUE_STRINGS:
                    return True
                if norm in _FALSE_STRINGS:
                    return False
                if _ACCOMMODATION_NEGATIVE.search(value):
                    return False
                if _ACCOMMODATION_POSITIVE.search(value):
                    return True
    text = " ".join(
        str(v)
        for v in (doc.get("title"), doc.get("description"), doc.get("ai_search_text"))
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
# Record fields (used by results, comparison and detail views)
# ---------------------------------------------------------------------------

_YEARS_RE = re.compile(
    r"\b(\d{1,2})\s*(?:\+|plus)?\s*(?:-|–|to)?\s*(\d{1,2})?\s*\+?\s*(?:years?|yrs?)\b",
    re.I,
)


def experience_value(doc: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (value, source) where source is "field" or "description"."""
    for path in (
        "job.experience",
        "job.experience_years",
        "job.years_experience",
        "metadata.experience",
        "metadata.experience_years",
        "professional.experience_years",
        "professional.years_experience",
        "professional.experience",
        "experience",
        "experience_years",
    ):
        for value in all_values(doc, path):
            text = first_text(value)
            if text:
                if re.fullmatch(r"\d+(?:\.\d+)?", text):
                    text = f"{text} years"
                return text, "field"
    description = " ".join(
        str(v)
        for v in (doc.get("description"), nested(doc, "job", "description"))
        if isinstance(v, str)
    )
    m = _YEARS_RE.search(description)
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
    for path in (
        "job.salary",
        "salary",
        "metadata.salary",
        "job.salary_range",
        "metadata.salary_range",
    ):
        for value in all_values(doc, path):
            if isinstance(value, dict):
                low = value.get("min") or value.get("from")
                high = value.get("max") or value.get("to")
                currency = value.get("currency") or ""
                if low or high:
                    span = f"{low}–{high}" if low and high else str(low or high)
                    return f"{currency} {span}".strip()
                continue
            text = first_text(value)
            if text:
                return text
    low = first_text(
        nested(doc, "job", "salary_min"),
        doc.get("salary_min"),
        nested(doc, "metadata", "salary_min"),
    )
    high = first_text(
        nested(doc, "job", "salary_max"),
        doc.get("salary_max"),
        nested(doc, "metadata", "salary_max"),
    )
    currency = (
        first_text(
            nested(doc, "job", "salary_currency"),
            nested(doc, "metadata", "salary_currency"),
        )
        or ""
    )
    if low or high:
        span = f"{low}–{high}" if low and high else str(low or high)
        return f"{currency} {span}".strip()
    return None


def company_name(doc: dict[str, Any]) -> str | None:
    return first_text(
        nested(doc, "company", "name"),
        nested(doc, "job", "company", "name"),
        nested(doc, "metadata", "company"),
        nested(doc, "metadata", "company_name"),
        nested(doc, "professional", "current_company"),
        nested(doc, "professional", "current_company", "name"),
        doc.get("company_name"),
    )


def location_parts(doc: dict[str, Any]) -> dict[str, str | None]:
    location = doc.get("location") if isinstance(doc.get("location"), dict) else {}
    country = (
        location.get("country") if isinstance(location.get("country"), dict) else {}
    )
    country_name = country.get("name") or country.get("ac_name") or country.get("code")
    if not country_name and isinstance(location.get("country"), str):
        country_name = location.get("country")
    return {
        "city": first_text(
            location.get("city"),
            location.get("current_location"),
            location.get("prime_city"),
        ),
        "country": first_text(country_name),
        "address": first_text(location.get("address")),
    }


def posted_date(doc: dict[str, Any]) -> str | None:
    for key in ("created_at", "start_datetime", "award_date"):
        value = doc.get(key)
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, str) and value.strip():
            return value.strip()[:10]
    start = nested(doc, "dates", "start")
    if isinstance(start, datetime):
        return start.date().isoformat()
    return None


def employment_type(doc: dict[str, Any]) -> str | None:
    return first_text(
        nested(doc, "job", "employment_type"),
        doc.get("employment_type"),
        nested(doc, "metadata", "employment_type"),
        nested(doc, "job", "job_type"),
        nested(doc, "metadata", "job_type"),
    )


def level_text(doc: dict[str, Any]) -> str | None:
    values = level_values(doc)
    return values[0] if values else None


def category_name(doc: dict[str, Any]) -> str | None:
    category = doc.get("category")
    if isinstance(category, dict):
        return first_text(category.get("name"))
    return first_text(category, nested(doc, "metadata", "category"))
