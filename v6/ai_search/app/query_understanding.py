from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .normalization import canonical_entity, normalize, tokens


@dataclass
class SearchPlan:
    intent: str = "search"
    entity: str | None = None
    keywords: list[str] = field(default_factory=list)
    city: str | None = None
    country: str | None = None
    experience: int | None = None
    level: str | None = None
    department: str | None = None
    industry: str | None = None
    category: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    clarification: str | None = None
    clarification_options: list[str] = field(default_factory=list)
    original_query: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "entity": self.entity,
            "keywords": self.keywords,
            "city": self.city,
            "country": self.country,
            "experience": self.experience,
            "level": self.level,
            "department": self.department,
            "industry": self.industry,
            "category": self.category,
            "date": {
                "from": self.date_from.isoformat() if self.date_from else None,
                "to": self.date_to.isoformat() if self.date_to else None,
            },
            "filters": self.filters,
            "confidence": round(self.confidence, 3),
            "clarification": self.clarification,
            "clarification_options": self.clarification_options,
        }


ENTITY_PATTERNS = {
    "job": r"\b(jobs?|vacanc(?:y|ies)|positions?|careers?|openings?|employment)\b",
    "professional": r"\b(professionals?|candidates?|people|persons?|experts?|talent)\b",
    "company": r"\b(companies|company|employers?|hotel groups?|businesses?)\b",
    "product": r"\b(products?|suppliers?|marketplace|vendors?)\b",
    "article": r"\b(articles?|stories?|news|blogs?|blog posts?)\b",
    "event": r"\b(events?|conferences?|exhibitions?|summits?)\b",
    "award": r"\b(awards?|honou?rs?|recognition)\b",
    "faq": r"\b(faqs?|questions?|frequently asked)\b",
}

ENTITY_WORDS = {
    entity: set(re.findall(r"[a-z]+", pattern))
    for entity, pattern in ENTITY_PATTERNS.items()
}

LEVELS = {
    "senior": ["senior", "sr", "lead", "principal"],
    "junior": ["junior", "jr", "entry level", "entry-level"],
    "mid": ["mid level", "mid-level", "midlevel", "associate"],
    "manager": ["manager", "management", "head"],
    "executive": ["executive", "director", "vp", "vice president", "c-suite"],
    "intern": ["intern", "internship", "trainee", "graduate"],
}

DEPARTMENTS = [
    "food and beverage", "f&b", "culinary", "kitchen", "front office",
    "housekeeping", "human resources", "hr", "finance", "accounting",
    "sales", "marketing", "digital marketing", "revenue management",
    "revenue", "engineering", "maintenance", "security", "procurement",
    "operations", "general management", "information technology", "it",
]

INDUSTRIES = [
    "hotels and resorts", "hotels & resorts", "hospitality", "restaurant",
    "restaurants", "travel", "tourism", "catering", "airline", "cruise",
    "spa", "wellness", "food and beverage", "f&b",
]

COUNTRIES = {
    "uae": "United Arab Emirates", "u.a.e": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
    "dubai": "United Arab Emirates", "usa": "United States",
    "us": "United States", "united states": "United States",
    "uk": "United Kingdom", "united kingdom": "United Kingdom",
    "india": "India", "canada": "Canada", "australia": "Australia",
    "singapore": "Singapore", "saudi arabia": "Saudi Arabia",
    "qatar": "Qatar", "bahrain": "Bahrain", "oman": "Oman",
    "kuwait": "Kuwait", "maldives": "Maldives", "germany": "Germany",
    "france": "France", "spain": "Spain", "italy": "Italy",
    "netherlands": "Netherlands", "switzerland": "Switzerland",
    "japan": "Japan", "china": "China", "south africa": "South Africa",
}

CITIES = {
    "dubai": "Dubai", "abu dhabi": "Abu Dhabi", "sharjah": "Sharjah",
    "ajman": "Ajman", "ras al khaimah": "Ras Al Khaimah",
    "mumbai": "Mumbai", "bombay": "Mumbai", "delhi": "Delhi",
    "new delhi": "New Delhi", "gurugram": "Gurugram", "gurgaon": "Gurugram",
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "hyderabad": "Hyderabad",
    "chennai": "Chennai", "kolkata": "Kolkata", "pune": "Pune",
    "goa": "Goa", "jaipur": "Jaipur", "lucknow": "Lucknow",
    "singapore": "Singapore", "london": "London", "new york": "New York",
    "los angeles": "Los Angeles", "toronto": "Toronto", "melbourne": "Melbourne",
    "sydney": "Sydney", "riyadh": "Riyadh", "doha": "Doha",
    "muscat": "Muscat", "manama": "Manama", "kuwait city": "Kuwait City",
    "paris": "Paris", "berlin": "Berlin", "amsterdam": "Amsterdam",
    "tokyo": "Tokyo", "hong kong": "Hong Kong",
}

CATEGORY_MARKERS = r"\b(?:category|type|section|topic)\s+(?:is\s+)?([a-z][a-z0-9 &/'-]{2,60})"


def _phrase_in(text: str, phrases: list[str]) -> str | None:
    low = text.casefold()
    for phrase in sorted(phrases, key=len, reverse=True):
        value = phrase.casefold()
        if len(value) <= 3:
            if re.search(rf"(?<!\w){re.escape(value)}(?!\w)", low):
                return phrase
        elif value in low:
            return phrase
    return None


def _entity(text: str) -> str | None:
    low = text.casefold()
    found: list[tuple[int, str]] = []
    for entity, pattern in ENTITY_PATTERNS.items():
        m = re.search(pattern, low)
        if m:
            found.append((m.start(), entity))
    if not found:
        return None
    # First explicit module noun wins; job/professional role phrases are handled below.
    return sorted(found)[0][1]


def _experience(text: str) -> int | None:
    patterns = [
        r"\b(?:minimum|min\.?|at least|over|more than)\s+(\d{1,2})\s*(?:\+|plus)?\s*(?:years?|yrs?)\b",
        r"\b(\d{1,2})\s*(?:\+|plus)?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)\b",
        r"\b(?:experience|exp)\s*(?:of|:)?\s*(\d{1,2})\s*(?:\+|plus)?\s*(?:years?|yrs?)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text.casefold())
        if m:
            return int(m.group(1))
    return None


def _date_range(text: str) -> tuple[datetime | None, datetime | None]:
    now = datetime.now(timezone.utc)
    low = text.casefold()
    start = end = None
    if re.search(r"\btoday\b", low):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif re.search(r"\btomorrow\b", low):
        start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif re.search(r"\bthis week\b", low):
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
    elif re.search(r"\bthis month\b", low):
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month = start.month % 12 + 1
        year = start.year + (1 if start.month == 12 else 0)
        end = start.replace(year=year, month=month)
    elif re.search(r"\bthis year\b|\bthis calendar year\b", low):
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(year=start.year + 1)
    elif re.search(r"\b(latest|recent|newest|new)\b", low):
        # "latest" is represented as a sort preference, not an arbitrary date filter.
        return None, None
    return start, end


def _extract_filters(text: str, entity: str | None) -> dict[str, Any]:
    low = text.casefold()
    filters: dict[str, Any] = {}

    m = re.search(r"\b(?:salary|pay|paying|compensation)\s*(?:of|at|from|over|above|>=|:)?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(aed|usd|inr|gbp|eur|sar|qar|a?ed)?\b", low)
    if m:
        filters["salary_min"] = float(m.group(1).replace(",", ""))
        if m.group(2):
            filters["salary_currency"] = m.group(2).upper()

    employment = _phrase_in(low, ["full time", "full-time", "part time", "part-time", "contract", "temporary", "remote"])
    if employment:
        filters["employment_type"] = employment

    if re.search(r"\bverified\b", low):
        filters["verified"] = True
    if re.search(r"\bfeatured\b", low):
        filters["featured"] = True
    if re.search(r"\bcurrently working\b|\bworking professionals?\b", low):
        filters["currently_working"] = True

    return filters


def understand(query: str) -> SearchPlan:
    original = " ".join(query.strip().split())
    low = original.casefold()
    plan = SearchPlan(original_query=original)

    entity = _entity(original)

    # A role such as "chef" is naturally a job/professional intent even without
    # the word "job". Explicit module nouns always take precedence.
    if not entity:
        if re.search(r"\b(?:hire|hiring|vacancy|vacancies|salary|paying)\b", low):
            entity = "job"
        elif re.search(r"\b(?:chef|manager|director|engineer|recruiter|developer|waiter|bartender|housekeeper|receptionist)\b", low):
            entity = "professional"

    plan.entity = entity
    plan.experience = _experience(original)

    for level, values in LEVELS.items():
        if any(re.search(rf"\b{re.escape(v)}\b", low) if " " not in v else v in low for v in values):
            plan.level = level
            break

    plan.department = _phrase_in(low, DEPARTMENTS)
    plan.industry = _phrase_in(low, INDUSTRIES)

    # Explicit country first. "Dubai" is both a city and a UAE location.
    for alias, canonical in sorted(COUNTRIES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", low):
            if alias == "dubai":
                continue
            plan.country = canonical
            break

    for alias, canonical in sorted(CITIES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", low):
            plan.city = canonical
            if canonical == "Dubai":
                plan.country = "United Arab Emirates"
            break

    # "in <place>" catches configured cities/countries while avoiding
    # accidental extraction of job-role phrases.
    m = re.search(r"\bin\s+([A-Za-z][A-Za-z .'-]{1,50})(?=$|,|\s+with\b|\s+for\b|\s+and\b)", original, re.I)
    if m and not plan.city and not plan.country:
        candidate = normalize(m.group(1))
        if candidate in CITIES:
            plan.city = CITIES[candidate]
        elif candidate in COUNTRIES:
            plan.country = COUNTRIES[candidate]

    cm = re.search(CATEGORY_MARKERS, original, re.I)
    if cm:
        plan.category = cm.group(1).strip(" .,-")

    plan.filters = _extract_filters(original, entity)
    plan.date_from, plan.date_to = _date_range(original)

    # Remove control words and extracted filter phrases to form search keywords.
    keyword_text = original
    removals = list(ENTITY_WORDS.get(entity, set())) if entity else []
    if entity and entity in ENTITY_PATTERNS:
        keyword_text = re.sub(ENTITY_PATTERNS[entity], " ", keyword_text, flags=re.I)
    removals += ["senior", "junior", "mid", "level", "minimum", "years", "year", "experience", "exp"]
    removals += [v for v in LEVELS.get(plan.level, [])] if plan.level else []
    removals += DEPARTMENTS + INDUSTRIES
    removals += list(CITIES.keys()) + list(COUNTRIES.keys())
    removals += ["find", "search", "show", "list", "get", "give", "me", "please", "need", "want",
                 "looking", "for", "in", "with", "of", "the", "a", "an", "and", "from", "at",
                 "minimum", "latest", "recent", "new", "today", "tomorrow", "this", "week", "month", "year",
                 "remote", "full", "time", "part", "contract", "temporary", "verified", "featured"]

    for phrase in sorted(set(removals), key=len, reverse=True):
        if phrase:
            keyword_text = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", keyword_text, flags=re.I)

    # Remove salary/experience numbers and punctuation.
    keyword_text = re.sub(r"\b\d+(?:\.\d+)?\b", " ", keyword_text)
    plan.keywords = tokens(keyword_text)[:12]

    confidence = 0.2
    if entity: confidence += 0.25
    if plan.keywords: confidence += 0.2
    if plan.city or plan.country: confidence += 0.15
    if plan.experience is not None or plan.level or plan.department or plan.industry: confidence += 0.15
    if plan.category: confidence += 0.05
    plan.confidence = min(confidence, 0.98)

    return plan


def clarification_for(plan: SearchPlan) -> str | None:
    if plan.intent != "search":
        return None

    if plan.entity == "job":
        if not plan.keywords:
            return "What type of job are you looking for?"
        if not plan.city and not plan.country:
            return "Which location would you prefer?"
    elif plan.entity == "professional":
        if not plan.keywords:
            return "What type of professional or skill are you looking for?"
        if not plan.city and not plan.country:
            return "Which location would you prefer?"
    elif plan.entity == "company":
        if not plan.keywords:
            return "What type of company or hospitality business are you looking for?"
        if not plan.city and not plan.country:
            return "Which location would you prefer?"
    elif plan.entity == "product":
        if not plan.keywords:
            return "What type of product or supplier are you looking for?"
        if not plan.city and not plan.country:
            return "Which location would you prefer?"
    elif plan.entity == "article":
        if not plan.keywords and not plan.category:
            return "What topic or category would you like me to search for in the articles?"
    elif plan.entity == "event":
        if not plan.keywords and not plan.category:
            return "What type of event are you looking for?"
        if not plan.city and not plan.country:
            return "Which location would you prefer?"
    elif plan.entity == "award":
        if not plan.keywords and not plan.category:
            return "What type of award are you looking for?"
        if not plan.city and not plan.country:
            return "Which location would you prefer?"
    elif plan.entity == "faq":
        if not plan.keywords and not plan.category:
            return "What topic or question should I search for?"

    return None
