from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "do", "for", "from",
    "find", "get", "give", "has", "have", "how", "i", "in", "is", "it", "list",
    "me", "my", "of", "on", "or", "please", "search", "show", "some", "tell",
    "the", "to", "what", "where", "which", "who", "with", "about", "any", "all",
    "latest", "recent", "new", "looking", "look", "need", "want", "would", "like",
    "named", "name", "called",
}

ENTITY_ALIASES = {
    "job": {"job", "jobs", "vacancy", "vacancies", "position", "positions", "career", "careers"},
    "professional": {"professional", "professionals", "candidate", "candidates", "person", "people"},
    "company": {"company", "companies", "employer", "employers", "hotel", "hotels"},
    "product": {"product", "products", "supplier", "suppliers"},
    "article": {"article", "articles", "story", "stories", "news"},
    "event": {"event", "events"},
    "award": {"award", "awards"},
    "faq": {"faq", "faqs", "question", "questions"},
}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9\s&.'-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens(value: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9][a-z0-9&.'-]*", normalize(value))
        if len(token) > 1 and token not in STOPWORDS
    ]


def canonical_entity(value: str | None) -> str | None:
    if not value:
        return None
    n = normalize(value)
    for entity, aliases in ENTITY_ALIASES.items():
        if n == entity or n in aliases:
            return entity
    return n


def normalized_location(value: str | None) -> str | None:
    if not value:
        return None
    return normalize(value)
