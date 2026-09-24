from __future__ import annotations

from rapidfuzz import fuzz

from .normalization import normalize, tokens


def _values(doc: dict, *fields: str) -> list[str]:
    result: list[str] = []
    for field in fields:
        value = doc.get(field, [])
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, list):
            result.extend(str(v) for v in value if v)
    return result


def _nested_name(doc: dict, *path: str) -> str:
    value = doc
    for key in path:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return str(value or "")


def score_document(doc: dict, query: str, mongo_score: float = 0.0) -> tuple[float, list[str]]:
    """Deterministic lexical ranking for Phase 1.

    Exact identity signals dominate MongoDB text relevance; aliases and
    keywords follow, then token coverage and fuzzy similarity. This is an
    ordering score, not a probability.
    """
    q = normalize(query)
    qt = tokens(query)
    title = normalize(str(doc.get("title") or doc.get("short_title") or ""))
    aliases = [normalize(v) for v in _values(doc, "search_aliases")]
    keywords = [normalize(v) for v in _values(doc, "search_keywords", "keywords")]
    category = normalize(_nested_name(doc, "category", "name"))
    company = normalize(_nested_name(doc, "company", "name"))
    user_name = normalize(_nested_name(doc, "user", "name"))
    canonical = normalize(str(doc.get("ai_search_text") or ""))

    score = min(float(mongo_score or 0.0), 25.0)
    matched: list[str] = []

    def add(points: float, label: str) -> None:
        nonlocal score
        score += points
        if label not in matched:
            matched.append(label)

    if q and q == title:
        add(300, "exact_title")
    elif q and q in title:
        add(180, "phrase_title")

    if q and q in aliases:
        add(170, "exact_alias")
    elif q and any(q in alias for alias in aliases):
        add(105, "alias")

    if q and q in keywords:
        add(150, "exact_keyword")
    elif q and any(q in keyword for keyword in keywords):
        add(90, "keyword")

    if q and q == category:
        add(85, "exact_category")
    elif q and q in category:
        add(45, "category")

    if q and q == company:
        add(75, "exact_company")
    elif q and q in company:
        add(40, "company")

    if q and q == user_name:
        add(75, "exact_person")

    if q and q in canonical:
        add(15, "canonical_text")

    # Reward coverage across identity fields. A result matching every query
    # token in its title/alias/keyword fields should outrank a single-token
    # broad description hit.
    identity = " ".join([title, *aliases, *keywords, category, company, user_name])
    covered = sum(1 for token in qt if token in identity)
    if qt:
        coverage = covered / len(qt)
        if coverage == 1:
            add(55, "all_tokens")
        elif coverage >= 0.5:
            add(25, "partial_tokens")

    # Location is a useful relevance signal, but explicit location filtering
    # is handled as a repository hard filter.
    location = " ".join([
        normalize(_nested_name(doc, "location", "city")),
        normalize(_nested_name(doc, "location", "country", "name")),
        normalize(_nested_name(doc, "location", "current_location")),
        normalize(_nested_name(doc, "location", "prime_city")),
    ])
    if any(token in location for token in qt):
        add(20, "location")

    candidates = [title, *aliases, *keywords, category, company, user_name]
    candidates = [candidate for candidate in candidates if candidate]
    if candidates and q:
        fuzzy = max(fuzz.token_set_ratio(q, candidate) for candidate in candidates)
        if fuzzy >= 92:
            add(30, "fuzzy")
        elif fuzzy >= 84:
            add(12, "fuzzy")

    return round(score, 4), matched
