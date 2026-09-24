from __future__ import annotations

from rapidfuzz import fuzz

from .normalization import normalize, tokens


def _values(doc: dict, field: str) -> list[str]:
    value = doc.get(field, [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def score_document(
    doc: dict,
    query: str,
    mongo_score: float = 0.0,
) -> tuple[float, list[str]]:
    """Deterministic Phase 1 ranking.

    The score is not intended to be a statistical probability. It is a stable
    ordering function that prioritizes exact identity signals, phrase matches,
    aliases, taxonomy and then broad canonical ai_search_text relevance.
    """
    q = normalize(query)
    qt = tokens(query)

    title = normalize(str(doc.get("title") or ""))
    entity_name = normalize(str(doc.get("entity_name") or ""))
    category = normalize(str(doc.get("category") or ""))
    company = normalize(str(doc.get("company_name") or ""))
    user_name = normalize(str(doc.get("user_name") or ""))
    city = normalize(str(doc.get("city") or ""))
    country = normalize(str(doc.get("country") or ""))
    canonical = normalize(str(doc.get("ai_search_text") or ""))

    aliases = [normalize(v) for v in _values(doc, "aliases")]
    keywords = [normalize(v) for v in _values(doc, "ai_keywords")]

    score = min(float(mongo_score or 0.0), 10.0)
    matched: list[str] = []

    if q and q == title:
        score += 220
        matched.append("exact_title")
    elif q and q in title:
        score += 140
        matched.append("phrase_title")

    if q and q == entity_name:
        score += 120
        matched.append("exact_entity")

    if q and q in aliases:
        score += 115
        matched.append("exact_alias")
    elif any(q in alias for alias in aliases):
        score += 80
        matched.append("alias")

    if q and q in keywords:
        score += 100
        matched.append("exact_keyword")

    if q and q == category:
        score += 70
        matched.append("exact_category")
    elif q and q in category:
        score += 40
        matched.append("category")

    if q and q in canonical:
        score += 20
        matched.append("canonical_text")

    for token in qt:
        if token in title.split():
            score += 25
            if "title_keyword" not in matched:
                matched.append("title_keyword")
        elif any(token in alias.split() for alias in aliases):
            score += 20
            if "alias" not in matched:
                matched.append("alias")
        elif token in keywords:
            score += 18
            if "keyword" not in matched:
                matched.append("keyword")
        elif token in category:
            score += 12
            if "category" not in matched:
                matched.append("category")
        elif token in city or token in country:
            score += 10
            if "location" not in matched:
                matched.append("location")
        elif token in canonical:
            score += 2

    # If the caller did not already hard-filter location, matching location
    # tokens are still useful ranking signals.
    if city and any(token in city for token in qt):
        score += 35
        if "location" not in matched:
            matched.append("location")

    if country and any(token in country for token in qt):
        score += 35
        if "location" not in matched:
            matched.append("location")

    candidates = [title, entity_name, *aliases, *keywords, category, company, user_name]
    candidates = [candidate for candidate in candidates if candidate]

    if candidates and q:
        fuzzy = max(fuzz.token_set_ratio(q, candidate) for candidate in candidates)
        score += max(0.0, fuzzy - 70.0) * 0.35
        if fuzzy >= 82 and "fuzzy" not in matched:
            matched.append("fuzzy")

    return round(score, 4), matched
