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


def score_document(doc: dict, query: str, mongo_score: float = 0.0) -> tuple[float, list[str]]:
    q = normalize(query)
    qt = tokens(query)
    title = normalize(str(doc.get("title") or ""))
    aliases = [normalize(v) for v in _values(doc, "aliases")]
    keywords = [normalize(v) for v in _values(doc, "keywords")]
    category = normalize(str(doc.get("category") or ""))
    city = normalize(str((doc.get("location") or {}).get("city") or ""))
    country = normalize(str((doc.get("location") or {}).get("country") or ""))
    description = normalize(str(doc.get("description") or ""))

    score = min(float(mongo_score), 5.0)
    matched: list[str] = []

    if q == title:
        score += 100
        matched.append("exact_title")
    elif q in title:
        score += 70
        matched.append("phrase_title")

    if q in aliases:
        score += 65
        matched.append("exact_alias")

    if q in keywords:
        score += 55
        matched.append("exact_keyword")

    if q in category:
        score += 35
        matched.append("category")

    if city and city in q:
        score += 20
        matched.append("location")
    if country and country in q:
        score += 20
        matched.append("location")

    for token in qt:
        if token in title.split():
            score += 15
            if "keyword" not in matched:
                matched.append("title_keyword")
        elif any(token in alias.split() for alias in aliases):
            score += 12
            if "alias" not in matched:
                matched.append("alias")
        elif token in keywords:
            score += 10
            if "keyword" not in matched:
                matched.append("keyword")
        elif token in category:
            score += 6
        elif token in description:
            score += 2

    # Fuzzy similarity is used only for ranking, not as a substitute for
    # explicit hard filters.
    candidates = [title, *aliases, *keywords]
    if candidates:
        fuzzy = max(fuzz.token_set_ratio(q, c) for c in candidates if c)
        score += max(0.0, fuzzy - 70.0) * 0.25
        if fuzzy >= 82 and "fuzzy" not in matched:
            matched.append("fuzzy")

    return round(score, 4), matched
