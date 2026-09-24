from __future__ import annotations

from collections import Counter
from typing import Iterable

from rapidfuzz import process, fuzz


def correct_tokens(
    query_tokens: list[str],
    vocabulary: Iterable[str],
    threshold: int = 82,
    min_length: int = 3,
) -> tuple[list[str], list[dict[str, object]]]:
    vocab = sorted({v for v in vocabulary if len(v) >= min_length})
    if not vocab:
        return query_tokens, []

    corrected: list[str] = []
    changes: list[dict[str, object]] = []

    for token in query_tokens:
        if len(token) < min_length:
            corrected.append(token)
            continue

        if token in vocab:
            corrected.append(token)
            continue

        match = process.extractOne(token, vocab, scorer=fuzz.WRatio)
        if not match:
            corrected.append(token)
            continue

        candidate, score, _ = match
        if score >= threshold and candidate != token:
            corrected.append(candidate)
            changes.append({"from": token, "to": candidate, "score": round(score, 1)})
        else:
            corrected.append(token)

    return corrected, changes


def build_vocabulary(documents: Iterable[dict]) -> list[str]:
    counter: Counter[str] = Counter()
    for doc in documents:
        for field in ("title", "keywords", "aliases", "category"):
            value = doc.get(field, [])
            if isinstance(value, str):
                values = [value]
            elif isinstance(value, list):
                values = value
            else:
                values = [value] if value else []
            for item in values:
                if isinstance(item, str):
                    for token in item.casefold().split():
                        token = "".join(ch for ch in token if ch.isalnum() or ch in "-&")
                        if len(token) >= 3:
                            counter[token] += 1
    return [word for word, _ in counter.most_common(10000)]
