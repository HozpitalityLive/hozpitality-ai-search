from __future__ import annotations

import re
from rapidfuzz import fuzz, process


def _singular(token: str) -> str:
    """Small, conservative plural normalizer used only for typo safety."""
    token = token.casefold()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and not token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _same_word_family(a: str, b: str) -> bool:
    return _singular(a) == _singular(b)


def correct_tokens(query_tokens, vocabulary, threshold=82, min_length=3):
    """Conservatively correct misspelled query tokens against search vocabulary."""
    vocab = sorted({
        str(v).casefold()
        for v in vocabulary
        if isinstance(v, str) and len(v) >= min_length
    })

    if not vocab:
        return list(query_tokens), []

    corrected = []
    changes = []

    for raw_token in query_tokens:
        token = str(raw_token).casefold()

        if len(token) < min_length or token in vocab:
            corrected.append(token)
            continue

        # Valid singular/plural variants must never be "corrected" into an
        # unrelated compound term such as chefs -> chefs-kitchen.
        family_matches = [candidate for candidate in vocab if _same_word_family(token, candidate)]
        if family_matches:
            corrected.append(token)
            continue

        match = process.extractOne(
            token,
            vocab,
            scorer=fuzz.WRatio,
        )

        if match:
            candidate, score, _ = match
            required = threshold + (8 if len(token) <= 4 else 0)

            # Require a strong match and a meaningful lexical shape match.
            # This avoids broad WRatio substitutions for already-plausible
            # hospitality terms.
            ratio = fuzz.ratio(token, candidate)
            if score >= required and ratio >= 78 and candidate != token:
                corrected.append(candidate)
                changes.append({
                    "from": token,
                    "to": candidate,
                    "score": round(score, 1),
                })
                continue

        corrected.append(token)

    return corrected, changes
