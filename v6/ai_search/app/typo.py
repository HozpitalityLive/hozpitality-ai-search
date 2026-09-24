from __future__ import annotations

from rapidfuzz import fuzz, process


def correct_tokens(query_tokens, vocabulary, threshold=82, min_length=3):
    """Correct misspelled query tokens against the search vocabulary."""
    vocab = sorted({
        str(v).casefold()
        for v in vocabulary
        if isinstance(v, str) and len(v) >= min_length
    })

    if not vocab:
        return query_tokens, []

    corrected = []
    changes = []

    for raw_token in query_tokens:
        token = raw_token.casefold()

        if len(token) < min_length or token in vocab:
            corrected.append(token)
            continue

        match = process.extractOne(
            token,
            vocab,
            scorer=fuzz.WRatio,
        )

        if match:
            candidate, score, _ = match

            # Short tokens are more vulnerable to false corrections.
            required = threshold + (6 if len(token) <= 4 else 0)

            if score >= required and candidate != token:
                corrected.append(candidate)
                changes.append({
                    "from": token,
                    "to": candidate,
                    "score": round(score, 1),
                })
                continue

        corrected.append(token)

    return corrected, changes
