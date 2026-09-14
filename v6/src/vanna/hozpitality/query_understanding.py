"""Fast, deterministic query understanding for Hozpitality V6.

Uses:
- spaCy EntityRuler for entity/domain-intent extraction (no model download).
- SymSpell for data-driven spelling correction.
- No LLM call is made for query understanding.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

try:
    import spacy
    from spacy.pipeline import EntityRuler
except Exception:  # optional dependency guard
    spacy = None
    EntityRuler = None

try:
    from symspellpy import SymSpell, Verbosity
except Exception:  # optional dependency guard
    SymSpell = None
    Verbosity = None


ENTITY_ALIASES = {
    "job": ["job", "jobs", "vacancy", "vacancies", "career", "careers",
            "employment", "opening", "openings", "position", "positions"],
    "professional": ["professional", "professionals", "candidate", "candidates",
                     "profile", "profiles", "resume", "resumes", "cv"],
    "company": ["company", "companies", "employer", "employers", "organization",
                "organisations", "organisation"],
    "article": ["article", "articles", "news", "blog", "blogs", "story", "stories"],
    "event": ["event", "events", "conference", "conferences", "expo",
              "exhibition", "exhibitions"],
    "product": ["product", "products", "marketplace", "supplier", "suppliers"],
    "faq": ["faq", "faqs", "question", "questions", "help"],
    "award": ["award", "awards", "recognition", "winner", "winners",
              "nomination", "nominations"],
}

# Only these are generic stop words. Domain vocabulary is NOT hardcoded here.
STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "do", "for", "from",
    "find", "get", "give", "has", "have", "how", "i", "in", "is", "it", "list",
    "me", "my", "of", "on", "or", "please", "search", "show", "some", "tell",
    "the", "to", "what", "where", "which", "who", "with", "about", "any", "all",
    "latest", "recent", "new", "looking", "look", "need", "want", "would",
    "like", "named", "name", "called",
}


class QueryUnderstanding:
    """spaCy EntityRuler + SymSpell query normalizer."""

    def __init__(self, dictionary_path: str | None = None):
        self.dictionary_path = Path(
            dictionary_path
            or os.getenv(
                "SYMSPELL_DICTIONARY",
                os.getenv(
                    "SYMSpell_DICTIONARY",
                    str(
                        Path(__file__).resolve().parents[3]
                        / "data"
                        / "symspell_dictionary.txt"
                    ),
                ),
            )
        )
        self.max_edit_distance = int(
            os.getenv(
                "SYMSPELL_MAX_EDIT_DISTANCE",
                os.getenv("SYMSpell_MAX_EDIT_DISTANCE", "2"),
            )
        )
        self._nlp = self._build_nlp()
        self._symspell = self._build_symspell()

    @staticmethod
    def _build_nlp():
        if spacy is None:
            return None
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
        patterns = []
        for entity, aliases in ENTITY_ALIASES.items():
            for alias in aliases:
                patterns.append(
                    {
                        "label": f"ENTITY_{entity.upper()}",
                        "pattern": [{"LOWER": token} for token in alias.split()],
                    }
                )
        ruler.add_patterns(patterns)
        return nlp

    def _build_symspell(self):
        if SymSpell is None or not self.dictionary_path.exists():
            return None
        try:
            sym = SymSpell(
                max_dictionary_edit_distance=int(os.getenv("SYMSPELL_MAX_EDIT_DISTANCE", os.getenv("SYMSpell_MAX_EDIT_DISTANCE", "2"))),
                prefix_length=7,
            )
            # term_index=0, count_index=1 matches:
            # <word><TAB><frequency>
            loaded = sym.load_dictionary(str(self.dictionary_path), 0, 1)
            if not loaded:
                return None
            return sym
        except Exception:
            return None

    @staticmethod
    def tokens(text: str) -> list[str]:
        return [
            t for t in re.findall(r"[\w][\w'&.-]*", (text or "").lower())
            if len(t) > 1 and t not in STOPWORDS
        ]

    def extract_entity(self, text: str) -> tuple[str | None, list[str]]:
        """Return canonical entity type and exact entity tokens/spans."""
        if self._nlp is None:
            return None, []
        doc = self._nlp(text or "")
        for ent in doc.ents:
            if ent.label_.startswith("ENTITY_"):
                return ent.label_[7:].lower(), [t.text.lower() for t in ent]
        return None, []

    def correct_token(self, token: str) -> str:
        """Correct only when SymSpell has a sufficiently close dictionary term."""
        if not self._symspell or len(token) < 3 or not token.isalpha():
            return token
        try:
            suggestions = self._symspell.lookup(
                token,
                Verbosity.CLOSEST,
                max_edit_distance=int(os.getenv("SYMSPELL_MAX_EDIT_DISTANCE", os.getenv("SYMSpell_MAX_EDIT_DISTANCE", "2"))),
                include_unknown=False,
            )
            if not suggestions:
                return token
            best = suggestions[0]
            # Never silently replace a token with a much less frequent/ambiguous
            # candidate. SymSpell already ranks by edit distance then frequency.
            return best.term.lower()
        except Exception:
            return token

    def normalize_tokens(self, tokens: Iterable[str]) -> list[str]:
        return [self.correct_token(t) for t in tokens]

    def normalize_query(self, text: str) -> tuple[str, list[str], str | None]:
        raw = re.sub(r"\s+", " ", (text or "").strip().lower())
        entity, entity_tokens = self.extract_entity(raw)
        tokens = self.tokens(raw)
        entity_token_set = set(entity_tokens)
        semantic = [t for t in tokens if t not in entity_token_set]
        corrected = self.normalize_tokens(semantic)
        return " ".join(corrected), corrected, entity
