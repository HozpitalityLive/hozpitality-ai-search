from __future__ import annotations

import re
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any

from . import schema_map
from .evidence import all_values, satisfies_strict
from .llm import OllamaQueryInterpreter
from .normalization import canonical_entity, tokens
from .observability import timed
from .query_understanding import (
    CITY_COUNTRY,
    LEVELS,
    SearchPlan,
    clarification_for,
    understand,
)
from .ranking import score_document
from .repository import HARD_CONCEPT_KEYS, SearchDocumentsRepository
from .repository import _parse_date as parse_date
from .results import result_payload
from .typo import correct_tokens
from .vector import SemanticVectorIndex

ENTITY_TYPES = {
    "job",
    "professional",
    "company",
    "product",
    "article",
    "event",
    "award",
    "faq",
}

NO_EXACT_MESSAGE = (
    "I couldn't find an exact match for your requested filters. "
    "Here are some related results you can check."
)


def result_key(entity_type: Any, entity_id: Any) -> str:
    return f"{entity_type}:{entity_id}"


class SearchService:
    MAX_RESULTS = 5
    VOCAB_CACHE_MAX = 5000
    VOCAB_CACHE_TTL_SECONDS = 600.0

    def __init__(
        self,
        repository: SearchDocumentsRepository,
        fuzzy_threshold: int = 82,
        llm: OllamaQueryInterpreter | None = None,
        semantic: SemanticVectorIndex | None = None,
    ):
        self.repository = repository
        self.fuzzy_threshold = fuzzy_threshold
        # Bounded, per-token vocabulary cache. The previous implementation
        # accumulated every suggestion from every request into one ever-growing
        # list, so corrections depended on unrelated traffic and memory grew
        # without limit.
        self._token_vocab: OrderedDict[str, tuple[float, list[str]]] = OrderedDict()
        self._full_vocabulary: list[str] | None = None
        self.llm = llm or OllamaQueryInterpreter()
        self.semantic = semantic or SemanticVectorIndex()

    # ------------------------------------------------------------------
    # Vocabulary for typo correction
    # ------------------------------------------------------------------

    def _get_vocabulary(self, query_tokens: list[str] | None = None) -> list[str]:
        suggest = getattr(self.repository, "suggest_vocabulary", None)
        if callable(suggest):
            values: set[str] = set()
            now = time.monotonic()
            for token in query_tokens or []:
                cached = self._token_vocab.get(token)
                if cached is None or now - cached[0] > self.VOCAB_CACHE_TTL_SECONDS:
                    try:
                        terms = list(suggest(token))
                    except Exception:
                        terms = []
                    self._token_vocab[token] = (now, terms)
                    while len(self._token_vocab) > self.VOCAB_CACHE_MAX:
                        self._token_vocab.popitem(last=False)
                else:
                    self._token_vocab.move_to_end(token)
                    terms = cached[1]
                values.update(terms)
            return sorted(values)

        # Repositories without per-token suggestions expose a bounded
        # vocabulary; load it once.
        full = getattr(self.repository, "vocabulary", None)
        if callable(full):
            if self._full_vocabulary is None:
                try:
                    self._full_vocabulary = sorted({str(v) for v in full()})
                except Exception:
                    self._full_vocabulary = []
            return self._full_vocabulary
        return []

    def invalidate_vocabulary(self) -> None:
        self._token_vocab.clear()
        self._full_vocabulary = None

    # ------------------------------------------------------------------
    # Result payload (normalized contract: results.py)
    # ------------------------------------------------------------------

    @staticmethod
    def doc_entity_id(doc: dict) -> str:
        return str((doc.get("source") or {}).get("object_id") or doc.get("_id") or "")

    @classmethod
    def doc_keys(cls, doc: dict) -> set[str]:
        entity_id = cls.doc_entity_id(doc)
        return {
            entity_id,
            result_key(doc.get("entity_type") or "", entity_id),
            str(doc.get("_id") or ""),
        } - {""}

    @staticmethod
    def _result_payload(
        doc: dict, score: float, matched: list[str], corrected_query: str | None = None
    ) -> dict:
        return result_payload(doc, score, matched, corrected_query)

    # ------------------------------------------------------------------
    # Hard filters
    # ------------------------------------------------------------------

    def _hard_filter_docs(
        self,
        docs: list[dict],
        plan: SearchPlan,
        status: str | None,
        is_live: bool | None,
        structured: dict,
    ) -> list[dict]:
        """
        Final authoritative filter applied after every retrieval path.

        This is intentionally performed after lexical, fallback and semantic
        retrieval so no candidate can bypass explicit user constraints.
        """
        return [
            doc
            for doc in docs
            if self._document_matches_filters(
                doc,
                entity=plan.entity,
                city=plan.city,
                country=plan.country,
                status=status,
                is_live=is_live,
                structured=structured,
                date_from=plan.date_from,
                date_to=plan.date_to,
            )
        ]

    @classmethod
    def _document_matches_filters(
        cls,
        doc: dict,
        *,
        entity: str | None,
        city: str | None,
        country: str | None,
        status: str | None,
        is_live: bool | None,
        structured: dict | None = None,
        date_from=None,
        date_to=None,
    ) -> bool:
        """Delegate the authoritative document-level filter to the repository."""
        return SearchDocumentsRepository._document_matches_filters(
            doc,
            entity=entity,
            city=city,
            country=country,
            status=status,
            is_live=is_live,
            structured=structured,
            date_from=date_from,
            date_to=date_to,
        )

    # ------------------------------------------------------------------
    # Phase 2: understanding
    # ------------------------------------------------------------------

    def plan_query(
        self,
        original: str,
        *,
        entity: str | None = None,
        city: str | None = None,
        country: str | None = None,
    ) -> SearchPlan:
        """Deterministic understanding first; Ollama only for ambiguous text."""
        plan = understand(original)
        deterministic_plan = understand(original)
        if entity:
            plan.entity = canonical_entity(entity)
        if city:
            plan.city = city
        if country:
            plan.country = country

        # Ollama is a fallback, not the default path. Simple/clear searches
        # never need an LLM request.
        if plan.confidence < 0.65 or (
            plan.entity and not plan.keywords and not plan.category
        ):
            plan = self.llm.interpret(original, plan)

        # Deterministic extraction is authoritative, and explicit API/chat
        # context is authoritative over the current query. This prevents a
        # role word such as "chef" from silently switching a job search into a
        # professional search.
        if entity:
            plan.entity = canonical_entity(entity)
            plan.entity_source = (
                deterministic_plan.entity_source
                if deterministic_plan.entity == plan.entity
                else "api"
            )
        elif deterministic_plan.entity:
            plan.entity = deterministic_plan.entity
            plan.entity_source = deterministic_plan.entity_source
        elif plan.entity:
            # Entity supplied only by the LLM fallback: usable for retrieval
            # but never enough to demand clarification.
            plan.entity_source = "llm"

        if city:
            plan.city = city
        elif deterministic_plan.city:
            plan.city = deterministic_plan.city

        if country:
            plan.country = country
        elif deterministic_plan.country:
            plan.country = deterministic_plan.country

        if entity or city or country:
            plan.explicit_location = bool(plan.explicit_location or city or country)
        if deterministic_plan.experience is not None:
            plan.experience = deterministic_plan.experience
        if deterministic_plan.level:
            plan.level = deterministic_plan.level
        if deterministic_plan.department:
            plan.department = deterministic_plan.department
        if deterministic_plan.industry:
            plan.industry = deterministic_plan.industry
        if deterministic_plan.category:
            plan.category = deterministic_plan.category
        plan.filters = dict(deterministic_plan.filters)
        plan.date_from, plan.date_to = (
            deterministic_plan.date_from,
            deterministic_plan.date_to,
        )
        return plan

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        *,
        query: str,
        entity: str | None = None,
        city: str | None = None,
        country: str | None = None,
        status: str | None = None,
        is_live: bool | None = None,
        limit: int = 5,
        structured_filters: dict | None = None,
        exclude_ids: list[str] | None = None,
    ) -> dict:
        original = " ".join(query.strip().split())
        if not original:
            return {
                "query": original,
                "corrected_query": None,
                "total": 0,
                "results": [],
            }

        plan = self.plan_query(original, entity=entity, city=city, country=country)
        return self.execute_plan(
            plan,
            original=original,
            status=status,
            is_live=is_live,
            limit=limit,
            structured_filters=structured_filters,
            exclude_ids=exclude_ids,
            clarify=True,
        )

    # ------------------------------------------------------------------
    # Phase 1 retrieval + ranking for an already-understood plan
    # ------------------------------------------------------------------

    def execute_plan(
        self,
        plan: SearchPlan,
        *,
        original: str = "",
        status: str | None = None,
        is_live: bool | None = None,
        limit: int = 5,
        structured_filters: dict | None = None,
        exclude_ids: list[str] | None = None,
        clarify: bool = True,
    ) -> dict:
        with timed("search_ms"):
            return self._execute_plan(
                plan,
                original=original,
                status=status,
                is_live=is_live,
                limit=limit,
                structured_filters=structured_filters,
                exclude_ids=exclude_ids,
                clarify=clarify,
            )

    def _execute_plan(
        self,
        plan: SearchPlan,
        *,
        original: str,
        status: str | None,
        is_live: bool | None,
        limit: int,
        structured_filters: dict | None,
        exclude_ids: list[str] | None,
        clarify: bool,
    ) -> dict:
        original = original or " ".join(plan.keywords)
        clarification = clarification_for(plan) if clarify else None
        understanding = plan.as_dict()
        understanding["clarification"] = clarification

        empty = {
            "query": original,
            "corrected_query": None,
            "total": 0,
            "results": [],
            "message": None,
            "related_results": [],
            "understanding": understanding,
        }
        if clarification:
            return empty

        notes: list[str] = []
        if plan.intent == "facet" and plan.facet:
            return self._facet(plan, understanding)
        if plan.related == "companies_hiring":
            return self._companies_hiring(
                plan, understanding, status=status, is_live=is_live, limit=limit
            )
        if plan.entity == "faq" and (
            plan.city or plan.country or plan.explicit_location
        ):
            # FAQ records carry no location (migrate_faqs.py).
            plan.city = plan.country = plan.location_text = None
            plan.explicit_location = False
            notes.append("FAQs have no location; location ignored")
        elif (
            plan.city
            and plan.entity
            and not schema_map.location_paths(plan.entity, "city")
        ):
            # Articles store countries only: use the city's country.
            country = CITY_COUNTRY.get(plan.city)
            notes.append(
                f"{plan.entity} records have no city field; searched by country {country}"
                if country
                else f"{plan.entity} records have no city field; city ignored"
            )
            plan.city, plan.country = None, country or plan.country

        # Search terms: user keywords, else the extracted topic (department /
        # industry / category, e.g. "restaurant suppliers"), else - when the
        # request is purely structural ("events in Dubai") - browse by filters.
        topic_tokens: list[str] = []
        for value in (plan.department, plan.industry, plan.category):
            if value:
                topic_tokens.extend(tokens(str(value)))
        search_tokens = list(dict.fromkeys(plan.keywords))
        if plan.location_text and not plan.city and not plan.country:
            # An unknown place ("Antarctica") is a location constraint, not a
            # topic: it must not be required in related-result retrieval.
            place_tokens = set(tokens(plan.location_text))
            search_tokens = [t for t in search_tokens if t not in place_tokens]
        browse_mode = False
        if not search_tokens:
            if topic_tokens:
                search_tokens = list(dict.fromkeys(topic_tokens))
            elif plan.entity or plan.city or plan.country:
                browse_mode = True
            else:
                search_tokens = tokens(original)
        if not search_tokens and not browse_mode:
            return empty

        changes: list[dict] = []
        corrected_tokens = list(search_tokens)
        if search_tokens:
            # Typo correction applies to the actual search terms, not control
            # words such as "jobs", "in", "Dubai", etc.
            vocabulary = self._get_vocabulary(search_tokens)
            corrected_tokens, changes = correct_tokens(
                search_tokens,
                vocabulary,
                threshold=self.fuzzy_threshold,
            )

            # A typo-corrected level is a structured signal, not just a
            # keyword. Example: "excutive chef" -> level="executive".
            level_values = {
                level: {level, *(v.casefold() for v in values if " " not in v)}
                for level, values in LEVELS.items()
            }
            if not plan.level:
                for token in corrected_tokens:
                    token_low = token.casefold()
                    for level, values in level_values.items():
                        if token_low in values:
                            plan.level = level
                            break
                    if plan.level:
                        break

            # Keep explicit level + role queries in natural order for retrieval.
            # "excutive chef" should become "executive chef", not "chef executive".
            if plan.level and plan.level in {t.casefold() for t in corrected_tokens}:
                corrected_tokens = [
                    plan.level,
                    *[t for t in corrected_tokens if t.casefold() != plan.level],
                ]
        corrected_query = " ".join(corrected_tokens).strip()
        effective_query = corrected_query or " ".join(search_tokens)

        # Department/industry/category add context when they are explicitly
        # present in indexed text. Level/experience stay ranking signals.
        retrieval_terms = list(corrected_tokens if changes else search_tokens)
        for value in (plan.department, plan.industry, plan.category):
            if value:
                retrieval_terms.extend(tokens(str(value)))
        retrieval_terms = list(dict.fromkeys(retrieval_terms))
        retrieval_query = " ".join(retrieval_terms).strip() or effective_query

        canonical_entity_name = canonical_entity(plan.entity)
        if canonical_entity_name in ENTITY_TYPES:
            plan.entity = canonical_entity_name

        # Explicit filter expressions (salary, employment type, accommodation,
        # ...) are hard constraints; semantic concepts (level, department,
        # industry, experience, category) remain ranking signals unless the
        # user explicitly required them (plan.strict_filters).
        structured = dict(plan.filters)
        if structured_filters:
            structured.update(structured_filters)
            plan.filters = dict(structured)

        strict = set(plan.strict_filters)
        if structured.get("accommodation") is True:
            strict.add("accommodation")
        evidence_filters = {**structured, "level": plan.level}

        limit = min(max(int(limit), 1), self.MAX_RESULTS)
        excluded = {str(value) for value in (exclude_ids or []) if str(value).strip()}

        def is_excluded(doc: dict) -> bool:
            return bool(excluded and self.doc_keys(doc) & excluded)

        # An explicit but unrecognized location is still a hard location
        # constraint. We must never silently broaden it to a global search.
        # Example: "chef jobs in Antarctica" has zero exact results unless an
        # indexed document actually carries Antarctica as its location.
        unknown_explicit_location = (
            plan.explicit_location and not plan.city and not plan.country
        )

        docs: list[dict] = []
        semantic_scores: dict[str, float] = {}

        def repo_search(
            query_text: str, *, city, country, structured_: dict, limit_: int = 100
        ) -> list[dict]:
            if browse_mode:
                return self.repository.browse(
                    entity=plan.entity,
                    city=city,
                    country=country,
                    status=status,
                    is_live=is_live,
                    limit=limit_,
                    structured=structured_,
                    date_from=plan.date_from,
                    date_to=plan.date_to,
                )
            return self.repository.search(
                query_text,
                entity=plan.entity,
                city=city,
                country=country,
                status=status,
                is_live=is_live,
                limit=limit_,
                structured=structured_,
                date_from=plan.date_from,
                date_to=plan.date_to,
            )

        if not unknown_explicit_location:
            docs = repo_search(
                retrieval_query,
                city=plan.city,
                country=plan.country,
                structured_=structured,
            )

            # Descriptive structured filters stay soft when the schema cannot
            # express them; schema concepts (supplier, supplier industry,
            # supplier category) define WHAT is searched and are never dropped.
            concept_filters = {
                k: v for k, v in structured.items() if k in HARD_CONCEPT_KEYS
            }
            if not docs and set(structured) - set(concept_filters):
                docs = repo_search(
                    retrieval_query,
                    city=plan.city,
                    country=plan.country,
                    structured_=concept_filters,
                )

            if not browse_mode:
                with timed("vector_ms"):
                    for doc_id, semantic_score in self.semantic.search(
                        effective_query, limit=50
                    ):
                        semantic_scores[doc_id] = semantic_score

            if semantic_scores:
                semantic_docs = self.repository.fetch_by_ids(list(semantic_scores))
                semantic_docs = [
                    d
                    for d in semantic_docs
                    if self.repository._document_matches_filters(
                        d,
                        entity=plan.entity,
                        city=plan.city,
                        country=plan.country,
                        status=status,
                        is_live=is_live,
                        structured=structured,
                        date_from=plan.date_from,
                        date_to=plan.date_to,
                    )
                ]
                seen = {str(d.get("_id")) for d in docs}
                docs.extend(d for d in semantic_docs if str(d.get("_id")) not in seen)

            # If a typo correction occurred, search the original keyword phrase
            # too so correction can never hide an exact MongoDB match.
            if changes and corrected_query != " ".join(search_tokens):
                original_docs = repo_search(
                    " ".join(search_tokens),
                    city=plan.city,
                    country=plan.country,
                    structured_=structured,
                )
                seen = {str(d.get("_id")) for d in docs}
                docs.extend(d for d in original_docs if str(d.get("_id")) not in seen)

        # Final authoritative guard: every candidate source (lexical,
        # semantic, fallback) must satisfy the same hard constraints.
        docs = self._hard_filter_docs(docs, plan, status, is_live, structured)
        docs = [doc for doc in docs if not is_excluded(doc)]

        ranked = self._rank(
            docs, plan, retrieval_query, corrected_tokens, semantic_scores, browse_mode
        )

        # Strict (user-required) constraints need positive evidence on the
        # record itself. Candidates without evidence are never presented as
        # exact matches; they are offered as clearly-labelled related results.
        exact: list[tuple[float, dict, list[str]]] = []
        unverified: list[tuple[float, dict, list[str]]] = []
        for item in ranked:
            if strict and not satisfies_strict(item[1], evidence_filters, strict):
                unverified.append(item)
            elif plan.entity == "faq" and not self._faq_answers(item[1], search_tokens):
                # An FAQ is an exact answer only when its question covers the
                # user's question; otherwise it is merely related.
                unverified.append(item)
            else:
                exact.append(item)

        correction = corrected_query if changes else None
        results = [
            self._result_payload(doc, score, matched, correction)
            for score, doc, matched in exact[:limit]
        ]

        related_results: list[dict] = []
        related_label = None
        if not results:
            related_ranked: list[tuple[float, dict, list[str]]] = list(unverified)
            if not related_ranked and (
                plan.city
                or plan.country
                or plan.explicit_location
                or structured
                or strict
            ):
                # Exact constraints are authoritative. Do a separate relaxed
                # search for helpful alternatives rather than silently
                # returning wrong-location documents as exact matches.
                relaxed = repo_search(
                    effective_query,
                    city=None,
                    country=None,
                    structured_={
                        k: v for k, v in structured.items() if k in HARD_CONCEPT_KEYS
                    },
                    limit_=50,
                )
                relaxed = [doc for doc in relaxed if not is_excluded(doc)]
                for doc in relaxed:
                    score, matched = score_document(
                        doc, retrieval_query, float(doc.get("text_score") or 0.0)
                    )
                    related_ranked.append((score, doc, matched))
                related_ranked.sort(
                    key=lambda item: (-item[0], str(item[1].get("_id")))
                )
            related_results = [
                self._result_payload(
                    doc,
                    score,
                    list(dict.fromkeys(matched + ["related_result"])),
                    correction,
                )
                for score, doc, matched in related_ranked[:limit]
            ]
            if related_results:
                related_label = NO_EXACT_MESSAGE

        understanding["keywords"] = search_tokens
        understanding["corrected_keywords"] = corrected_tokens
        understanding["corrections"] = changes
        understanding["level"] = plan.level
        understanding["explicit_location"] = plan.explicit_location
        understanding["strict_filters"] = sorted(strict)
        understanding["browse"] = browse_mode
        understanding["notes"] = notes

        return {
            "query": original,
            "corrected_query": correction,
            "total": len(results),
            "results": results,
            "message": related_label,
            "related_results": related_results,
            "understanding": understanding,
            "exhausted": len(exact) <= limit,
        }

    # ------------------------------------------------------------------
    # FAQ, facets and relationship queries
    # ------------------------------------------------------------------

    @staticmethod
    def _faq_answers(doc: dict, query_tokens: list[str]) -> bool:
        """True when the FAQ question/aliases/keywords cover the question."""
        from .dialogue import singular

        wanted = {singular(t) for t in query_tokens if len(t) > 2}
        if not wanted:
            return False
        text = " ".join(
            str(v)
            for v in [
                doc.get("question"),
                *(doc.get("search_aliases") or []),
                *(doc.get("search_keywords") or []),
            ]
            if isinstance(v, str)
        )
        have = {singular(t) for t in tokens(text)}
        covered = len(wanted & have) / len(wanted)
        return covered >= 0.6

    def _facet(self, plan: SearchPlan, understanding: dict) -> dict:
        """Distinct values of a concept stored inside documents (DB-backed)."""
        entity, path, match = schema_map.FACETS[plan.facet]
        extra: dict = dict(match)
        for key, value in (plan.filters or {}).items():
            if key in HARD_CONCEPT_KEYS and key != "supplier_category":
                for fpath in schema_map.filter_paths(entity, key):
                    extra[fpath] = value
        values = self.repository.facet_values(
            entity=entity,
            path=path,
            extra=extra,
            city=plan.city,
            country=plan.country,
            limit=25,
        )
        understanding["facet"] = plan.facet
        understanding["facet_path"] = path
        return {
            "query": plan.original_query,
            "corrected_query": None,
            "total": 0,
            "results": [],
            "related_results": [],
            "facets": values,
            "message": None,
            "understanding": understanding,
            "exhausted": True,
        }

    def _companies_hiring(
        self, plan: SearchPlan, understanding: dict, *, status, is_live, limit: int
    ) -> dict:
        """Companies behind jobs matching a role: job.company.id -> company:<id>."""
        from .dialogue import normalize_keywords

        role_tokens = normalize_keywords([plan.related_role or ""]) or list(
            plan.keywords
        )
        docs = self.repository.search(
            " ".join(role_tokens),
            entity="job",
            city=plan.city,
            country=plan.country,
            status=status,
            is_live=is_live,
            limit=100,
            structured={},
        )
        job_plan = SearchPlan(
            entity="job", keywords=role_tokens, city=plan.city, country=plan.country
        )
        docs = self._hard_filter_docs(docs, job_plan, status, is_live, {})
        ranked = self._rank(
            docs, job_plan, " ".join(role_tokens), role_tokens, {}, False
        )
        hiring: dict[str, list[str]] = {}
        names: dict[str, str] = {}
        for _, doc, _ in ranked:
            company = doc.get("company") if isinstance(doc.get("company"), dict) else {}
            if company.get("id") is None:
                continue
            key = f"company:{company['id']}"
            hiring.setdefault(key, []).append(str(doc.get("title") or ""))
            names.setdefault(key, str(company.get("name") or ""))
        profiles = self.repository.fetch_by_ids(list(hiring)[:50])
        results = []
        for doc in profiles[: min(max(int(limit), 1), self.MAX_RESULTS)]:
            payload = self._result_payload(doc, 0.0, ["hiring_match"])
            payload["metadata"]["hiring_for"] = hiring[str(doc["_id"])][:5]
            results.append(payload)
        missing = [
            names[k]
            for k in hiring
            if k not in {str(d["_id"]) for d in profiles} and names.get(k)
        ]
        understanding["related"] = "companies_hiring"
        understanding["related_role"] = role_tokens
        understanding["hiring_companies_without_profile"] = missing[:10]
        return {
            "query": plan.original_query,
            "corrected_query": None,
            "total": len(results),
            "results": results,
            "related_results": [],
            "message": None,
            "understanding": understanding,
            "exhausted": True,
        }

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank(
        self,
        docs: list[dict],
        plan: SearchPlan,
        retrieval_query: str,
        corrected_tokens: list[str],
        semantic_scores: dict[str, float],
        browse_mode: bool,
    ) -> list[tuple[float, dict, list[str]]]:
        ranked = []
        for position, doc in enumerate(docs):
            if browse_mode:
                # Browse results keep MongoDB recency order.
                score, matched = 100.0 - position * 0.01, ["browse"]
            else:
                score, matched = score_document(
                    doc,
                    retrieval_query,
                    float(doc.get("text_score") or 0.0),
                )
            semantic_score = semantic_scores.get(str(doc.get("_id")))
            if semantic_score is not None:
                score += max(0.0, semantic_score) * 35.0
                matched.append("semantic")
            # Small, deterministic boosts for structured constraints. These
            # never override exact lexical identity.
            if plan.city or plan.country:
                matched.append("location_filter")
                score += 12

            if (
                plan.level
                or plan.department
                or plan.industry
                or plan.experience is not None
            ):
                matched.append("structured_filter")
                score += 8

            title_text = " ".join(
                str(v or "") for v in (doc.get("title"), doc.get("short_title"))
            ).casefold()

            # Explicit role/level phrases are high-value ranking signals: an
            # exact "executive chef" title outranks a document that merely
            # mentions the level and chef somewhere in its description.
            if plan.level:
                role_terms = [
                    t.casefold() for t in corrected_tokens if t.casefold() != plan.level
                ]
                if role_terms:
                    phrase = f"{plan.level} {' '.join(role_terms[:2])}"
                    if phrase in title_text:
                        score += 45
                        matched.append("role_phrase_match")

            # Level relevance: prefer explicit level metadata/title signals.
            # Do not scan the whole description: a senior job often mentions
            # the junior roles it supervises.
            if plan.level:
                level_terms = {
                    "senior": ("senior", "sr ", "sr.", "lead", "principal"),
                    "junior": ("junior", "jr ", "jr.", "entry level", "entry-level"),
                    "mid": ("mid level", "mid-level", "midlevel", "associate"),
                    "manager": ("manager", "management", "head"),
                    "executive": ("executive", "director", "vp", "vice president"),
                    "intern": ("intern", "internship", "trainee", "graduate"),
                }
                wanted_terms = level_terms.get(plan.level, (plan.level,))

                if any(term in title_text for term in wanted_terms):
                    score += 32
                    matched.append("level_match")
                else:
                    metadata = (
                        doc.get("metadata")
                        if isinstance(doc.get("metadata"), dict)
                        else {}
                    )
                    metadata_level = str(metadata.get("job_level") or "").casefold()
                    professional = (
                        doc.get("professional")
                        if isinstance(doc.get("professional"), dict)
                        else {}
                    )
                    job_level = (
                        professional.get("job_level")
                        if isinstance(professional.get("job_level"), dict)
                        else {}
                    )
                    nested_level = str(job_level.get("name") or "").casefold()
                    if any(
                        term in metadata_level or term in nested_level
                        for term in wanted_terms
                    ):
                        score += 28
                        matched.append("level_match")

                conflicts = {
                    "senior": (
                        "junior",
                        "intern",
                        "internship",
                        "trainee",
                        "apprentice",
                    ),
                    "junior": ("senior", "lead", "principal", "executive"),
                    "mid": ("junior", "senior", "executive"),
                    "manager": ("intern", "trainee", "apprentice", "commis"),
                    "executive": ("junior", "intern", "trainee", "apprentice"),
                    "intern": ("senior", "lead", "principal", "manager", "executive"),
                }
                if any(term in title_text for term in conflicts.get(plan.level, ())):
                    score -= 24
                    matched.append("level_conflict")

            # Experience is a relevance signal when the document exposes years
            # in its searchable text; sparse documents remain eligible.
            if plan.experience is not None:
                exp_matches = re.findall(
                    r"\b(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?\s*\+?\s*(?:years?|yrs?)\b",
                    " ".join(
                        str(v or "")
                        for v in (
                            doc.get("description"),
                            doc.get("ai_search_text"),
                            doc.get("title"),
                        )
                    ).casefold(),
                )
                numeric_experience = [
                    float(second or first) for first, second in exp_matches
                ]
                if numeric_experience:
                    if max(numeric_experience) >= float(plan.experience):
                        score += 10
                        matched.append("experience_match")
                    else:
                        score -= 10
                        matched.append("experience_below_request")
            if plan.category:
                matched.append("category_filter")
                score += 5
            if plan.date_from or plan.date_to:
                matched.append("date_filter")
                score += 3
            ranked.append((score, doc, matched))

        def recency(doc: dict) -> float:
            # Module date fields differ (jobs/articles: ISO strings in metadata).
            item = schema_map.schema(str(doc.get("entity_type") or ""))
            for path in item.created if item else ("created_at",):
                for value in all_values(doc, path):
                    parsed = parse_date(value)
                    if parsed:
                        return parsed.timestamp()
            return 0.0

        # Equal relevance: newer records first, then a stable id order.
        ranked.sort(
            key=lambda item: (-item[0], -recency(item[1]), str(item[1].get("_id")))
        )
        return ranked
