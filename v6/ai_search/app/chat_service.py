"""Phase 3 conversational orchestration.

User message
  -> conversation state (MongoDB)
  -> deterministic turn interpretation (dialogue.py)
  -> explicit state merge (state.py)
  -> MongoDB search from state (SearchService.execute_plan)
  -> top 5 real results, result history update
  -> one optional LLM call to phrase the answer (Qwen3 via Ollama)
  -> validated answer, persisted turn

The chat turn is split into prepare() / LLM / finalize() so the WebSocket
endpoint can stream model tokens between the deterministic parts.
"""

from __future__ import annotations

import copy
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from . import answers
from .config import settings
from .conversation import ConversationRepository
from .dialogue import ENTITY_LABELS, TurnIntent, entity_label, interpret
from .evidence import company_name
from .observability import (
    current_timings,
    ensure_timings,
    log_event,
    metrics,
    set_conversation_id,
)
from .ollama_client import LlmUnavailable, OllamaChatClient
from .query_understanding import LOCATION_QUESTION, TYPE_QUESTIONS
from .service import SearchService
from .state import (
    apply_intent,
    empty_state,
    has_location,
    has_search,
    history_lookup,
    latest_page,
    normalize_state,
    numbered_list,
    public_state,
    record_results,
    resolve_refs,
    set_focus,
    to_plan,
)
from .security import valid_conversation_id

CONCEPT_FILTERS = {"is_supplier", "industry_context", "supplier_category"}


def _present(value: Any) -> bool:
    if isinstance(value, dict):
        return any(v for v in value.values())
    return bool(value)


ORDINAL_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
LOCATION_REQUIRED = {"job", "professional"}


@dataclass
class PreparedTurn:
    conversation_id: str
    version: int | None
    user_message: str
    action: str
    state: dict[str, Any]
    response: dict[str, Any]
    fallback_answer: str
    llm_kind: str | None = None
    llm_messages: list[dict[str, str]] | None = None
    max_tokens: int = 320
    allowed_urls: set[str] = field(default_factory=set)
    allowed_counts: set[int] = field(default_factory=set)
    result_refs: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)


class ChatService:
    """Conversational orchestration over the Phase 2 search service."""

    def __init__(
        self,
        search_service: SearchService,
        conversations: ConversationRepository,
        llm: OllamaChatClient | None = None,
    ):
        self.search = search_service
        self.conversations = conversations
        self.llm = llm or OllamaChatClient()

    # ------------------------------------------------------------------
    # Compatibility helpers (used by routes and older callers)
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self.llm.model

    @property
    def base_url(self) -> str:
        return self.llm.base_url

    @staticmethod
    def _action(message: str) -> str:
        action = interpret(
            message, {"entity": "job", "keywords": ["x"], "last_results": ["job:1"]}
        ).action
        return {
            "detail": "search",
            "smalltalk": "search",
            "clarify": "search",
            "related_entity": "search",
        }.get(action, action)

    @staticmethod
    def _public_state(state: dict[str, Any]) -> dict[str, Any]:
        return public_state(normalize_state(state))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        limit: int = 5,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        if valid_conversation_id(conversation_id):
            # Serialize turns of one conversation within this worker; the
            # versioned write in finalize() protects across workers.
            with self.conversations.lock(str(conversation_id)):
                return self._chat_once(message, conversation_id, limit, owner_id)
        return self._chat_once(message, conversation_id, limit, owner_id)

    def _chat_once(
        self,
        message: str,
        conversation_id: str | None,
        limit: int,
        owner_id: str | None,
    ) -> dict[str, Any]:
        turn = self.prepare(
            message=message,
            conversation_id=conversation_id,
            limit=limit,
            owner_id=owner_id,
        )
        return self.finalize(turn, self.generate(turn))

    def generate(self, turn: PreparedTurn) -> str | None:
        """Blocking LLM call for the non-streaming path."""
        if not turn.llm_messages:
            return None
        try:
            return self.llm.complete(turn.llm_messages, max_tokens=turn.max_tokens).text
        except LlmUnavailable as exc:
            self.note_fallback(turn, exc.reason)
            return None

    def note_fallback(self, turn: PreparedTurn, reason: str) -> None:
        metrics.incr("llm_fallback")
        metrics.incr(f"llm_fallback_{reason}")
        turn.response["llm"] = {"used": False, "fallback_reason": reason}
        log_event(
            "llm_fallback",
            level=logging.WARNING,
            reason=reason,
            kind=turn.llm_kind,
            action=turn.action,
        )

    def prepare(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        limit: int = 5,
        owner_id: str | None = None,
    ) -> PreparedTurn:
        ensure_timings()
        message = " ".join((message or "").strip().split())[
            : settings.chat_max_message_chars
        ]
        conversation = self.conversations.ensure(conversation_id, owner_id=owner_id)
        cid = conversation["conversation_id"]
        set_conversation_id(cid)
        state = normalize_state(conversation.get("state"))
        history = conversation.get("messages") or []
        version = conversation.get("version")
        limit = min(max(int(limit), 1), settings.max_results)

        intent = interpret(message, state)
        handler = {
            "reset": self._reset,
            "smalltalk": self._smalltalk,
            "clarify": self._clarify_generic,
            "more": self._more,
            "compare": self._compare,
            "detail": self._detail,
            "related_entity": self._related_entity,
            "faq": self._faq,
            "facet": self._facet,
        }.get(intent.action, self._search)

        new_state = copy.deepcopy(state)
        new_state["turn"] = int(new_state.get("turn") or 0) + 1
        if intent.action in {"more", "compare", "detail", "related_entity"}:
            # These act on the current search without changing it.
            intent.transition = "continuation"
            intent.transition_reason = f"'{intent.action}' on the current {state.get('entity') or 'search'} results"
            new_state["last_transition"] = {
                "transition": "continuation",
                "reason": intent.transition_reason,
                "is_new_search": False,
                "is_context_continuation": True,
                "inherited": [
                    k
                    for k in ("entity", "keywords", "location", "filters")
                    if _present(state.get(k))
                ],
                "discarded": [],
            }
        turn = PreparedTurn(
            conversation_id=cid,
            version=version if isinstance(version, int) else 0,
            user_message=message,
            action=intent.action,
            state=new_state,
            response={},
            fallback_answer="",
        )
        handler(turn, intent, state, history, limit)
        turn.response.setdefault("results", [])
        turn.response.setdefault("related_results", [])
        turn.response.setdefault(
            "understanding", self._understanding(intent, turn.state, {})
        )
        turn.response["action"] = turn.action
        turn.response["conversation_id"] = cid
        turn.response["state"] = public_state(turn.state)
        turn.response["references"] = self._references(turn.state)
        turn.response.setdefault("suggestions", self._suggestions(turn))
        turn.result_refs = list(turn.state.get("last_results") or [])
        if turn.llm_messages and not self.llm.enabled:
            # Deterministic-only mode (CHAT_LLM_ENABLED=false or no Ollama
            # configured) is a configuration, not a failure.
            turn.llm_messages = None
            turn.response["llm"] = {"used": False, "fallback_reason": "disabled"}
            metrics.incr("llm_disabled_answers")
        return turn

    def finalize(self, turn: PreparedTurn, llm_text: str | None) -> dict[str, Any]:
        answer = None
        if llm_text is not None:
            answer = answers.validate_answer(
                llm_text,
                allowed_urls=turn.allowed_urls,
                allowed_counts=turn.allowed_counts,
            )
            if answer is None:
                self.note_fallback(turn, "invalid_output")
            else:
                turn.response["llm"] = {"used": True}
        elif turn.llm_messages and "llm" not in turn.response:
            self.note_fallback(turn, "no_output")
        answer = answer or turn.fallback_answer
        turn.response["answer"] = answer

        self.conversations.save_turn(
            turn.conversation_id,
            user_message=turn.user_message,
            assistant_message=answer,
            state=turn.state,
            action=turn.action,
            expected_version=turn.version,
            result_refs=turn.result_refs,
        )
        total_ms = round((time.perf_counter() - turn.started) * 1000, 1)
        timings = current_timings()
        metrics.incr("chat_turns")
        metrics.incr(f"chat_action_{turn.action}")
        metrics.observe("chat_ms", total_ms)
        fields: dict[str, Any] = {
            "action": turn.action,
            "results": len(turn.response.get("results") or []),
            "related": len(turn.response.get("related_results") or []),
            "message_chars": len(turn.user_message),
            "llm_used": bool((turn.response.get("llm") or {}).get("used")),
            "fallback_reason": (turn.response.get("llm") or {}).get("fallback_reason"),
            "total_ms": total_ms,
            **timings,
        }
        log_event("chat_turn", **fields)
        return turn.response

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _reset(
        self, turn: PreparedTurn, intent: TurnIntent, state, history, limit
    ) -> None:
        turn.state = empty_state()
        turn.fallback_answer = "Sure — starting a new search. What are you looking for?"

    def _smalltalk(
        self, turn: PreparedTurn, intent: TurnIntent, state, history, limit
    ) -> None:
        turn.fallback_answer = answers.smalltalk_answer(intent.smalltalk)

    def _clarify_generic(
        self, turn: PreparedTurn, intent: TurnIntent, state, history, limit
    ) -> None:
        turn.action = "clarify"
        turn.fallback_answer = (
            "What would you like me to search for? You can ask for jobs, professionals, companies, "
            'products, articles, events, awards or FAQs — for example "Find chef jobs in Dubai".'
        )

    def _clarification(
        self, state: dict[str, Any], intent: TurnIntent
    ) -> tuple[str | None, str | None]:
        """Return (question, pending field) when the merged state is under-specified."""
        entity = state.get("entity")
        filters = state.get("filters") or {}
        topic = bool(
            state.get("keywords")
            or filters.get("department")
            or filters.get("industry")
            or filters.get("category")
        )
        topic = (
            topic or bool(CONCEPT_FILTERS & set(filters)) or bool(state.get("related"))
        )
        if not has_search(state):
            return None, None
        if entity and not topic and not has_location(state):
            # "Find me a job" is too vague; "Find professionals in Dubai" is a
            # valid browse request and is searched directly.
            return TYPE_QUESTIONS.get(entity), "keywords"
        if (
            entity in LOCATION_REQUIRED
            and not has_location(state)
            and not state.get("location_any")
            and not state.get("location_asked")
            and state.get("entity_source") in {"text", "intent"}
        ):
            return LOCATION_QUESTION, "location"
        return None, None

    def _search(
        self, turn: PreparedTurn, intent: TurnIntent, state, history, limit
    ) -> None:
        new_state = apply_intent(turn.state, intent)
        if intent.set_entity and intent.plan is not None:
            new_state["entity_source"] = intent.plan.entity_source or "text"
            new_state["location_asked"] = False
        turn.state = new_state

        if not has_search(new_state):
            self._clarify_generic(turn, intent, state, history, limit)
            return

        question, pending = self._clarification(new_state, intent)
        if question:
            turn.action = "clarify"
            new_state["pending"] = {"field": pending}
            if pending == "location":
                new_state["location_asked"] = True
                previous = (
                    (new_state.get("last_transition") or {}).get("previous") or {}
                ).get("location") or {}
                city = (
                    previous.get("city")
                    or previous.get("raw")
                    or previous.get("country")
                )
                turn.response["suggestions"] = [
                    s for s in (city, "Dubai", "Abu Dhabi", "Anywhere") if s
                ][:4]
                turn.response["suggestions"] = list(
                    dict.fromkeys(turn.response["suggestions"])
                )
            turn.fallback_answer = question
            turn.response["understanding"] = self._understanding(
                intent, new_state, {"clarification": question}
            )
            return

        plan = to_plan(new_state)
        result = self.search.execute_plan(
            plan, original=plan.original_query, limit=limit, clarify=False
        )
        self._present_search(turn, intent, result, more=False, history=history)

    def _more(
        self, turn: PreparedTurn, intent: TurnIntent, state, history, limit
    ) -> None:
        if not has_search(state):
            turn.action = "more"
            turn.fallback_answer = (
                "Run a search first, then I can show you more results."
            )
            return
        plan = to_plan(turn.state)
        result = self.search.execute_plan(
            plan,
            original=plan.original_query,
            limit=limit,
            exclude_ids=list(turn.state.get("shown") or []),
            clarify=False,
        )
        self._present_search(turn, intent, result, more=True, history=history)

    def _present_search(
        self,
        turn: PreparedTurn,
        intent: TurnIntent,
        result: dict,
        *,
        more: bool,
        history,
    ) -> None:
        results = result.get("results") or []
        related = result.get("related_results") or []
        state = turn.state
        entries: list[dict[str, Any]] = []
        if results:
            entries = record_results(state, results, append=more)
        elif related:
            entries = record_results(state, related, related=True, append=more)
        elif not more:
            state["last_results"] = []
            state["current_list"] = []
        state["pending"] = None
        numbers = {entry["key"]: entry["position"] for entry in entries}
        for item in (*results, *related):
            item["number"] = numbers.get(
                f"{item.get('entity_type')}:{item.get('entity_id')}"
            )

        summary = answers.search_answer(
            state, results, related, more=more, corrected=result.get("corrected_query")
        )
        turn.fallback_answer = summary
        turn.response.update(
            {
                "results": results,
                "related_results": related,
                "message": result.get("message"),
                "understanding": self._understanding(
                    intent, state, result.get("understanding") or {}
                ),
            }
        )
        shown = results or related
        turn.allowed_urls = {r["url"] for r in shown if r.get("url")}
        turn.allowed_counts = set(range(0, max(len(results), len(related)) + 1))
        if shown:
            turn.llm_kind = "more" if more else "search"
            turn.llm_messages = answers.build_messages(
                turn.llm_kind,
                user_message=turn.user_message,
                data=answers.search_llm_data(
                    public_state(state), summary, results, related
                ),
                history=history,
            )
            turn.max_tokens = settings.ollama_max_answer_tokens

    def _entries_for_compare(
        self, state: dict[str, Any], intent: TurnIntent
    ) -> tuple[list[dict], str | None]:
        latest = numbered_list(state)
        if intent.ref_mode in {"ordinal", "last"} and intent.refs:
            entries = resolve_refs(state, intent)
            if entries:
                return entries, None
            # Asked for more records than the latest list holds.
            if (
                len(latest) >= 2
                and intent.ref_mode == "ordinal"
                and intent.refs == list(range(len(intent.refs)))
            ):
                return latest, (
                    f"The current list only has {len(latest)} results, so here is a comparison of those."
                )
            return [], None
        if intent.ref_mode in {"focus", "previous"}:
            return resolve_refs(state, intent), None
        return latest_page(state)[:5], None

    def _compare(
        self, turn: PreparedTurn, intent: TurnIntent, state, history, limit
    ) -> None:
        entries, note = self._entries_for_compare(state, intent)
        requested = intent.requested_count or len(intent.refs) or 2
        if len(entries) < 2:
            count_word = {2: "two", 3: "three", 4: "four", 5: "five"}.get(
                requested, str(requested)
            )
            if not state.get("last_results"):
                turn.fallback_answer = f"I don't have {count_word} recent results to compare yet. Run a search first."
            else:
                turn.fallback_answer = (
                    f"I don't have {count_word} results in the current list to compare. "
                    'Try "compare the first two" or run a broader search.'
                )
            return

        docs = self.search.repository.fetch_by_ids(
            [e["doc_id"] for e in entries if e.get("doc_id")]
        )
        by_id = {str(d.get("_id")): d for d in docs}
        pairs = [(e, by_id[e["doc_id"]]) for e in entries if e.get("doc_id") in by_id]
        missing = len(entries) - len(pairs)
        if len(pairs) < 2:
            turn.fallback_answer = "Those records are no longer available, so I can't compare them. Try running the search again."
            return
        if missing:
            extra = f"{missing} of the requested records are no longer available."
            note = f"{note} {extra}" if note else extra

        entries_ok = [p[0] for p in pairs]
        docs_ok = [p[1] for p in pairs]
        comparison = answers.build_comparison(
            entries_ok, docs_ok, intent.compare_fields
        )
        results = [
            {
                **SearchService._result_payload(doc, 0.0, ["compared"]),
                "number": entry.get("position"),
            }
            for entry, doc in pairs
        ]
        turn.response.update(
            {"results": results, "comparison": comparison, "message": note}
        )
        turn.fallback_answer = answers.comparison_answer(
            comparison, intent.compare_fields, note
        )
        turn.allowed_urls = {r["url"] for r in results if r.get("url")}
        turn.allowed_counts = set(range(0, len(results) + 1))
        turn.llm_kind = "compare"
        turn.llm_messages = answers.build_messages(
            "compare",
            user_message=turn.user_message,
            data=answers.comparison_llm_data(comparison, intent.compare_fields),
            history=history,
        )
        turn.max_tokens = settings.ollama_max_compare_tokens

    def _detail(
        self, turn: PreparedTurn, intent: TurnIntent, state, history, limit
    ) -> None:
        entries = resolve_refs(state, intent)
        if not entries:
            if intent.refs and intent.refs[0] >= 0:
                word = ORDINAL_WORDS.get(intent.refs[0] + 1, f"#{intent.refs[0] + 1}")
                available = len(numbered_list(state))
                turn.fallback_answer = (
                    f"I don't have a {word} result in the current list"
                    + (
                        f" (it has {available})."
                        if available
                        else ". Run a search first."
                    )
                )
            else:
                turn.fallback_answer = "I'm not sure which result you mean. Run a search first, then refer to a result by its number."
            return
        entry = entries[0]
        docs = (
            self.search.repository.fetch_by_ids([entry["doc_id"]])
            if entry.get("doc_id")
            else []
        )
        if not docs:
            turn.fallback_answer = (
                f"{entry.get('title') or 'That record'} is no longer available."
            )
            return
        doc = docs[0]
        detail = answers.detail_fields(doc, entry)
        set_focus(turn.state, entry["key"])
        result = {
            **SearchService._result_payload(doc, 0.0, ["referenced"]),
            "number": entry.get("position"),
        }
        results = [result]
        company_profile = None
        low = turn.user_message.casefold()
        if (
            intent.question_field == "company"
            and doc.get("entity_type") != "company"
            and detail.get("company")
            and re.search(r"\b(?:tell|about|profile|more|details?|show|who)\b", low)
        ):
            profiles = self.search.repository.find_by_titles(
                "company", [detail["company"]], limit=1
            )
            if profiles:
                company_profile = SearchService._result_payload(
                    profiles[0], 0.0, ["company_profile"]
                )
                results.append(company_profile)

        turn.response.update({"results": results, "detail": detail})
        answer = answers.detail_answer(detail, intent.question_field, intent.open_link)
        if intent.question_field == "company" and detail.get("company"):
            answer += " I found their company profile below." if company_profile else ""
        turn.fallback_answer = answer
        turn.allowed_urls = {r["url"] for r in results if r.get("url")}
        turn.allowed_counts = {0, 1}
        # Field questions and "open" are answered exactly and instantly; the
        # LLM only summarizes a record when the user asked for more about it.
        if (
            intent.question_field is None
            and not intent.open_link
            and detail.get("description")
        ):
            turn.llm_kind = "detail"
            turn.llm_messages = answers.build_messages(
                "detail",
                user_message=turn.user_message,
                data=answers.detail_llm_data(detail, intent.question_field),
                history=history,
            )
            turn.max_tokens = settings.ollama_max_answer_tokens

    def _related_entity(
        self, turn: PreparedTurn, intent: TurnIntent, state, history, limit
    ) -> None:
        entries = (
            resolve_refs(state, intent)
            if intent.ref_mode not in {None, "all"}
            else resolve_refs(state, TurnIntent(ref_mode="all"))
        )
        docs = self.search.repository.fetch_by_ids(
            [e["doc_id"] for e in entries if e.get("doc_id")]
        )
        names: list[str] = []
        for doc in docs:
            name = company_name(doc)
            if name and name not in names:
                names.append(name)
        if not names:
            turn.fallback_answer = "The latest results don't include company information, so I can't tell which companies are behind them."
            return
        profiles = self.search.repository.find_by_titles("company", names, limit=limit)
        results = [
            SearchService._result_payload(p, 0.0, ["company_profile"]) for p in profiles
        ]
        listed = ", ".join(names[:5])
        if results:
            for entry, item in zip(
                record_results(turn.state, results, mark_shown=False), results
            ):
                item["number"] = entry["position"]
            answer = (
                f"The latest results are from {listed}. "
                f"I found {len(results)} matching company profile{'s' if len(results) != 1 else ''}."
            )
        else:
            answer = f"The latest results are from {listed}. I couldn't find company profiles for them."
        turn.response.update({"results": results})
        turn.fallback_answer = answer

    def _faq(
        self, turn: PreparedTurn, intent: TurnIntent, state, history, limit
    ) -> None:
        """Information question: answered only from FAQ records in MongoDB."""
        plan = intent.plan
        new_state = empty_state()
        for key in (
            "result_history",
            "current_list",
            "last_results",
            "focus",
            "previous_focus",
            "turn",
        ):
            new_state[key] = copy.deepcopy(turn.state.get(key))
        new_state.update(
            {
                "entity": "faq",
                "entity_source": "faq",
                "keywords": list(plan.keywords if plan else []),
            }
        )
        new_state["last_transition"] = {
            "transition": "faq",
            "reason": intent.transition_reason,
            "is_new_search": True,
            "is_context_continuation": False,
            "inherited": [],
            "discarded": [
                k
                for k in ("entity", "keywords", "location", "filters")
                if _present(state.get(k))
            ],
            "previous": {
                "entity": state.get("entity"),
                "keywords": state.get("keywords") or [],
            },
        }
        new_state["topic"] = None
        turn.state = new_state
        result = self.search.execute_plan(
            to_plan(new_state),
            original=plan.original_query if plan else "",
            limit=limit,
            clarify=False,
        )
        results = result.get("results") or []
        related = result.get("related_results") or []
        entries = record_results(new_state, results or related, related=not results)
        numbers = {entry["key"]: entry["position"] for entry in entries}
        for item in (*results, *related):
            item["number"] = numbers.get(
                f"{item.get('entity_type')}:{item.get('entity_id')}"
            )
        if results:
            # FAQ responses are answer-only. The UI should not turn an information
            # question into a list of FAQ links/cards or append a secondary FAQ CTA.
            top = results[0]
            answer = (top["metadata"].get("answer") or top.get("description") or "").strip()
            turn.fallback_answer = answer
        else:
            # A question can be useful even when no FAQ record matches exactly.
            # Fall back to deterministic platform guidance rather than inventing
            # database facts or exposing loosely related FAQ records.
            guidance = answers.generic_guidance_answer(turn.user_message)
            if guidance:
                turn.fallback_answer = guidance
            elif related:
                turn.fallback_answer = (
                    "I couldn't find an exact FAQ answer for that. "
                    "If you tell me what you want to do on Hozpitality, I can guide you from the available platform information."
                )
            else:
                turn.fallback_answer = (
                    "I couldn't find an answer to that in the Hozpitality information available to me. "
                    "Try rephrasing the question or tell me what you want to do on the platform."
                )
        turn.response.update(
            {
                "results": results,
                "related_results": related,
                "understanding": self._understanding(
                    intent, new_state, result.get("understanding") or {}
                ),
            }
        )
        # FAQ answers are returned verbatim from the database; no LLM call.

    def _facet(
        self, turn: PreparedTurn, intent: TurnIntent, state, history, limit
    ) -> None:
        """List a concept stored inside documents (e.g. supplier categories)."""
        plan = intent.plan
        new_state = apply_intent(
            turn.state,
            TurnIntent(
                action="search",
                transition="new_search",
                transition_reason=intent.transition_reason,
                set_entity=plan.entity if plan else None,
                set_filters={
                    k: v
                    for k, v in (plan.filters if plan else {}).items()
                    if k in CONCEPT_FILTERS
                },
                set_location=(
                    {"city": plan.city, "country": plan.country, "raw": None}
                    if plan and (plan.city or plan.country)
                    else None
                ),
                plan=plan,
            ),
        )
        new_state["entity_source"] = "concept"
        turn.state = new_state
        result = (
            self.search.execute_plan(plan, original=plan.original_query, clarify=False)
            if plan
            else {}
        )
        values = result.get("facets") or []
        from .schema_map import FACET_LABELS

        label = FACET_LABELS.get(plan.facet if plan else "", "values")
        if values:
            listed = ", ".join(f"{v['name']} ({v['count']})" for v in values[:15])
            turn.fallback_answer = f"These are the {label} on Hozpitality: {listed}."
            if plan and plan.facet == "supplier_category":
                turn.response["suggestions"] = [
                    f"Find suppliers in supplier category {v['name']}"
                    for v in values[:3]
                ]
        else:
            turn.fallback_answer = (
                f"I couldn't find any {label} in the Hozpitality data."
            )
        turn.response.update(
            {
                "facets": values,
                "understanding": self._understanding(
                    intent, new_state, result.get("understanding") or {}
                ),
            }
        )

    @staticmethod
    def _understanding(
        intent: TurnIntent, state: dict[str, Any], extra: dict[str, Any]
    ) -> dict[str, Any]:
        """Debuggable account of the turn: entity choice, transition, filters."""
        plan = intent.plan
        context = dict(state.get("last_transition") or {})
        context.pop("previous", None)
        return {
            **extra,
            "turn": intent.as_dict(),
            "entity": state.get("entity"),
            "entity_reason": plan.entity_reason if plan else None,
            "is_faq": intent.action == "faq",
            "is_new_search": bool(context.get("is_new_search")),
            "is_context_continuation": bool(context.get("is_context_continuation")),
            "context": context,
        }

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _references(state: dict[str, Any]) -> list[dict[str, Any]]:
        refs = []
        for key in state.get("last_results") or []:
            entry = history_lookup(state, key)
            if entry:
                refs.append(
                    {
                        "number": entry.get("position"),
                        "key": key,
                        "entity_type": entry.get("entity_type"),
                        "entity_id": entry.get("entity_id"),
                        "title": entry.get("title"),
                        "url": entry.get("url"),
                        "related": entry.get("related", False),
                    }
                )
        return refs

    @staticmethod
    def _suggestions(turn: PreparedTurn) -> list[str]:
        action = turn.action
        state = turn.state
        results = turn.response.get("results") or []
        if action == "clarify":
            pending = (state.get("pending") or {}).get("field")
            if pending == "location":
                return ["Dubai", "Abu Dhabi", "Anywhere"]
            return []
        if action in {"search", "more"} and results:
            out = ["Show me more"]
            if len(results) >= 2:
                out.append(
                    f"Compare the first {min(3, len(results))}"
                    if len(results) >= 3
                    else "Compare the first two"
                )
            out.append("Tell me more about the first one")
            if state.get("entity") == "job" and "level" not in (
                state.get("filters") or {}
            ):
                out.append("Only management positions")
            return out[:4]
        if action in {"search", "more"} and not results:
            out = []
            if has_location(state):
                out.append("Anywhere")
            if state.get("filters"):
                out.append("Remove all filters")
            return out
        if action == "reset":
            return [
                "Find chef jobs in Dubai",
                "Hotel companies in Dubai",
                "Articles about hotel technology",
            ]
        return []


__all__ = ["ChatService", "PreparedTurn", "ENTITY_LABELS", "entity_label"]
