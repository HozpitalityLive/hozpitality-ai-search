"""Conversation state: shape, legacy migration, merging and result history.

State document (stored under ``state`` in ai_search_conversations):

{
  "entity": "job",
  "location": {"city": "Dubai", "country": "United Arab Emirates", "raw": null},
  "location_any": false,              # user explicitly asked for "anywhere"
  "keywords": ["chef"],
  "filters": {"level": "manager", "accommodation": true},
  "strict_filters": ["level", "accommodation"],
  "pending": null | {"field": "keywords" | "location"},
  "topic": "<signature of entity+keywords+location>",
  "shown": ["job:1", ...],            # everything displayed for this topic (<= 100)
  "current_list": ["job:1", ...],     # numbered list of the current search incl.
                                      # "show more" pages: #1..#n (<= 50)
  "last_results": ["job:6"],          # the page displayed by the latest turn (<= 5)
  "result_history": [{...}],          # bounded reference history (<= 50)
  "focus": "job:1" | null,            # last result the user referred to
  "previous_focus": "job:9" | null,
  "turn": 3
}
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from .config import settings
from .dialogue import FILTER_ENTITIES, TurnIntent
from .query_understanding import SearchPlan

REPO_FILTER_KEYS = {
    "salary_min",
    "salary_currency",
    "employment_type",
    "verified",
    "featured",
    "currently_working",
    "accommodation",
}
PLAN_FIELD_KEYS = {"level", "experience", "department", "industry", "category"}


def empty_state() -> dict[str, Any]:
    return {
        "entity": None,
        "entity_source": None,
        "location": {"city": None, "country": None, "raw": None},
        "location_asked": False,
        "location_any": False,
        "keywords": [],
        "filters": {},
        "strict_filters": [],
        "pending": None,
        "topic": None,
        "shown": [],
        "current_list": [],
        "last_results": [],
        "result_history": [],
        "focus": None,
        "previous_focus": None,
        "turn": 0,
    }


def normalize_state(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Load a stored state, migrating the Phase 3 prototype shape."""
    state = empty_state()
    if not isinstance(raw, dict):
        return state
    for key in state:
        if key in raw and raw[key] is not None:
            state[key] = copy.deepcopy(raw[key])
    location = raw.get("location") if isinstance(raw.get("location"), dict) else {}
    state["location"] = {
        "city": location.get("city"),
        "country": location.get("country"),
        "raw": location.get("raw"),
    }
    state["filters"] = dict(raw.get("filters") or {})
    state["keywords"] = [str(k) for k in (raw.get("keywords") or [])][:12]

    # Legacy prototype stored full result dicts in last_result_items and
    # "type:id" keys in last_results. Rebuild a proper history from them.
    legacy_items = raw.get("last_result_items")
    if legacy_items and not raw.get("result_history"):
        entries = [
            history_entry(item, turn=0, position=i + 1)
            for i, item in enumerate(legacy_items)
        ]
        state["result_history"] = entries
        state["last_results"] = [entry["key"] for entry in entries]
        state["current_list"] = list(state["last_results"])
        state["shown"] = list(state["last_results"])
    if not state["topic"] and (state["entity"] or state["keywords"]):
        state["topic"] = topic_signature(state)
    return state


def public_state(state: dict[str, Any]) -> dict[str, Any]:
    location = state.get("location") or {}
    return {
        "entity": state.get("entity"),
        "location": {
            "city": location.get("city"),
            "country": location.get("country"),
            **({"raw": location["raw"]} if location.get("raw") else {}),
        },
        "keywords": list(state.get("keywords") or []),
        "filters": dict(state.get("filters") or {}),
        "strict_filters": list(state.get("strict_filters") or []),
        "last_results": list(state.get("last_results") or []),
        "current_list": list(state.get("current_list") or []),
        "pending": state.get("pending"),
    }


def topic_signature(state: dict[str, Any]) -> str:
    """Identity of the current topic; filters are refinements of a topic."""
    location = state.get("location") or {}
    payload = {
        "entity": state.get("entity"),
        "keywords": sorted(state.get("keywords") or []),
        "location": [
            location.get("city"),
            location.get("country"),
            location.get("raw"),
        ],
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def has_search(state: dict[str, Any]) -> bool:
    return bool(state.get("entity") or state.get("keywords"))


def has_location(state: dict[str, Any]) -> bool:
    location = state.get("location") or {}
    return bool(location.get("city") or location.get("country") or location.get("raw"))


def apply_intent(state: dict[str, Any], intent: TurnIntent) -> dict[str, Any]:
    """Merge the explicit changes of one message into the state (pure)."""
    new = copy.deepcopy(state)
    filters: dict[str, Any] = dict(new.get("filters") or {})
    strict: list[str] = list(new.get("strict_filters") or [])

    if intent.fresh or intent.clear_filters:
        filters, strict = {}, []
    if intent.fresh:
        new["location_asked"] = False

    if intent.set_entity and intent.set_entity != new.get("entity"):
        previous = new.get("entity")
        new["entity"] = intent.set_entity
        if previous:
            # Keep context that applies to the new entity; drop the rest.
            filters = {
                key: value
                for key, value in filters.items()
                if intent.set_entity in FILTER_ENTITIES.get(key, {intent.set_entity})
            }
            # A restriction such as "only management positions" described
            # the previous entity; it becomes a ranking preference.
            strict = []

    if intent.remove_location:
        new["location"] = {"city": None, "country": None, "raw": None}
        new["location_any"] = True
    if intent.set_location:
        new["location"] = {
            "city": intent.set_location.get("city"),
            "country": intent.set_location.get("country"),
            "raw": intent.set_location.get("raw"),
        }
        new["location_any"] = False

    if intent.set_keywords is not None:
        new["keywords"] = list(intent.set_keywords)[:12]
    elif intent.add_keywords:
        new["keywords"] = list(
            dict.fromkeys([*(new.get("keywords") or []), *intent.add_keywords])
        )[:12]

    for key in intent.remove_filters:
        filters.pop(key, None)
        if key in strict:
            strict.remove(key)

    for key, value in intent.set_filters.items():
        filters[key] = value
        if key in intent.strict_filters:
            if key not in strict:
                strict.append(key)
        elif key in strict:
            strict.remove(key)

    new["filters"] = filters
    new["strict_filters"] = [key for key in strict if key in filters]

    # A pending clarification is satisfied once its field is present.
    pending = new.get("pending") or {}
    if pending.get("field") == "keywords" and (
        new.get("keywords") or filters.get("department") or filters.get("industry")
    ):
        new["pending"] = None
    elif pending.get("field") == "location" and (
        has_location(new) or new.get("location_any")
    ):
        new["pending"] = None

    topic = topic_signature(new)
    if topic != new.get("topic"):
        new["topic"] = topic
        new["shown"] = []
    return new


def to_plan(state: dict[str, Any]) -> SearchPlan:
    """Build a search plan from authoritative state (no text re-parsing)."""
    filters = dict(state.get("filters") or {})
    location = state.get("location") or {}
    city = location.get("city")
    # A known city implies its country; searching the city alone keeps
    # documents that only carry a country code.
    country = None if city else location.get("country")
    plan = SearchPlan(
        intent="search",
        entity=state.get("entity"),
        keywords=list(state.get("keywords") or []),
        city=city,
        country=country,
        experience=filters.get("experience"),
        level=filters.get("level"),
        department=filters.get("department"),
        industry=filters.get("industry"),
        category=filters.get("category"),
        filters={k: v for k, v in filters.items() if k in REPO_FILTER_KEYS},
        explicit_location=bool(city or location.get("country") or location.get("raw")),
        entity_source="state",
        location_text=location.get("raw"),
        strict_filters=[k for k in (state.get("strict_filters") or []) if k in filters],
    )
    plan.original_query = " ".join(plan.keywords)
    return plan


# ---------------------------------------------------------------------------
# Result history
# ---------------------------------------------------------------------------


def history_entry(
    result: dict[str, Any], *, turn: int, position: int, related: bool = False
) -> dict[str, Any]:
    location = result.get("location") or {}
    return {
        "key": f"{result.get('entity_type')}:{result.get('entity_id')}",
        "entity_type": result.get("entity_type"),
        "entity_id": str(result.get("entity_id") or ""),
        "doc_id": str(result.get("doc_id") or ""),
        "title": result.get("title"),
        "url": result.get("url"),
        "company": result.get("company"),
        "city": location.get("city"),
        "country": location.get("country"),
        "turn": turn,
        "position": position,
        "related": related,
    }


def record_results(
    state: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    related: bool = False,
    mark_shown: bool = True,
    append: bool = False,
) -> list[dict[str, Any]]:
    """Store a displayed page and return its history entries.

    ``append=True`` continues the numbering of the current search ("show me
    more" -> #6..#10); otherwise the page starts a new numbered list.
    """
    turn = int(state.get("turn") or 0)
    current = list(state.get("current_list") or []) if append else []
    offset = len(current)
    entries = [
        history_entry(r, turn=turn, position=offset + i + 1, related=related)
        for i, r in enumerate(results)
    ]
    keys = [entry["key"] for entry in entries]
    state["last_results"] = keys
    state["current_list"] = (current + [k for k in keys if k not in current])[
        -settings.chat_max_history :
    ]
    history = [
        h for h in (state.get("result_history") or []) if h.get("key") not in set(keys)
    ]
    history.extend(entries)
    state["result_history"] = history[-settings.chat_max_history :]
    if mark_shown:
        shown = list(state.get("shown") or [])
        shown.extend(k for k in keys if k not in shown)
        state["shown"] = shown[-settings.chat_max_shown :]
    return entries


def history_lookup(state: dict[str, Any], key: str | None) -> dict[str, Any] | None:
    if not key:
        return None
    for entry in reversed(state.get("result_history") or []):
        if entry.get("key") == key:
            return entry
    return None


def numbered_list(state: dict[str, Any]) -> list[dict[str, Any]]:
    keys = state.get("current_list") or state.get("last_results") or []
    entries = [history_lookup(state, key) for key in keys]
    return [entry for entry in entries if entry]


def latest_page(state: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [history_lookup(state, key) for key in state.get("last_results") or []]
    return [entry for entry in entries if entry]


def resolve_refs(state: dict[str, Any], intent: TurnIntent) -> list[dict[str, Any]]:
    """Resolve references deterministically - never by the LLM.

    Ordinals ("the second one", "the first three") use the numbering shown to
    the user for the current search, including "show me more" pages. "these"
    means the page shown last. "that one" / "it" is the result discussed last
    (focus); "the previous one" the one before it.
    """
    numbered = numbered_list(state)
    last = latest_page(state)
    mode = intent.ref_mode
    if mode in {"ordinal", "last"}:
        resolved = []
        for index in intent.refs:
            position = index if index >= 0 else len(numbered) + index
            if 0 <= position < len(numbered):
                resolved.append(numbered[position])
            else:
                return []
        return resolved
    if mode == "focus":
        entry = history_lookup(state, state.get("focus"))
        return [entry] if entry else last[:1]
    if mode == "previous":
        entry = history_lookup(state, state.get("previous_focus")) or history_lookup(
            state, state.get("focus")
        )
        return [entry] if entry else []
    # "these", "them", or nothing specific: the whole latest list.
    return last


def set_focus(state: dict[str, Any], key: str) -> None:
    if state.get("focus") != key:
        state["previous_focus"] = state.get("focus")
        state["focus"] = key
