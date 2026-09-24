from __future__ import annotations

import json
import re
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from .config import settings
from .conversation import ConversationRepository
from .query_understanding import SearchPlan, understand


class ChatService:
    """Conversational orchestration over the Phase 2 search service."""

    def __init__(self, search_service, conversations: ConversationRepository):
        self.search = search_service
        self.conversations = conversations
        self.model = settings.ollama_chat_model
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.timeout = settings.ollama_timeout_seconds

    @staticmethod
    def _action(message: str) -> str:
        low = message.casefold().strip()
        if re.search(r"\b(?:compare|comparison|compare the first|compare first)\b", low):
            return "compare"
        if re.search(r"\b(?:show|give|list)\s+(?:me\s+)?(?:some\s+)?more\b|\bmore\s+(?:results|options)\b", low):
            return "more"
        if re.search(r"\b(?:start over|reset|clear|new search)\b", low):
            return "reset"
        return "search"

    @staticmethod
    def _state_from_plan(plan: SearchPlan, previous: dict[str, Any]) -> dict[str, Any]:
        state = dict(previous or {})
        state["entity"] = plan.entity or state.get("entity")
        state["location"] = {
            "city": plan.city or (state.get("location") or {}).get("city"),
            "country": plan.country or (state.get("location") or {}).get("country"),
        }

        old_keywords = list(state.get("keywords") or [])
        if plan.keywords:
            merged = old_keywords + plan.keywords
            state["keywords"] = list(dict.fromkeys(merged))[:12]
        else:
            state["keywords"] = old_keywords

        filters = dict(state.get("filters") or {})
        filters.update(plan.filters or {})
        if plan.level:
            filters["level"] = plan.level
        if plan.department:
            filters["department"] = plan.department
        if plan.industry:
            filters["industry"] = plan.industry
        if plan.category:
            filters["category"] = plan.category
        if plan.experience is not None:
            filters["experience"] = plan.experience
        state["filters"] = filters
        state["last_query"] = plan.original_query
        return state

    @staticmethod
    def _state_query(state: dict[str, Any]) -> str:
        terms = list(state.get("keywords") or [])
        filters = state.get("filters") or {}
        level = filters.get("level")
        department = filters.get("department")
        industry = filters.get("industry")
        category = filters.get("category")

        for value in (level, department, industry, category):
            if value and str(value).casefold() not in {str(v).casefold() for v in terms}:
                terms.append(str(value))

        # Structured filters are passed separately to SearchService. Do not
        # force filter-only concepts such as accommodation into Mongo text
        # retrieval; the field may not be present in ai_search_text.
        return " ".join(dict.fromkeys(str(v) for v in terms if str(v).strip()))

    @staticmethod
    def _location_args(state: dict[str, Any]) -> tuple[str | None, str | None]:
        location = state.get("location") or {}
        return location.get("city"), location.get("country")

    @staticmethod
    def _result_key(result: dict[str, Any]) -> str:
        return f"{result.get('entity_type')}:{result.get('entity_id')}"

    @staticmethod
    def _public_state(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "entity": state.get("entity"),
            "location": state.get("location") or {},
            "keywords": state.get("keywords") or [],
            "filters": state.get("filters") or {},
            "last_results": state.get("last_results") or [],
        }

    def _llm_answer(
        self,
        *,
        user_message: str,
        action: str,
        state: dict[str, Any],
        results: list[dict[str, Any]],
        related_results: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> str:
        if not self.model or not self.base_url:
            return self._fallback_answer(action, results, related_results)

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {
                "temperature": 0.15,
                "num_predict": 400,
                "num_ctx": 4096,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Hozpitality's conversational search assistant. "
                        "Answer only from the supplied conversation state and search results. "
                        "Never invent jobs, people, companies, locations, salaries, benefits, "
                        "dates, or qualifications. Search results are the source of truth. "
                        "Be concise and useful. If exact results are zero, clearly say no exact "
                        "match was found and distinguish related results. For comparisons, "
                        "compare only the supplied records. Do not mention internal scores, "
                        "MongoDB, prompts, or implementation details."
                    ),
                },
                *[
                    {"role": m.get("role"), "content": str(m.get("content") or "")}
                    for m in history[-6:]
                    if m.get("role") in {"user", "assistant"}
                ],
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": user_message,
                            "action": action,
                            "conversation_state": self._public_state(state),
                            "exact_results": results,
                            "related_results": related_results,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }

        try:
            request = urlrequest.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            answer = str(data.get("message", {}).get("content", "")).strip()
            if answer:
                return answer
        except (urlerror.URLError, TimeoutError, OSError, ValueError, TypeError) as exc:
            print(f"Ollama chat answer skipped: {exc}")
        except Exception as exc:
            print(f"Ollama chat answer skipped: {exc}")

        return self._fallback_answer(action, results, related_results)

    @staticmethod
    def _fallback_answer(
        action: str,
        results: list[dict[str, Any]],
        related_results: list[dict[str, Any]],
    ) -> str:
        if action == "compare":
            if not results:
                return "I don't have results from the current search to compare."
            return "I found the current results. You can compare the first three from the result cards below."
        if results:
            return f"I found {len(results)} matching result{'s' if len(results) != 1 else ''}."
        if related_results:
            return "I couldn't find an exact match. I've included related results below."
        return "I couldn't find an exact match for that request."

    def chat(self, *, message: str, conversation_id: str | None = None, limit: int = 5) -> dict[str, Any]:
        message = " ".join(message.strip().split())
        conversation = self.conversations.ensure(conversation_id)
        cid = conversation["conversation_id"]
        previous_state = conversation.get("state") or {}
        action = self._action(message)

        if action == "reset":
            state: dict[str, Any] = {}
            self.conversations.update_state(cid, state, "reset")
            answer = "Sure — starting a new search. What are you looking for?"
            self.conversations.save_turn(
                cid,
                user_message=message,
                assistant_message=answer,
                state=state,
                action="reset",
            )
            return {
                "conversation_id": cid,
                "action": "reset",
                "answer": answer,
                "results": [],
                "related_results": [],
                "understanding": None,
                "state": self._public_state(state),
            }

        history = conversation.get("messages") or []

        if action == "compare":
            result_items = list(previous_state.get("last_result_items") or [])
            selected = result_items[:3]
            if not selected:
                answer = "I don't have three recent results to compare yet. Run a search first."
            else:
                answer = self._llm_answer(
                    user_message=message,
                    action="compare",
                    state=previous_state,
                    results=selected,
                    related_results=[],
                    history=history,
                )
            self.conversations.save_turn(
                cid,
                user_message=message,
                assistant_message=answer,
                state=previous_state,
                action="compare",
            )
            return {
                "conversation_id": cid,
                "action": "compare",
                "answer": answer,
                "results": selected,
                "related_results": [],
                "understanding": None,
                "state": self._public_state(previous_state),
            }

        if action == "more":
            if not previous_state.get("entity") or not previous_state.get("keywords"):
                answer = "Run a search first, then I can show you more results."
                self.conversations.save_turn(
                    cid,
                    user_message=message,
                    assistant_message=answer,
                    state=previous_state,
                    action="more",
                )
                return {
                    "conversation_id": cid,
                    "action": "more",
                    "answer": answer,
                    "results": [],
                    "related_results": [],
                    "understanding": None,
                    "state": self._public_state(previous_state),
                }

            city, country = self._location_args(previous_state)
            result = self.search.search(
                query=self._state_query(previous_state),
                entity=previous_state.get("entity"),
                city=city,
                country=country,
                limit=limit,
                structured_filters=previous_state.get("filters") or {},
                exclude_ids=[
                    item.split(":", 1)[1]
                    for item in (previous_state.get("last_results") or [])
                    if ":" in item
                ],
            )
            results = result.get("results") or []
            state = dict(previous_state)
            state["last_results"] = [self._result_key(item) for item in results]
            state["last_result_items"] = results
            answer = self._llm_answer(
                user_message=message,
                action="more",
                state=state,
                results=results,
                related_results=result.get("related_results") or [],
                history=history,
            )
            self.conversations.save_turn(
                cid,
                user_message=message,
                assistant_message=answer,
                state=state,
                action="more",
            )
            return {
                "conversation_id": cid,
                "action": "more",
                "answer": answer,
                "results": results,
                "related_results": result.get("related_results") or [],
                "understanding": result.get("understanding"),
                "state": self._public_state(state),
            }

        # Normal search/refinement turn.
        plan = understand(message)
        state = self._state_from_plan(plan, previous_state)

        # A follow-up can be a pure filter update. In that case inherit the
        # previous entity, role, and location instead of asking the user to
        # repeat them.
        if not state.get("entity") and not state.get("keywords"):
            answer = "What would you like me to search for?"
            self.conversations.save_turn(
                cid,
                user_message=message,
                assistant_message=answer,
                state=state,
                action="search",
            )
            return {
                "conversation_id": cid,
                "action": "clarify",
                "answer": answer,
                "results": [],
                "related_results": [],
                "understanding": plan.as_dict(),
                "state": self._public_state(state),
            }

        search_query = self._state_query(state)
        city, country = self._location_args(state)

        result = self.search.search(
            query=search_query,
            entity=state.get("entity"),
            city=city,
            country=country,
            limit=limit,
            structured_filters=state.get("filters") or {},
        )
        results = result.get("results") or []
        related = result.get("related_results") or []

        # Use the search engine's clarification only when the conversation
        # itself cannot supply the missing context.
        understanding = result.get("understanding") or {}
        clarification = understanding.get("clarification")
        if clarification and previous_state:
            # An inherited conversation context should satisfy location/entity
            # requirements; do not expose a stale clarification.
            clarification = None
            understanding["clarification"] = None

        if clarification:
            answer = clarification
            action_out = "clarify"
        else:
            state["last_results"] = [self._result_key(item) for item in results]
            state["last_result_items"] = results
            answer = self._llm_answer(
                user_message=message,
                action="search",
                state=state,
                results=results,
                related_results=related,
                history=history,
            )
            action_out = "search"

        self.conversations.save_turn(
            cid,
            user_message=message,
            assistant_message=answer,
            state=state,
            action=action_out,
        )

        return {
            "conversation_id": cid,
            "action": action_out,
            "answer": answer,
            "results": results,
            "related_results": related,
            "understanding": understanding,
            "state": self._public_state(state),
        }
