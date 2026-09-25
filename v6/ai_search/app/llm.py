from __future__ import annotations

import json
import logging
import re
import time
from urllib import error as urlerror
from urllib import request as urlrequest

from .config import settings
from .observability import add_timing, log_event, metrics
from .query_understanding import SearchPlan


class OllamaQueryInterpreter:
    """Local Ollama fallback for ambiguous natural-language search queries.

    Ollama is optional: deterministic parsing remains the first layer. When
    enabled, this class calls the local Ollama HTTP API and asks the model for
    a constrained JSON search plan. No external LLM API is used.
    """

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_query_model
        # Query understanding sits on the /search hot path, so it uses its own
        # short budget rather than the chat answer timeout.
        self.timeout = getattr(
            settings, "ollama_query_timeout_seconds", settings.ollama_timeout_seconds
        )

    @property
    def enabled(self) -> bool:
        return bool(self.model and self.base_url)

    @staticmethod
    def _schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "entity": {
                    "type": ["string", "null"],
                    "enum": [
                        "job", "professional", "company", "product",
                        "article", "event", "award", "faq", None,
                    ],
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "city": {"type": ["string", "null"]},
                "country": {"type": ["string", "null"]},
                "experience": {"type": ["integer", "null"]},
                "level": {"type": ["string", "null"]},
                "department": {"type": ["string", "null"]},
                "industry": {"type": ["string", "null"]},
                "category": {"type": ["string", "null"]},
                "filters": {"type": "object"},
            },
            "required": [
                "intent", "entity", "keywords", "city", "country",
                "experience", "level", "department", "industry",
                "category", "filters",
            ],
        }

    def _prompt(self, query: str, base: SearchPlan) -> str:
        return f"""
You are the natural-language query understanding layer for Hozpitality Search.

Your ONLY task is to convert the user's search request into a search plan.
Return JSON matching the supplied schema.

Rules:
- Do not answer the user.
- Do not invent a location, experience, company, date, or filter.
- Extract only information explicitly stated or strongly implied.
- entity must be exactly one of:
  job, professional, company, product, article, event, award, faq
- "chef jobs" means entity=job and keyword=chef.
- "find chefs" / "chefs in Dubai" can mean professional unless the query
  explicitly asks for jobs/vacancies/positions.
- Keep useful search concepts in keywords; remove conversational filler.
- Normalize common location names when obvious:
  Dubai -> Dubai, United Arab Emirates
  UAE -> United Arab Emirates
- Experience must be a number of years.
- level should be a concise normalized value such as senior, junior, mid,
  manager, executive, or intern.
- filters may contain structured values such as salary_min, salary_currency,
  employment_type, verified, featured, currently_working.
- If something is not present, return null or an empty array/object.

User query:
{query}

Deterministic parser result:
{json.dumps(base.as_dict(), ensure_ascii=False)}
""".strip()

    @staticmethod
    def _extract_json(value: str) -> dict | None:
        value = (value or "").strip()
        if not value:
            return None

        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        # Defensive fallback for models that wrap JSON in markdown.
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(value[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def interpret(self, query: str, base: SearchPlan) -> SearchPlan:
        if not self.enabled:
            return base

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "format": self._schema(),
            "options": {
                "temperature": 0,
                "num_predict": 300,
                "num_ctx": 4096,
            },
            "messages": [
                {
                    "role": "system",
                    "content": self._prompt(query, base),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        }

        endpoint = f"{self.base_url}/api/chat"
        started = time.perf_counter()

        try:
            req = urlrequest.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))

            text = (
                data.get("message", {}).get("content", "")
                if isinstance(data, dict)
                else ""
            )
            parsed = self._extract_json(text)
            if not parsed:
                return base

            allowed_entities = {
                "job", "professional", "company", "product",
                "article", "event", "award", "faq",
            }

            entity = parsed.get("entity")
            if entity in allowed_entities:
                base.entity = entity

            for key in (
                "city", "country", "level", "department",
                "industry", "category",
            ):
                value = parsed.get(key)
                if value is not None and str(value).strip():
                    setattr(base, key, str(value).strip())

            experience = parsed.get("experience")
            if isinstance(experience, int) and 0 <= experience <= 60:
                base.experience = experience

            keywords = parsed.get("keywords")
            if isinstance(keywords, list):
                cleaned = [
                    str(value).strip()
                    for value in keywords[:12]
                    if str(value).strip()
                ]
                if cleaned:
                    base.keywords = cleaned

            # Never allow the LLM to invent filters. Deterministic parsing is
            # authoritative for filters; model output is accepted only for
            # fields that are explicitly represented in the user query.
            # In particular, "Find me a job" must produce filters={}.
            filters = parsed.get("filters")
            if isinstance(filters, dict):
                query_low = query.casefold()
                allowed_markers = {
                    "salary_min": r"\b(?:salary|pay|paying|compensation)\b",
                    "salary_currency": r"\b(?:aed|usd|inr|gbp|eur|sar|qar)\b",
                    "employment_type": r"\b(?:full[- ]time|part[- ]time|contract|temporary|remote)\b",
                    "verified": r"\bverified\b",
                    "featured": r"\bfeatured\b",
                    "currently_working": r"\b(?:currently working|working professionals?)\b",
                }
                for key, value in filters.items():
                    marker = allowed_markers.get(key)
                    if (
                        key not in base.filters
                        and value is not None
                        and marker
                        and re.search(marker, query_low)
                    ):
                        base.filters[key] = value

            intent = parsed.get("intent")
            if isinstance(intent, str) and intent.strip():
                base.intent = intent.strip()

            base.confidence = max(base.confidence, 0.82)
            metrics.incr("llm_query_success")

        except Exception as exc:
            # Never make Ollama availability a dependency for MongoDB search,
            # but never hide the failure either.
            timed_out = isinstance(exc, TimeoutError) or "timed out" in str(exc).casefold()
            metrics.incr("llm_query_timeout" if timed_out else "llm_query_error")
            metrics.incr("llm_query_fallback")
            log_event(
                "llm_query_fallback",
                level=logging.WARNING,
                reason="timeout" if timed_out else type(exc).__name__,
                timeout_s=self.timeout,
            )
        finally:
            add_timing("llm_query_ms", (time.perf_counter() - started) * 1000)

        return base

