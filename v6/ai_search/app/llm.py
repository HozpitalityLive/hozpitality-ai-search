from __future__ import annotations

import json
from urllib import error as urlerror
from urllib import request as urlrequest

from .config import settings
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
        self.timeout = settings.ollama_timeout_seconds

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

            filters = parsed.get("filters")
            if isinstance(filters, dict):
                # LLM filters supplement deterministic filters. They do not
                # overwrite values already extracted by deterministic rules.
                for key, value in filters.items():
                    if key not in base.filters and value is not None:
                        base.filters[key] = value

            intent = parsed.get("intent")
            if isinstance(intent, str) and intent.strip():
                base.intent = intent.strip()

            base.confidence = max(base.confidence, 0.82)

        except (urlerror.URLError, TimeoutError, OSError, ValueError, TypeError) as exc:
            print(f"Ollama query understanding skipped: {exc}")
        except Exception as exc:
            # Never make Ollama availability a dependency for MongoDB search.
            print(f"Ollama query understanding skipped: {exc}")

        return base

