from __future__ import annotations

import json
import os
import re
from urllib import request as urlrequest

from .query_understanding import SearchPlan


class GeminiQueryInterpreter:
    """Optional LLM fallback for queries the deterministic parser cannot resolve."""

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "4"))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def interpret(self, query: str, base: SearchPlan) -> SearchPlan:
        if not self.enabled:
            return base

        prompt = f"""
You are the query-understanding layer for Hozpitality search.
Return ONLY valid JSON. Do not invent facts or locations.
Allowed entity: job, professional, company, product, article, event, award, faq.
Extract only information explicitly stated or strongly implied.

Schema:
{{
  "intent":"search",
  "entity":null,
  "keywords":[],
  "city":null,
  "country":null,
  "experience":null,
  "level":null,
  "department":null,
  "industry":null,
  "category":null,
  "filters":{{}}
}}

Query: {query}
Existing deterministic parse: {json.dumps(base.as_dict(), ensure_ascii=False)}
"""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        try:
            req = urlrequest.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                return base
            for key in ("entity", "city", "country", "experience", "level", "department", "industry", "category"):
                if parsed.get(key) is not None:
                    setattr(base, key, parsed[key])
            if isinstance(parsed.get("keywords"), list):
                base.keywords = [str(v) for v in parsed["keywords"][:12] if str(v).strip()]
            if isinstance(parsed.get("filters"), dict):
                base.filters.update(parsed["filters"])
            base.confidence = max(base.confidence, 0.8)
        except Exception as exc:
            print(f"Gemini query understanding skipped: {exc}")
        return base
