import json

from ai_v4.config.logger import logger
from ai_v4.llm.ollama import OllamaClient


class ClarificationDetector:

    def __init__(self):
        self.ollama = OllamaClient()

    async def analyze(
        self,
        query: str,
        intent,
        entities: dict
    ) -> dict:

        prompt = f"""
You are the planning engine for Hozpitality AI.

Your job is NOT to answer the user.

Your task is to determine whether the user's request contains enough
information to perform an accurate search.

Intent:
{intent}

User Query:
{query}

Extracted Entities:
{json.dumps(entities, indent=2)}

Rules:

1. If enough information exists to perform the search:

Return ONLY

{{
    "required": false,
    "question": null,
    "missing": [],
    "reason": null,
    "rephrased_query": "{query}"
}}

2. If important information is missing:

Return ONLY

{{
    "required": true,
    "question": "Ask ONE natural follow-up question.",
    "missing": [],
    "reason": "Explain briefly why clarification is needed.",
    "rephrased_query": null
}}

Do NOT answer the user's request.

Return JSON ONLY.
"""

        try:

            result = await self.ollama.generate(
                prompt=prompt
            )

            response = result.get("response", "").strip()

            start = response.find("{")
            end = response.rfind("}")

            if start == -1 or end == -1:
                raise ValueError("No JSON")

            response = response[start:end + 1]

            clarification = json.loads(response)

            logger.info("Clarification Decision")
            logger.info(clarification)

            return clarification

        except Exception:

            logger.exception("Clarification Detection Failed")

            return {
                "required": False,
                "question": None,
                "missing": [],
                "reason": None,
                "rephrased_query": query
            }