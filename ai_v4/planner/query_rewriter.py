import json

from ai_v4.config.logger import logger
from ai_v4.llm.ollama import OllamaClient


class QueryRewriter:

    def __init__(self):
        self.ollama = OllamaClient()

    async def rewrite(
        self,
        original_query: str,
        clarification_answer: str,
        intent=None,
        entities=None
    ) -> str:

        prompt = f"""
You are an AI query rewriting engine.

Your job is to merge a user's original request with their clarification
answer into ONE complete search query.

Do NOT answer the user.

Do NOT explain anything.

Return ONLY valid JSON.

Intent:
{intent}

Original Query:
{original_query}

Clarification Answer:
{clarification_answer}

Previously Extracted Entities:
{json.dumps(entities or {}, indent=2)}

Examples

Original Query:
Find a job

Clarification:
Waiter

Output:
{{
    "query": "Find waiter jobs"
}}

Original Query:
Find waiter jobs

Clarification:
Dubai

Output:
{{
    "query": "Find waiter jobs in Dubai"
}}

Original Query:
Show hotels

Clarification:
Abu Dhabi

Output:
{{
    "query": "Show hotels in Abu Dhabi"
}}

Return JSON ONLY.

{{
    "query": ""
}}
"""

        try:

            result = await self.ollama.generate(
                prompt=prompt,
                model="phi3-hoz"
            )

            response = result.get("response", "").strip()

            if response.startswith("```json"):
                response = response.replace("```json", "", 1)

            if response.startswith("```"):
                response = response.replace("```", "", 1)

            if response.endswith("```"):
                response = response[:-3]

            response = response.strip()

            start = response.find("{")
            end = response.rfind("}")

            if start == -1 or end == -1:
                raise ValueError("No JSON object found.")

            response = response[start:end + 1]

            rewritten = json.loads(response)

            query = rewritten.get("query")

            if not query:
                raise ValueError("Missing query")

            logger.info("=" * 80)
            logger.info("QUERY REWRITER")
            logger.info(f"Original : {original_query}")
            logger.info(f"Answer   : {clarification_answer}")
            logger.info(f"Rewritten: {query}")
            logger.info("=" * 80)

            return query

        except Exception:

            logger.exception("Query Rewriter Failed")

            return f"{original_query} {clarification_answer}"