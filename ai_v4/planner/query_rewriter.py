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
You are an AI Query Rewriter.

Your ONLY task is to combine the user's original request with their latest clarification answer into ONE complete search query.

IMPORTANT RULES

1. NEVER invent information.
2. NEVER assume missing values.
3. NEVER use values from the examples.
4. ONLY use information explicitly provided in:
   - Original Query
   - Clarification Answer
5. Preserve the user's intent.
6. Remove duplicate words.
7. Make the query natural and concise.
8. Do NOT add locations, job titles, company names, dates, skills, experience, or any other information unless the user explicitly provided them.
9. Return ONLY valid JSON.
10. Do NOT explain your reasoning.

Intent:
{intent}

Original Query:
{original_query}

Clarification Answer:
{clarification_answer}

Previously Extracted Entities:
{json.dumps(entities or {}, indent=2)}

Examples

Example 1

Original Query:
Find a job

Clarification Answer:
Waiter

Output:
{{
    "query": "Find waiter jobs"
}}

Example 2

Original Query:
Find waiter jobs

Clarification Answer:
Dubai

Output:
{{
    "query": "Find waiter jobs in Dubai"
}}

Example 3

Original Query:
Show hotels

Clarification Answer:
Abu Dhabi

Output:
{{
    "query": "Show hotels in Abu Dhabi"
}}

Now rewrite ONLY the following request.

Original Query:
{original_query}

Clarification Answer:
{clarification_answer}

Remember:

- Do NOT invent any information.
- Do NOT use words from the examples.
- ONLY use information from the Original Query and Clarification Answer.

Return ONLY:

{{
    "query": "<rewritten query>"
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