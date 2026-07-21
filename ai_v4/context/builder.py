from collections import Counter

from ai_v4.context.formatter import ContextFormatter
from ai_v4.config.logger import logger


class ContextBuilder:

    def __init__(self):
        self.formatter = ContextFormatter()

    async def build(
        self,
        query: str,
        search_results: list,
        memory: dict | None = None
    ):

        logger.info("[3/4] Building context...")
        logger.info("Building Context")

        categories = []
        locations = []
        companies = []

        for item in search_results:

            doc = item.get("document", {})

            category = (
                doc.get("category")
                or doc.get("category_text")
            )

            location = (
                doc.get("location")
                or doc.get("location_text")
            )

            company = (
                doc.get("user_name")
                or doc.get("company")
            )

            if category:
                categories.append(category)

            if location:
                locations.append(location)

            if company:
                companies.append(company)

        summary = {
            "query": query,
            "total": len(search_results),
            "categories": list(dict.fromkeys(categories))[:5],
            "top_locations": list(dict.fromkeys(locations))[:5],
            "top_companies": list(dict.fromkeys(companies))[:5]
        }

        logger.info("=" * 80)
        logger.info("LLM SUMMARY")
        logger.info(summary)
        logger.info("=" * 80)

        return {
            "query": query,
            "memory": memory,
            "summary": summary,
            "documents": self.formatter.format_documents(
                search_results[:5]
            ),
            "results": search_results,
            "total": len(search_results)
        }