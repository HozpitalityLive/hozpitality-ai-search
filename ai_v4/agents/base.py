from abc import ABC
from pprint import pformat

from ai_v4.config.logger import logger
from ai_v4.services.search_service import SearchService


class BaseAgent(ABC):

    def __init__(
        self,
        name: str,
        builder
    ):
        self.name = name
        self.builder = builder
        self.search = SearchService()

    async def execute(
        self,
        query: str,
        plan: dict,
        memory: dict
    ):

        filters = self.builder.build(plan)

        logger.info("=" * 80)
        logger.info(f"{self.name.upper()} FILTERS")
        logger.info(pformat(filters))
        logger.info("=" * 80)

        results = await self.search.search(
            query=query,
            filters=filters
        )

        return {
            "agent": self.name,
            "query": query,
            "filters": filters,
            "total": len(results),
            "page": 1,
            "page_size": 5,
            "results": results
        }