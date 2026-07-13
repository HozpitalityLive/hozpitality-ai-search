from ai_v4.agents.base import BaseAgent

from ai_v4.agents.job.filter_builder import JobFilterBuilder
from ai_v4.agents.job.search import JobSearch
from ai_v4.config.logger import logger
from pprint import pformat


class JobAgent(BaseAgent):

    def __init__(self):
        super().__init__("job")
        self.builder = JobFilterBuilder()
        self.search = JobSearch()

    async def execute(
        self,
        query,
        plan,
        memory
    ):

        filters = self.builder.build(
            plan
        )

        logger.info("=" * 80)
        logger.info("JOB FILTERS")
        logger.info(pformat(filters))
        logger.info("=" * 80)

        results = await self.search.search_jobs(
            query=query,
            filters=filters
        )

        return {
            "agent": self.name,
            "filters": filters,
            "results": results
        }