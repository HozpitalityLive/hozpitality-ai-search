import asyncio

from ai_v4.config.logger import logger

from ai_v4.search.elastic_search import ElasticSearch
from ai_v4.search.postgres_search import PostgresSearch
from ai_v4.search.merger import SearchMerger
from ai_v4.search.reranker import SearchReranker


class HybridSearch:

    def __init__(self):
        self.elastic = ElasticSearch()
        self.postgres = PostgresSearch()
        self.merger = SearchMerger()
        self.reranker = SearchReranker()


    async def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int = 20
    ):

        logger.info(f"Hybrid Search : {query}")

        elastic_task = self.elastic.search(
            query=query,
            filters=filters,
            size=limit
        )

        postgres_task = self.postgres.search(
            query=query,
            filters=filters,
            limit=limit
        )

        elastic_results, postgres_results = await asyncio.gather(
            elastic_task,
            postgres_task
        )

        merged = self.merger.merge(
            elastic_results,
            postgres_results
        )

        reranked = self.reranker.rerank(
            query=query,
            results=merged
        )

        return reranked[:limit]