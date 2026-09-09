import asyncio
from ai_v4.search.elastic_search import ElasticSearch
from ai_v4.search.postgres_search import PostgresSearch
from ai_v4.search.vector_search import VectorSearch
from ai_v4.search.merger import SearchMerger
from ai_v4.search.reranker import SearchReranker
class HybridSearch:
    def __init__(self):
        self.elastic=ElasticSearch();self.postgres=PostgresSearch();self.vector=VectorSearch()
        self.merger=SearchMerger();self.reranker=SearchReranker()
    async def search(self,query,filters=None,limit=20):
        tasks=[self.postgres.search(query,filters,limit)]
        from ai_v4.config.settings import settings
        if settings.USE_ELASTIC: tasks.append(self.elastic.search(query,filters,limit))
        if settings.USE_VECTOR: tasks.append(self.vector.search(query,filters,limit))
        groups=await asyncio.gather(*tasks,return_exceptions=True)
        groups=[g for g in groups if isinstance(g,list)]
        return self.reranker.rerank(query,self.merger.merge(*groups))[:limit]
