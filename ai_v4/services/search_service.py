from ai_v4.search.hybrid_search import HybridSearch
from ai_v4.search.postgres_search import PostgresSearch
class SearchService:
    def __init__(self): self.hybrid=HybridSearch();self.pg=PostgresSearch()
    async def search(self,query,filters=None,limit=20): return await self.hybrid.search(query,filters,limit)
    async def analytics(self,query,filters=None): return await self.pg.analytics(query,filters or {})
