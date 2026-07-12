from ai_v4.search.hybrid_search import HybridSearch


class SearchService:

    def __init__(self):
        self.hybrid = HybridSearch()

    async def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int = 20
    ):

        return await self.hybrid.search(
            query=query,
            filters=filters,
            limit=limit
        )