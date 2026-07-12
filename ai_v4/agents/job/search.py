from ai_v4.services.search_service import SearchService


class JobSearch:

    def __init__(self):
        self.search = SearchService()

    async def search_jobs(
        self,
        query,
        filters
    ):

        return await self.search.search(
            query=query,
            filters=filters
        )