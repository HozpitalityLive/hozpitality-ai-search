from ai_v4.agents.base import BaseAgent



class ArticleAgent(BaseAgent):

    def __init__(self):
        super().__init__("article")

    async def execute(
        self,
        query: str,
        plan: dict,
        memory: dict
    ):

        filters = {
            "category": "article"
        }

        results = await self.search_service.search(
            query=query,
            filters=filters
        )

        return {
            "agent": self.name,
            "results": results
        }