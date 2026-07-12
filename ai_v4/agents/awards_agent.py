from ai_v4.agents.base import BaseAgent



class AwardsAgent(BaseAgent):

    def __init__(self):
        super().__init__("awards")

    async def execute(
        self,
        query: str,
        plan: dict,
        memory: dict
    ):

        filters = {
            "category": "awards"
        }

        results = await self.search_service.search(
            query=query,
            filters=filters
        )

        return {
            "agent": self.name,
            "results": results
        }