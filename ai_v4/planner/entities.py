from ai_v4.planner.extractors.semantic_extractor import (
    SemanticExtractor,
)


class EntityExtractor:

    def __init__(self):

        self.extractor = SemanticExtractor()

    async def extract(
        self,
        query: str,
    ):

        return await self.extractor.extract(query)