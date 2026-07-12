from ai_v4.planner.extractors.gliner_extractor import GLiNERExtractor


class EntityExtractor:

    def __init__(self):

        self.extractor = GLiNERExtractor()

    async def extract(
        self,
        query: str
    ):

        entities = await self.extractor.extract(query)

        return entities