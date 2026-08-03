from ai_v4.planner.extractors.langextract_extractor import LangExtractExtractor
from ai_v4.planner.extractors.gliner_extractor import GLiNERExtractor


class SemanticExtractor:

    def __init__(self):

        self.langextract = LangExtractExtractor()
        self.gliner = GLiNERExtractor()

    async def extract(
        self,
        query: str,
    ):

        entities = await self.langextract.extract(query)

        if not entities["person_names"]:
            fallback = await self.gliner.extract(query)

            for key, value in fallback.items():
                if key == "query":
                    continue

                if isinstance(value, list):
                    entities.setdefault(key, [])
                    for item in value:
                        if item not in entities[key]:
                            entities[key].append(item)

        return entities