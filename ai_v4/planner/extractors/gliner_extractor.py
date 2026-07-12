from gliner import GLiNER

from ai_v4.config.logger import logger

class GLiNERExtractor:

    def __init__(self):

        logger.info("Loading GLiNER Model...")

        self.model = GLiNER.from_pretrained(
            "urchade/gliner_medium-v2.1"
        )

        self.labels = [

            "city",
            "country",
            "location",

            "company",
            "hotel",
            "restaurant",

            "skill",
            "technology",

            "job role",
            "department",

            "salary",
            "experience",

            "award",
            "event",
            "article"

        ]

        logger.info("GLiNER Loaded Successfully")

    async def extract(
        self,
        query: str
    ) -> dict:

        predictions = self.model.predict_entities(
            query,
            self.labels
        )

        return {
            "query": query,
            "raw_entities": predictions
        }