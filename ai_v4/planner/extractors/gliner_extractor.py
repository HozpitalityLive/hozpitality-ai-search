from gliner import GLiNER

from ai_v4.config.logger import logger


class GLiNERExtractor:

    def __init__(self):

        logger.info("Loading GLiNER Model...")

        self.model = GLiNER.from_pretrained(
            "urchade/gliner_medium-v2.1"
        )

        self.labels = [

            "person",

            "city",
            "country",
            "location",

            "organization",
            "company",
            "hotel",
            "restaurant",

            "skill",
            "technology",

            "job title",
            "department",
            "experience",
            "salary",

            "award",
            "event",
            "article",
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

        logger.info("=" * 80)
        logger.info("RAW GLINER ENTITIES")
        logger.info(predictions)
        logger.info("=" * 80)

        entities = {
            "query": query,

            "person_names": [],
            "companies": [],
            "locations": [],
            "skills": [],
            "technologies": [],
            "job_titles": [],
            "departments": [],
            "experience": [],
            "salary": [],
            "awards": [],
            "events": [],
            "articles": [],

            "raw_entities": predictions
        }

        for entity in predictions:

            label = entity["label"].lower().strip()
            text = entity["text"].strip()

            if label == "person":
                entities["person_names"].append(text)

            elif label in ["organization", "company", "hotel", "restaurant"]:
                entities["companies"].append(text)

            elif label in ["city", "country", "location"]:
                entities["locations"].append(text)

            elif label == "skill":
                entities["skills"].append(text)

            elif label == "technology":
                entities["technologies"].append(text)

            elif label == "job title":
                entities["job_titles"].append(text)

            elif label == "department":
                entities["departments"].append(text)

            elif label == "experience":
                entities["experience"].append(text)

            elif label == "salary":
                entities["salary"].append(text)

            elif label == "award":
                entities["awards"].append(text)

            elif label == "event":
                entities["events"].append(text)

            elif label == "article":
                entities["articles"].append(text)

        logger.info("=" * 80)
        logger.info("NORMALIZED ENTITIES")
        logger.info(entities)
        logger.info("=" * 80)

        return entities