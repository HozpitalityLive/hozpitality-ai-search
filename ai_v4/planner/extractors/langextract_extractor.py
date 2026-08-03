from ai_v4.config.settings import settings
import langextract as lx

from ai_v4.config.logger import logger
from ai_v4.planner.extractors.base import BaseExtractor
from ai_v4.planner.extractors.examples import EXAMPLES
from ai_v4.planner.extractors.schema import SEARCH_SCHEMA
from langextract.factory import ModelConfig
from langextract.factory import create_model


class LangExtractExtractor(BaseExtractor):

    def __init__(self):

        logger.info("=" * 80)
        logger.info("Initializing LangExtract...")
        logger.info("=" * 80)
        self.base_url = settings.OLLAMA_URL
        self.model = create_model(
            ModelConfig(
                model_id="llama3-hoz",
                provider="ollama",
                provider_kwargs={
                    "model_url": settings.OLLAMA_URL,
                },
            ),
            examples=EXAMPLES,
            use_schema_constraints=False,
        )

        self.prompt = """
You are Hozpitality AI's semantic extraction engine.

Extract ONLY entities explicitly mentioned by the user.

Do not infer missing information.

Use only the provided schema.

If an entity is absent, do not create one.

Return structured extractions only.
"""

    async def extract(
        self,
        query: str,
    ) -> dict:

        logger.info("=" * 80)
        logger.info("LANGEXTRACT QUERY")
        logger.info(query)
        logger.info("=" * 80)

        try:

            

            result = lx.extract(
                text_or_documents=query,
                prompt_description=self.prompt,
                examples=EXAMPLES,
                attributes=SEARCH_SCHEMA,
                model=self.model,
                fence_output=False,
                use_schema_constraints=False,
            )

            logger.info("=" * 80)
            logger.info("RAW LANGEXTRACT RESULT")
            logger.info(result)
            logger.info("=" * 80)

            return self.normalize(
                query=query,
                result=result,
            )

        except Exception:

            logger.exception("LangExtract Failed")

            return self.empty(query)

    def normalize(
        self,
        query,
        result,
    ):

        entities = self.empty(query)

        raw = []

        try:

            extractions = getattr(result, "extractions", [])

            for extraction in extractions:
                label = getattr(
                    extraction,
                    "extraction_class",
                    ""
                ).lower().strip()

                value = getattr(
                    extraction,
                    "extraction_text",
                    ""
                ).strip()

                if not label or not value:
                    continue

                raw.append({
                    "label": label,
                    "text": value,
                    "confidence": getattr(
                        extraction,
                        "confidence",
                        None,
                    )
                })

                if label == "person_name":
                    entities["person_names"].append(value)

                elif label == "company":
                    entities["companies"].append(value)

                elif label == "location":
                    entities["locations"].append(value)

                elif label == "job_title":
                    entities["job_titles"].append(value)

                elif label == "department":
                    entities["departments"].append(value)

                elif label == "skill":
                    entities["skills"].append(value)

                elif label == "technology":
                    entities["technologies"].append(value)

                elif label == "salary":
                    entities["salary"].append(value)

                elif label == "experience":
                    entities["experience"].append(value)

                elif label == "award":
                    entities["awards"].append(value)

                elif label == "event":
                    entities["events"].append(value)

                elif label == "article":
                    entities["articles"].append(value)

                elif label == "language":
                    entities.setdefault(
                        "languages",
                        []
                    ).append(value)

                elif label == "nationality":
                    entities.setdefault(
                        "nationalities",
                        []
                    ).append(value)

                elif label == "employment_type":
                    entities.setdefault(
                        "employment_types",
                        []
                    ).append(value)

                elif label == "visa":
                    entities.setdefault(
                        "visas",
                        []
                    ).append(value)

            entities["raw_entities"] = raw

            logger.info("=" * 80)
            logger.info("NORMALIZED ENTITIES")
            logger.info(entities)
            logger.info("=" * 80)

            return entities

        except Exception:

            logger.exception("Normalization Failed")

            return entities

    def empty(
        self,
        query,
    ):

        return {

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
            "languages": [],
            "nationalities": [],
            "employment_types": [],
            "visas": [],

            "raw_entities": []

        }