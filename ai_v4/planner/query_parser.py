from ai_v4.config.logger import logger

from ai_v4.planner.parsers import (
    KeywordParser,
    PersonParser,
    CompanyParser,
    LocationParser,
    JobParser,
    SkillParser,
    SalaryParser,
    ExperienceParser,
    DateParser,
    SortParser,
    # CategoryParser
)
from ai_v4.planner.utils import deep_merge


class QueryParser:

    def __init__(self):

        self.parsers = [
            KeywordParser(),
            PersonParser(),
            CompanyParser(),
            LocationParser(),
            JobParser(),
            SkillParser(),
            SalaryParser(),
            ExperienceParser(),
            DateParser(),
            SortParser(),
            # CategoryParser(),
        ]

    async def parse(
        self,
        query: str,
        intent,
        entities: dict,
    ) -> dict:

        logger.info("[2/4] Parsing Query...")

        filters = {
            "keyword": "",

            "person": {
                "name": None
            },

            "company": {
                "name": None
            },

            "location": {
                "city": None,
                "country": None
            },

            "job": {
                "title": None,
                "department": None
            },

            "salary": {
                "min": None,
                "max": None,
                "currency": None
            },

            "experience": {
                "min": None,
                "max": None,
                "unit": "years"
            },

            "date": {
                "from": None,
                "to": None,
            },

            "time_scope": None,

            "sort": {
                "field": None,
                "order": None
            },

            "category": None,

            "employment": {
                "type": None
            },

            "education": [],

            "certifications": [],

            "languages": [],

            "nationalities": [],

            "visa": {
                "type": None
            },

            "skills": [],

            "technologies": []
        }

        for parser in self.parsers:

            result = await parser.parse(
                query=query,
                entities=entities,
                filters=filters,
            )

            if result:
                filters = deep_merge(
                    filters,
                    result
                )

        logger.info("=" * 80)
        logger.info("FILTERS")
        logger.info(filters)
        logger.info("=" * 80)

        return filters