import re

from .base import BaseParser


class ExperienceParser(BaseParser):

    async def parse(
        self,
        query,
        entities,
        filters,
    ):

        if entities.get("experience"):
            return {
                "experience": entities["experience"][0]
            }

        match = re.search(
            r"(\d+)\+?\s*(year|years)",
            query,
            re.IGNORECASE
        )

        if not match:
            return {}

        return {
            "experience": {
                "min": int(match.group(1)),
                "unit": "years"
            }
        }