import re

from .base import BaseParser


class PersonParser(BaseParser):

    async def parse(
        self,
        query,
        entities,
        filters,
    ):

        if entities.get("person_names"):

            return {
                "name": entities["person_names"][0]
            }

        match = re.findall(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",
            query
        )

        if match:

            return {
                "name": match[0]
            }

        return {}