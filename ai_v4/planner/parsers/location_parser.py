from .base import BaseParser


class LocationParser(BaseParser):

    async def parse(
        self,
        query,
        entities,
        filters,
    ):

        if entities.get("locations"):

            return {
                "location": entities["locations"][0]
            }

        return {}