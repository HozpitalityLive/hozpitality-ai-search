from .base import BaseParser


class CompanyParser(BaseParser):

    async def parse(
        self,
        query,
        entities,
        filters,
    ):

        if entities.get("companies"):
            return {
                "company": entities["companies"][0]
            }
        return {}