from .base import BaseParser


class JobParser(BaseParser):

    async def parse(
        self,
        query,
        entities,
        filters,
    ):

        if entities.get("job_titles"):
            return {
                "job_title": entities["job_titles"][0]
            }

        return {}