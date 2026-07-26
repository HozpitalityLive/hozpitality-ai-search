from .base import BaseParser


class SkillParser(BaseParser):

    async def parse(
        self,
        query,
        entities,
        filters,
    ):

        if entities.get("skills"):

            return {
                "skills": entities["skills"]
            }

        return {}