import re

from .base import BaseParser


class SortParser(BaseParser):

    SORT_RULES = {

        "latest": {
            "field": "created_at",
            "order": "desc",
        },

        "recent": {
            "field": "created_at",
            "order": "desc",
        },

        "new": {
            "field": "created_at",
            "order": "desc",
        },

        "newest": {
            "field": "created_at",
            "order": "desc",
        },

        "oldest": {
            "field": "created_at",
            "order": "asc",
        },

        "highest salary": {
            "field": "salary",
            "order": "desc",
        },

        "lowest salary": {
            "field": "salary",
            "order": "asc",
        },

        "popular": {
            "field": "popularity",
            "order": "desc",
        },

        "most viewed": {
            "field": "views",
            "order": "desc",
        },

        "most applied": {
            "field": "applications",
            "order": "desc",
        },

        "top": {
            "field": "rating",
            "order": "desc",
        },

        "best": {
            "field": "rating",
            "order": "desc",
        },

        "a-z": {
            "field": "title",
            "order": "asc",
        },

        "z-a": {
            "field": "title",
            "order": "desc",
        },

        "nearest": {
            "field": "distance",
            "order": "asc",
        },

        "closest": {
            "field": "distance",
            "order": "asc",
        },
    }

    async def parse(
        self,
        query,
        entities,
        filters,
    ):

        query_lower = query.lower()

        for keyword, sort in self.SORT_RULES.items():

            if keyword in query_lower:

                return {
                    "sort": {
                        "field": sort["field"],
                        "order": sort["order"],
                    }
                }

        return {}