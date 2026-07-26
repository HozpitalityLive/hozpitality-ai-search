import re

from .base import BaseParser


class KeywordParser(BaseParser):

    STOP_WORDS = {
        "find",
        "search",
        "show",
        "get",
        "list",
        "profile",
        "profiles",
        "professional",
        "professionals",
        "job",
        "jobs",
        "company",
        "companies",
        "hotel",
        "hotels",
        "restaurant",
        "restaurants",
        "event",
        "events",
        "award",
        "awards",
        "article",
        "articles",
        "product",
        "products",
        "of",
        "in",
        "for",
        "with",
        "near",
        "at",
        "the",
        "a",
        "an",
    }

    async def parse(
        self,
        query,
        entities,
        filters,
    ):

        cleaned = query

        for word in self.STOP_WORDS:

            cleaned = re.sub(
                rf"\b{re.escape(word)}\b",
                " ",
                cleaned,
                flags=re.IGNORECASE
            )

        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return {
            "keyword": cleaned or query
        }