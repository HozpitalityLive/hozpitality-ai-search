import json

from ai_v4.config.elastic import es
from ai_v4.config.logger import logger


class ElasticSearch:

    def __init__(self):
        self.index = "hozpitality"

    async def search(
        self,
        query: str,
        filters: dict | None = None,
        size: int = 20
    ):

        logger.info(f"Elastic Search : {query}")

        filters = filters or {}

        roles = filters.get("roles", [])
        locations = filters.get("locations", [])
        profile_country = filters.get("profile_country")
        category = filters.get("category")

        must_clauses = []
        should_clauses = []
        filter_clauses = []

        must_clauses.append({
            "multi_match": {
                "query": query,
                "fields": [
                    "title^5",
                    "ai_keywords^6",
                    "content^2",
                    "location^3",
                    "user_name^2"
                ],
                "operator": "or",
                "fuzziness": "AUTO"
            }
        })

        for role in roles:

            should_clauses.append({
                "match": {
                    "title": {
                        "query": role,
                        "boost": 6
                    }
                }
            })

            should_clauses.append({
                "match": {
                    "content": {
                        "query": role,
                        "boost": 2
                    }
                }
            })

        LOCATION_SYNONYMS = {
            "uae": [
                "Dubai",
                "Abu Dhabi",
                "Sharjah",
                "United Arab Emirates",
                "UAE"
            ]
        }

        if locations:

            location_should = []

            for loc in locations:

                values = LOCATION_SYNONYMS.get(
                    loc.lower(),
                    [loc]
                )

                for value in values:

                    location_should.append({
                        "match": {
                            "location": {
                                "query": value,
                                "operator": "and"
                            }
                        }
                    })

            filter_clauses.append({
                "bool": {
                    "should": location_should,
                    "minimum_should_match": 1
                }
            })

        if profile_country:

            should_clauses.append({
                "term": {
                    "location": {
                        "value": profile_country,
                        "boost": 5
                    }
                }
            })

        if category:

            if isinstance(category, list):

                filter_clauses.append({
                    "terms": {
                        "category": category
                    }
                })

            else:

                filter_clauses.append({
                    "term": {
                        "category": category
                    }
                })

        body = {

            "size": size,

            "query": {

                "function_score": {

                    "query": {

                        "bool": {

                            "must": must_clauses,

                            "filter": filter_clauses,

                            "should": should_clauses

                        }

                    },

                    "functions": [

                        {
                            "filter": {
                                "term": {
                                    "is_EP": True
                                }
                            },
                            "weight": 5
                        },

                        {
                            "filter": {
                                "term": {
                                    "is_SP": True
                                }
                            },
                            "weight": 4
                        },

                        {
                            "filter": {
                                "term": {
                                    "is_GP": True
                                }
                            },
                            "weight": 3
                        },

                        {
                            "filter": {
                                "term": {
                                    "is_PREMIUM": True
                                }
                            },
                            "weight": 2
                        }

                    ],

                    "score_mode": "multiply",

                    "boost_mode": "multiply"

                }

            }

        }

        logger.info("=" * 80)
        logger.info("ELASTIC QUERY")
        logger.info(json.dumps(body, indent=2))
        logger.info("=" * 80)

        response = es.search(
            index=self.index,
            body=body
        )

        results = []

        logger.info("ELASTIC RESULTS")

        for hit in response["hits"]["hits"]:

            logger.info(
                f"{hit['_score']:.2f} | "
                f"{hit['_source'].get('category')} | "
                f"{hit['_source'].get('title')}"
            )

            results.append({
                "engine": "elastic",
                "score": hit["_score"],
                "document": {
                    "id": hit["_id"],
                    **hit["_source"]
                }
            })

        return results