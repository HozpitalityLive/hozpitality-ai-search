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
        body = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "title^5",
                        "ai_keywords^4",
                        "content^2",
                        "location",
                        "user_name"
                    ],
                    "type": "best_fields"
                }
            }
        }


        response = es.search(
            index=self.index,
            body=body
        )

        results = []

        for hit in response["hits"]["hits"]:
            results.append({
                "engine": "elastic",
                "score": hit["_score"],
                "document": {
                    "id": hit["_id"],
                    **hit["_source"]
                }
            })

        return results