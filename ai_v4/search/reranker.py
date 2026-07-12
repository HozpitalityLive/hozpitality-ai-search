from ai_v4.config.logger import logger


class SearchReranker:

    def rerank(
        self,
        query: str,
        results: list
    ) -> list:

        logger.info("Reranking Results")

        return sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )