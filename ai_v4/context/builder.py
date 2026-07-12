from ai_v4.context.formatter import ContextFormatter
from ai_v4.config.logger import logger


class ContextBuilder:

    def __init__(self):
        self.formatter = ContextFormatter()

    async def build(
        self,
        query: str,
        search_results: list,
        memory: dict | None = None
    ):
        
        logger.info("[3/4] Building context...")

        logger.info("Building Context")
        formatted_results = self.formatter.format_documents(
            search_results
        )

        return {
            "query": query,
            "memory": memory,
            "documents": formatted_results
        }