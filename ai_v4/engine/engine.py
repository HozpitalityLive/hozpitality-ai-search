from fastapi import WebSocket

from ai_v4.config.logger import logger
from ai_v4.engine.execution.manager import ExecutionManager
from ai_v4.planner.planner import Planner
from ai_v4.planner.query_rewriter import QueryRewriter
from ai_v4.services.memory_service import MemoryService


class AIEngine:

    def __init__(self):
        self.planner = Planner()
        self.memory_service = MemoryService()
        self.query_rewriter = QueryRewriter()
        self.execution = ExecutionManager()

    async def execute(
        self,
        user_id,
        websocket: WebSocket,
        query: str,
        memory=None
    ):

        memory = await self.memory_service.load(user_id)

        logger.info("=" * 80)
        logger.info("AI ENGINE STARTED")
        logger.info(f"Query : {query}")

        pending_query = memory.get("pending_query")

        if pending_query:

            logger.info(f"Pending Query : {pending_query}")

            query = await self.query_rewriter.rewrite(
                original_query=pending_query,
                clarification_answer=query
            )

            logger.info(f"Merged Query : {query}")

            memory.pop("pending_query", None)

        plan = await self.planner.create_plan(query)

        return await self.execution.execute(
            user_id=user_id,
            websocket=websocket,
            query=query,
            plan=plan,
            memory=memory
        )