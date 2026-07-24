import json

from ai_v4.services.agent_service import AgentService
from ai_v4.context.builder import ContextBuilder
from ai_v4.llm.response import ResponseGenerator
from ai_v4.services.memory_service import MemoryService
from ai_v4.config.logger import logger


class SearchExecution:

    def __init__(self):
        self.agent_service = AgentService()
        self.context_builder = ContextBuilder()
        self.response = ResponseGenerator()
        self.memory_service = MemoryService()

    async def execute(
        self,
        user_id,
        websocket,
        query,
        plan,
        memory
    ):

        logger.info(f"Plan : {plan}")

        agent_output = await self.agent_service.execute(
            plan=plan,
            query=query,
            memory=memory
        )

        memory["last_search"] = {
            "query": query,
            "agent": agent_output["agent"],
            "filters": agent_output["filters"],
            "page": 1,
            "page_size": 5,
            "total": len(agent_output["results"])
        }

        logger.info("=" * 80)
        logger.info("RAW SEARCH RESULTS")
        logger.info(json.dumps(agent_output["results"], indent=2, default=str))
        logger.info("=" * 80)

        context = await self.context_builder.build(
            query=query,
            search_results=agent_output["results"],
            memory=memory
        )

        response = await self.response.generate(
            websocket=websocket,
            agent=agent_output["agent"],
            query=query,
            context=context,
            memory=memory,
            model=plan["llm"]["model"]
        )

        memory["conversation"].append({
            "role": "user",
            "content": query
        })

        memory["conversation"].append({
            "role": "assistant",
            "content": response.get("intro", "")
        })

        await self.memory_service.save(
            user_id,
            memory
        )

        return response