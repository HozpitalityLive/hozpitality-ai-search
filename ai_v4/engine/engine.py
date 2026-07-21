from fastapi import WebSocket

from ai_v4.config.logger import logger
from ai_v4.planner.planner import Planner
from ai_v4.services.agent_service import AgentService
from ai_v4.services.memory_service import MemoryService
from ai_v4.context.builder import ContextBuilder
from ai_v4.llm.response import ResponseGenerator
import json


class AIEngine:

    def __init__(self):
        self.planner = Planner()
        self.agent_service = AgentService()
        self.context_builder = ContextBuilder()
        self.response = ResponseGenerator()
        self.memory_service = MemoryService()

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
 
        plan = await self.planner.create_plan(
            query=query
        )

        if plan["execution"]["type"] == "chat":
            response = await self.response.generate(
                websocket=websocket,
                agent="chat",
                query=query,
                context={
                    "query": query,
                    "memory": memory,
                    "documents": "",
                    "results": []
                },
                memory=memory,
                model=plan["llm"]["model"]
            )

            return response

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

        logger.info(
            f"Agent : {agent_output['agent']}"
        )

        logger.info(
            f"Results : {len(agent_output['results'])}"
        )

        context = await self.context_builder.build(
            query=query,
            search_results=agent_output["results"],
            memory=memory
        )

        logger.info("=" * 80)
        logger.info("FINAL CONTEXT")
        logger.info(json.dumps(context, indent=2, default=str))
        logger.info("=" * 80)

        logger.info("Context Created")
  
        response = await self.response.generate(
            websocket=websocket,
            agent=agent_output["agent"],
            query=query,
            context=context,
            memory=memory,
            model=plan["llm"]["model"]
        )

        logger.info("AI ENGINE FINISHED")


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