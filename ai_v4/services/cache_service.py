from fastapi import WebSocket

from ai_v4.config.logger import logger

from ai_v4.planner.planner import Planner
from ai_v4.memory.llama_memory import LlamaMemory
from ai_v4.context.builder import ContextBuilder
from ai_v4.llm.response import ResponseGenerator

class ChatService:

    def __init__(self):
        self.planner = Planner()
        self.memory = LlamaMemory()
        self.context = ContextBuilder()
        self.response = ResponseGenerator()

    async def handle_message(
        self,
        websocket: WebSocket,
        payload: dict
    ):

        try:

            query = payload.get("query", "").strip()
            user_id = payload.get("user_id", 0)
            conversation_id = payload.get(
                "conversation_id"
            )

            if not query:
                await websocket.send_json({
                    "type": "error",
                    "message": "Empty query."
                })
                return

            logger.info(
                f"Query : {query}"
            )

            memory = await self.memory.load(
                user_id=user_id,
                conversation_id=conversation_id
            )

            plan = await self.planner.run(
                query=query,
                memory=memory
            )

            context = await self.context.build(
                query=query,
                plan=plan,
                memory=memory
            )

            await self.response.generate(
                websocket=websocket,
                query=query,
                context=context
            )

            await self.memory.save(
                user_id=user_id,
                conversation_id=conversation_id,
                query=query
            )

        except Exception as e:
            logger.exception(e)
            await websocket.send_json({
                "type": "error",
                "message": "Internal server error."
            })