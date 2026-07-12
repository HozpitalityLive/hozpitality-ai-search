from fastapi import WebSocket

from ai_v4.config.logger import logger
from ai_v4.llm.brain import Brain


class ResponseGenerator:

    def __init__(self):

        self.brain = Brain()

    async def generate(
        self,
        websocket: WebSocket,
        agent: str,
        query: str,
        context: dict,
        memory: dict | None = None,
        model: str | None = None
    ):
        logger.info("[4/4] Generating AI response...")

        logger.info("Generating AI Response")

        full_response = ""

        async for chunk in self.brain.think(
            agent=agent,
            query=query,
            context=context,
            memory=memory,
            model=model
        ):

            token = chunk.get("response", "")

            if token:
                full_response += token
                await websocket.send_json({
                    "type": "token",
                    "content": token
                })

        await websocket.send_json({
            "type": "done"
        })

        return full_response