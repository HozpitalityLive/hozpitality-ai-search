from fastapi import WebSocket
import json

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

        llm_response = ""

        async for chunk in self.brain.think(
            agent=agent,
            query=query,
            context=context,
            memory=memory,
            model=model
        ):

            token = chunk.get("response", "")

            if token:
                llm_response += token

        logger.info("LLM Response Received")

        try:

            ai = json.loads(llm_response)

        except Exception:

            logger.exception("Invalid JSON from LLM")

            ai = {
                "intro": "I found matching results.",
                "description": "I'm unable to generate a summary right now, but the search results are available below.",
                "follow_up": []
            }

        await websocket.send_json({
            "type": "intro",
            "intent": ai.get("intent", ""),
            "content": ai.get("intro", "")
        })

        await websocket.send_json({
            "type": "description",
            "content": ai.get("description", "")
        })

        await websocket.send_json({
            "type": "results",
            "intent": ai.get("intent", ""),
            "total": context.get("total", 0),
            "results": context.get("results", [])
        })

        await websocket.send_json({
            "type": "follow_up",
            "questions": ai.get("follow_up", [])
        })

        await websocket.send_json({
            "type": "done"
        })

        return ai