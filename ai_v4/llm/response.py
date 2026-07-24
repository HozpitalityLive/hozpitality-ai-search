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

        await websocket.send_json({
            "type": "thinking",
            "message": "Thinking..."
        })

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

        logger.info("=" * 80)
        logger.info("RAW LLM RESPONSE")
        logger.info(llm_response)
        logger.info("=" * 80)

        try:

            response = llm_response.strip()

            if response.startswith("```json"):
                response = response.replace("```json", "", 1)

            if response.startswith("```"):
                response = response.replace("```", "", 1)

            if response.endswith("```"):
                response = response[:-3]

            response = response.strip()

            start = response.find("{")
            end = response.rfind("}")

            if start == -1 or end == -1:
                raise ValueError("No JSON object found.")

            response = response[start:end + 1]

            ai = json.loads(response)

            intro = ai.get("intro")
            description = ai.get("description")

            if not intro and "response" in ai:
                intro = ai["response"]
                description = ""

            ai["intro"] = intro or ""
            ai["description"] = description or ""
            ai.setdefault("follow_up", [])
            ai.setdefault("intent", agent)

        except Exception:

            logger.exception("Invalid JSON from LLM")
            logger.error("RAW RESPONSE:")
            logger.error(repr(llm_response))

            ai = {
                "intent": agent,
                "intro": "I found matching results.",
                "description": "I'm unable to generate a summary right now, but the search results are available below.",
                "follow_up": []
            }

        await websocket.send_json({
            "type": "intro",
            "intent": ai.get("intent", agent),
            "content": ai.get("intro", "")
        })

        await websocket.send_json({
            "type": "description",
            "content": ai.get("description", "")
        })

        await websocket.send_json({
            "type": "results",
            "intent": ai.get("intent", agent),
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

        logger.info("[4/4] Response sent successfully")

        return ai