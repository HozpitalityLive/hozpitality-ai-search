from fastapi import WebSocket
from ai_v4.engine.engine import AIEngine


class ChatService:

    def __init__(self):
        self.engine = AIEngine()

    async def handle_message(
        self,
        user_id:None,
        websocket: WebSocket,
        payload: dict
    ):

        query = payload.get("query", "").strip()

        if not query:
            await websocket.send_json({
                "type":"error",
                "message":"Query is required."
            })

            return

        await self.engine.execute(
            websocket=websocket,
            query=query,
            user_id=user_id,
        )