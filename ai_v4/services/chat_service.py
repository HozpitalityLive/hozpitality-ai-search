from fastapi import WebSocket
from ai_v4.engine.engine import AIEngine
class ChatService:
    def __init__(self): self.engine=AIEngine()
    async def handle_message(self,user_id,websocket,payload):
        query=(payload.get("query") or payload.get("message") or "").strip()
        if not query:
            await websocket.send_json({"type":"error","message":"Query is required."});return
        await self.engine.execute(user_id,websocket,query)
