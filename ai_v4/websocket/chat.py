from fastapi import APIRouter,WebSocket,WebSocketDisconnect
import json
from ai_v4.websocket.manager import manager
from ai_v4.services.chat_service import ChatService
from ai_v4.config.settings import settings

router=APIRouter(); chat_service=ChatService()

@router.websocket("/ws/chat")
async def ai_search(websocket:WebSocket):
    user_id=0
    await websocket.accept()
    try:
        first=json.loads(await websocket.receive_text())
        user_id=int(first.get("user_id",0))
        if settings.API_KEY and first.get("api_key")!=settings.API_KEY:
            await websocket.send_json({"type":"error","message":"Unauthorized"});await websocket.close(code=1008);return
        await manager.connect(websocket,user_id)
        await chat_service.handle_message(user_id,websocket,first)
        while True:
            payload=json.loads(await websocket.receive_text())
            await chat_service.handle_message(user_id,websocket,payload)
    except WebSocketDisconnect:
        await manager.disconnect(user_id)
    except Exception:
        await manager.disconnect(user_id)
