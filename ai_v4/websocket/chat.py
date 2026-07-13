# ai_v4/websocket/chat.py

from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

import json

from ai_v4.websocket.manager import manager
from ai_v4.services.chat_service import ChatService
from ai_v4.config.logger import logger

router = APIRouter()

chat_service = ChatService()

@router.websocket("/ws/chat")
async def ai_search(websocket: WebSocket):

    user_id = 0

    await websocket.accept()

    try:

        first_message = await websocket.receive_text()
        payload = json.loads(first_message)

        user_id = int(payload.get("user_id", 0))

        await manager.connect(
            websocket,
            user_id
        )

        logger.info(
            f"Connected User : {user_id}"
        )

        await chat_service.handle_message(
            websocket,
            payload
        )

        while True:
            message = await websocket.receive_text()
            payload = json.loads(message)

            await chat_service.handle_message(
                websocket,
                payload
            )

    except WebSocketDisconnect:
        logger.info(
            f"User Disconnected : {user_id}"
        )
        await manager.disconnect(user_id)

    except Exception as e:
        logger.exception(e)
        await manager.disconnect(user_id)