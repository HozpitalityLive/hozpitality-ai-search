from fastapi import WebSocket
from typing import Dict
import asyncio

from ai_v4.config.logger import logger

class ConnectionManager:

    def __init__(self):

        self.active_connections: Dict[int, WebSocket] = {}

        self.lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        user_id: int
    ):

        async with self.lock:

            self.active_connections[user_id] = websocket

        logger.info(
            f"✅ User Connected : {user_id}"
        )

    async def disconnect(
        self,
        user_id: int
    ):

        async with self.lock:

            if user_id in self.active_connections:

                del self.active_connections[user_id]

        logger.info(
            f"❌ User Disconnected : {user_id}"
        )

    async def send_json(
        self,
        user_id: int,
        data: dict
    ):

        websocket = self.active_connections.get(user_id)

        if not websocket:
            return

        try:

            await websocket.send_json(data)

        except Exception as e:

            logger.error(
                f"Send Error : {e}"
            )

            await self.disconnect(user_id)

    async def send_text(
        self,
        user_id: int,
        text: str
    ):

        websocket = self.active_connections.get(user_id)

        if not websocket:
            return

        try:

            await websocket.send_text(text)

        except Exception as e:

            logger.error(
                f"Send Error : {e}"
            )

            await self.disconnect(user_id)

    async def broadcast(
        self,
        data: dict
    ):

        disconnected = []

        for user_id, websocket in self.active_connections.items():

            try:

                await websocket.send_json(data)

            except:

                disconnected.append(user_id)

        for user_id in disconnected:

            await self.disconnect(user_id)

    def is_connected(
        self,
        user_id: int
    ) -> bool:

        return user_id in self.active_connections

    def total_connections(self) -> int:

        return len(self.active_connections)

manager = ConnectionManager()