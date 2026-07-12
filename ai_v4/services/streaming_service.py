from fastapi import WebSocket


class StreamingService:

    async def token(
        self,
        websocket: WebSocket,
        token: str
    ):

        await websocket.send_json({
            "type": "token",
            "content": token
        })

    async def done(
        self,
        websocket: WebSocket
    ):

        await websocket.send_json({
            "type": "done"
        })

    async def error(
        self,
        websocket: WebSocket,
        message: str
    ):

        await websocket.send_json({
            "type": "error",
            "message": message
        })

    async def status(
        self,
        websocket: WebSocket,
        message: str
    ):

        await websocket.send_json({
            "type": "status",
            "message": message
        })

    async def search_results(
        self,
        websocket: WebSocket,
        results: list
    ):

        await websocket.send_json({
            "type": "search_results",
            "results": results
        })

    async def followups(
        self,
        websocket: WebSocket,
        questions: list
    ):

        await websocket.send_json({
            "type": "followups",
            "questions": questions
        })