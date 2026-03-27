from fastapi import FastAPI
from ai_server import app as app1
from ai_server_2 import app as app2

from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket


main_app = FastAPI()

main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@main_app.websocket("/ws/chattest")
async def websocket_chat_main(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("connected")

main_app.mount("/v2", app2)
main_app.mount("", app1)
