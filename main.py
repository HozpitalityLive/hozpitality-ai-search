from fastapi import FastAPI
from ai_server import app as app1
from ai_v2 import app as app2v2

from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket


main_app = FastAPI()

main_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hozpitality.com",
        "https://www.hozpitality.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@main_app.websocket("/ws/chattest")
async def websocket_chat_main(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("connected")

main_app.mount("/v2", app1)
main_app.mount("/app2v2", app2v2)
# main_app.mount("", app1)
