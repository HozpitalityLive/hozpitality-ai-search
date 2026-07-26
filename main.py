# main.py

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from ai_server import app as app1
from ai_v2 import app as app2v2
from ai_v3 import app as app3v3 
from ai_v4.websocket.chat import router as chat_router





# ✅ CREATE APP
main_app = FastAPI()


# ✅ STARTUP (NO ES, NO INDEXING)
@main_app.on_event("startup")
def startup():
    print("🚀 MAIN APP STARTED (NO ES TOUCH)", flush=True)


# ✅ MIDDLEWARE
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


# ✅ TEST WEBSOCKET
@main_app.websocket("/ws/chattest")
async def websocket_chat_main(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("connected")


# ✅ MOUNT APPS
main_app.mount("/v2", app1)
main_app.mount("/app2v2", app2v2)
main_app.mount("/app3v3", app3v3)
main_app.mount("/appv4", chat_router)
