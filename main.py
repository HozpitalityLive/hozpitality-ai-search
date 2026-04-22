from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from ai_server import app as app1
from ai_v2 import app as app2v2
from ai_v2 import load_data, es

import time
import traceback


# ✅ CREATE APP FIRST
main_app = FastAPI()


# ✅ STARTUP MUST COME BEFORE mount()
@main_app.on_event("startup")
def startup():
    print("\n🚀 FAST STARTUP MODE", flush=True)

    try:
        # ⚡ fast ES wait
        es_ready = False
        for _ in range(5):
            if es.ping():
                es_ready = True
                break
            time.sleep(1)

        if not es_ready:
            print("❌ Elasticsearch not ready → skipping init")
            return

        exists = es.indices.exists(index="hozpitality")

        if exists:
            print("✅ ES index exists → rebuilding FAISS only")
            load_data()
            return

        print("⚡ First time setup → full load_data()")
        load_data()

    except Exception as e:
        print("❌ Startup error:", e)


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


# ✅ MOUNT APPS (MUST BE LAST)
main_app.mount("/v2", app1)
main_app.mount("/app2v2", app2v2)
# main_app.mount("", app1)