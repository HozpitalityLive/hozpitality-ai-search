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
    print("\n🚀 STARTING SYSTEM", flush=True)

    try:
        es_ready = False

        for i in range(30):
            try:
                print(f"⏳ Waiting for ES... {i+1}/30", flush=True)

                if es.ping():
                    print("✅ Elasticsearch connected", flush=True)

                    time.sleep(5)

                    es_ready = True
                    break

            except Exception as e:
                print("❌ ES ping error:", e, flush=True)

            time.sleep(2)

        if not es_ready:
            print("❌ Elasticsearch NOT ready after retries", flush=True)
            return  # safe exit

        exists = es.indices.exists(index="hozpitality")

        if exists:
            count = es.count(index="hozpitality")["count"]
            print(f"📊 ES docs: {count}", flush=True)

            if count == 0:
                print("⚡ Empty index → full load_data()", flush=True)
                load_data(force_reindex=True)
            else:
                print("✅ ES ready → rebuild FAISS only", flush=True)
                load_data(force_reindex=False)

        else:
            print("⚡ First time → creating index + FAISS", flush=True)
            load_data(force_reindex=True)

    except Exception as e:
        print("❌ Startup error:", e)
        traceback.print_exc()


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