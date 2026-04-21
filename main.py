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
    print("\n🚀 ===== MAIN APP STARTUP =====", flush=True)

    print("⏳ Waiting for Elasticsearch...", flush=True)

    es_ready = False

    for i in range(30):
        try:
            print(f"🔁 ES attempt {i+1}/30", flush=True)

            if es.ping():
                print("✅ Elasticsearch ping successful", flush=True)

                try:
                    health = es.cluster.health()
                    print(f"📊 ES Health: {health['status']}", flush=True)
                except Exception as e:
                    print(f"⚠️ ES health check failed: {e}", flush=True)

                print("⏳ Extra wait for ES readiness (5s)...", flush=True)
                time.sleep(5)

                es_ready = True
                break

        except Exception as e:
            print(f"❌ ES ping error: {e}", flush=True)

        time.sleep(2)

    if not es_ready:
        print("❌ Elasticsearch NOT reachable after retries", flush=True)
    else:
        print("✅ Elasticsearch is ready", flush=True)

    # 🔍 Check index BEFORE load
    try:
        exists = es.indices.exists(index="hozpitality")
        print(f"🔍 Index exists BEFORE load: {exists}", flush=True)
    except Exception as e:
        print(f"❌ Index check failed: {e}", flush=True)

    # 🚀 Load data
    print("🚀 Running load_data()", flush=True)

    try:
        load_data()
        print("✅ load_data() COMPLETED", flush=True)
    except Exception as e:
        print("❌ load_data() FAILED:", str(e), flush=True)
        traceback.print_exc()

    # 🔍 Verify index AFTER load
    try:
        exists = es.indices.exists(index="hozpitality")
        print(f"🔍 Index exists AFTER load: {exists}", flush=True)

        if exists:
            count = es.count(index="hozpitality")["count"]
            print(f"📊 Indexed documents: {count}", flush=True)

    except Exception as e:
        print(f"❌ Post-load verification failed: {e}", flush=True)

    print("🏁 ===== STARTUP COMPLETE =====\n", flush=True)


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