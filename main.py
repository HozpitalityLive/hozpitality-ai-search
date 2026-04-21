from fastapi import FastAPI
from ai_server import app as app1
from ai_v2 import app as app2v2

from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket
from ai_v2 import load_data, es
import time
import traceback


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


@main_app.on_event("startup")
def startup():
    print("\n🚀 MAIN APP STARTUP", flush=True)

    print("⏳ Waiting for Elasticsearch...", flush=True)

    for i in range(30):
        try:
            print(f"🔁 ES {i+1}/30", flush=True)

            if es.ping():
                print("✅ Elasticsearch ready", flush=True)
                time.sleep(5)
                break
        except Exception as e:
            print("❌ ES error:", e, flush=True)

        time.sleep(2)
    else:
        print("❌ ES NOT READY", flush=True)

    print("🚀 Running load_data()", flush=True)

    try:
        load_data()
        print("✅ load_data DONE", flush=True)
    except Exception as e:
        print("❌ load_data FAILED:", e, flush=True)
        traceback.print_exc()
