from fastapi import FastAPI, WebSocket , Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse
)

from ai_server import app as app1
from ai_v2 import app as app2v2
from ai_v3 import app as app3v3 
import httpx


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



@main_app.post("/api/ollama-chat")
async def ollama_chat(request: Request):

    print("🔥 OLLAMA ROUTE HIT", flush=True)

    try:

        body = await request.json()

        async with httpx.AsyncClient(
            timeout=120
        ) as client:

            response = await client.post(

                "http://ollama:11434/api/chat",

                json={

                    "model":
                        body.get(
                            "model",
                            "phi3-hoz:latest"
                        ),

                    "messages":
                        body.get(
                            "messages",
                            []
                        ),

                    "stream": False

                },

                headers={
                    "Content-Type":
                        "application/json"
                }

            )

        data = response.json()

        return JSONResponse(
            content=data
        )

    except Exception as e:

        print(
            "❌ OLLAMA PROXY ERROR:",
            repr(e),
            flush=True
        )

        return JSONResponse(

            status_code=500,

            content={
                "error": str(e)
            }

        )


# ✅ MOUNT APPS
main_app.mount("/v2", app1)
main_app.mount("/app2v2", app2v2)
main_app.mount("/app3v3", app3v3)
