# ai_v4/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_v4.config.logger import logger
from ai_v4.websocket.chat import router as websocket_router


app = FastAPI(
    title="Hozpitality AI Search v4",
    version="4.0.0"
)


@app.on_event("startup")
async def startup():

    logger.info("🚀 AI Search v4 Started")


@app.on_event("shutdown")
async def shutdown():

    logger.info("🛑 AI Search v4 Stopped")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hozpitality.com",
        "https://www.hozpitality.com",
        "http://localhost:3000",
    ],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    websocket_router,
    prefix="/v4"
)


@app.get("/")
async def home():

    return {
        "status": "running",
        "version": "4.0.0"
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }