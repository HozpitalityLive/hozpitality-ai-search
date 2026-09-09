from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_v4.config.settings import settings
from ai_v4.config.logger import logger
from ai_v4.websocket.chat import router as websocket_router
from ai_v4.ai_profile_writer import router as profile_writer_router
from ai_v4.llm.ollama import OllamaClient
from ai_v4.config.database import db_pool

@asynccontextmanager
async def lifespan(app):
    logger.info("Hozpitality AI Search V4 starting")
    yield
    try: db_pool.closeall()
    except Exception: pass

app=FastAPI(title="Hozpitality AI Search V4",version="4.1.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=[x.strip() for x in settings.CORS_ORIGINS.split(",") if x.strip()],
                   allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.include_router(websocket_router,prefix="/v4")
app.include_router(profile_writer_router,prefix="/v4/ai-profile-writer",tags=["AI Profile Writer"])

@app.get("/v4/")
async def home(): return {"status":"running","version":"4.1.0","architecture":"planner-agent-hybrid-search"}

@app.get("/v4/health")
async def health():
    return {"status":"healthy","version":"4.1.0"}

@app.get("/v4/ready")
async def ready():
    import asyncio
    ok=await OllamaClient().health()
    db_ok=False
    try:
        def check():
            c=db_pool.getconn()
            try:
                with c.cursor() as cur: cur.execute("SELECT 1")
            finally: db_pool.putconn(c)
        await asyncio.to_thread(check);db_ok=True
    except Exception: pass
    return {"status":"ready" if ok and db_ok else "degraded","postgres":db_ok,"ollama":ok}
