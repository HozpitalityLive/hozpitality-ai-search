from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .config import settings
from .db import close_client, ping
from .router import router, repository


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository.ensure_indexes()
    yield
    close_client()


app = FastAPI(
    title="Hozpitality AI Search",
    version="1.0.0-phase1",
    description="MongoDB-first Hozpitality search foundation.",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
def health() -> dict:
    try:
        ping()
        return {
            "ok": True,
            "service": "hozpitality-ai-search",
            "database": settings.mongodb_database,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB unavailable: {exc}",
        ) from exc
