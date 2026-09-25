"""Standalone Hozpitality AI Search service.

    uvicorn ai_search.app.main:app --host 127.0.0.1 --port 8086

The same routes are also mounted into the V6 application (root main.py) via
``register_mongo_search_routes``.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .container import get_container
from .db import close_client, ping
from .observability import configure_logging, log_event, metrics, new_request_id
from .router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        get_container().ensure_indexes()
    except Exception as exc:
        # Start anyway; /health reports MongoDB status.
        log_event("index_init_failed", level=logging.ERROR, error=type(exc).__name__)
    container = get_container()
    if container.search.semantic.enabled:
        container.search.semantic.warmup_async()
    yield
    container.llm.close()
    close_client()


app = FastAPI(
    title="Hozpitality AI Search",
    version="4.0.0",
    description="Hozpitality conversational AI search: MongoDB retrieval, conversation memory, grounded answers.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID", "X-User-Id"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    started = time.perf_counter()
    rid = new_request_id(request.headers.get("x-request-id"))
    response = await call_next(request)
    response.headers.setdefault("X-Request-ID", rid)
    duration = round((time.perf_counter() - started) * 1000, 1)
    metrics.observe("http_ms", duration)
    log_event(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration,
        request_id=response.headers.get("X-Request-ID"),
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    metrics.incr("validation_errors")
    # Do not echo the submitted body back (it may contain personal data).
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
                for e in exc.errors()
            ]
        },
    )


app.include_router(router)


@app.get("/health")
def health() -> dict:
    try:
        ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="MongoDB unavailable") from exc
    container = get_container()
    return {
        "ok": True,
        "service": "hozpitality-ai-search",
        "database": settings.mongodb_database,
        "llm": {
            "enabled": container.llm.enabled,
            "model": container.llm.model,
            "circuit": container.llm.breaker.state,
        },
    }
