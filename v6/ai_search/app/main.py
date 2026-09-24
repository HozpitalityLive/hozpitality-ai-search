from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from .config import settings
from .db import close_client, get_collection, ping
from .repository import SearchDocumentsRepository
from .schemas import SearchRequest, SearchResponse
from .service import SearchService


repository = SearchDocumentsRepository(get_collection())
service = SearchService(repository, fuzzy_threshold=settings.fuzzy_threshold)


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


@app.get("/health")
def health() -> dict:
    try:
        ping()
        return {"ok": True, "service": "hozpitality-ai-search", "database": "mongodb"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB unavailable: {exc}") from exc



@app.get("/search", response_model=SearchResponse)
def search_get(
    q: str = Query(min_length=1, max_length=300),
    entity: str | None = None,
    city: str | None = None,
    country: str | None = None,
    status: str | None = None,
    is_live: bool | None = True,
    limit: int = Query(default=5, ge=1, le=5),
) -> SearchResponse:
    return SearchResponse(
        **service.search(
            query=q,
            entity=entity,
            city=city,
            country=country,
            status=status,
            is_live=is_live,
            limit=min(limit, 5),
        )
    )

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    return SearchResponse(
        **service.search(
            query=request.q,
            entity=request.entity,
            city=request.city,
            country=request.country,
            status=request.status,
            is_live=request.is_live,
            limit=min(request.limit, 5),
        )
    )
