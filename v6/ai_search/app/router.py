from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .config import settings
from .db import get_collection, ping
from .repository import SearchDocumentsRepository
from .schemas import SearchRequest, SearchResponse
from .service import SearchService
from .query_understanding import understand, clarification_for


repository = SearchDocumentsRepository(get_collection())
service = SearchService(
    repository,
    fuzzy_threshold=settings.fuzzy_threshold,
)

router = APIRouter(tags=["MongoDB Search"])


@router.get("/search", response_model=SearchResponse)
def search_get(
    q: str = Query(min_length=1, max_length=300),
    entity: str | None = None,
    city: str | None = None,
    country: str | None = None,
    status: str | None = None,
    is_live: bool | None = None,
    limit: int = Query(default=5, ge=1, le=5),
) -> SearchResponse:
    try:
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
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB search unavailable: {exc}",
        ) from exc


@router.post("/search", response_model=SearchResponse)
def search_post(request: SearchRequest) -> SearchResponse:
    try:
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
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB search unavailable: {exc}",
        ) from exc


@router.get("/search/understand")
def search_understand(q: str = Query(min_length=1, max_length=300)) -> dict:
    plan = understand(q)
    plan.clarification = clarification_for(plan)
    return plan.as_dict()


@router.get("/search/health")
def search_health() -> dict:
    try:
        ping()
        count = repository.collection.count_documents({})
        return {
            "ok": True,
            "service": "hozpitality-ai-search-phase2",
            "database": settings.mongodb_database,
            "collection": settings.mongodb_collection,
            "documents": count,
            "semantic_search_enabled": service.semantic.enabled,
            "llm_fallback_enabled": service.llm.enabled,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB unavailable: {exc}",
        ) from exc


def register_mongo_search_routes(app) -> None:
    """Attach Phase 1 routes to the existing V6 FastAPI application."""
    app.include_router(router)
    try:
        repository.ensure_indexes()
    except Exception as exc:
        # Do not prevent the existing V6 API from booting if MongoDB is
        # temporarily unavailable. /search/health and /search will report it.
        print(f"MongoDB Phase 1 index initialization skipped: {exc}")
