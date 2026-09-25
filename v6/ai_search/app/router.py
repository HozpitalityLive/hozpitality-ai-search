from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from .chat_service import PreparedTurn
from .config import settings
from .container import get_container
from .conversation import ConversationConflict, ConversationForbidden
from .db import ping
from .observability import (
    configure_logging,
    log_event,
    metrics,
    request_context,
    set_conversation_id,
)
from .ollama_client import LlmUnavailable
from .query_understanding import clarification_for, understand
from .schemas import (
    ChatRequest,
    ChatResponse,
    ConversationView,
    SearchRequest,
    SearchResponse,
)
from .security import (
    api_key_valid,
    rate_limiter,
    trusted_user_id,
    valid_conversation_id,
)
from .dialogue import interpret
from .state import apply_intent, empty_state, normalize_state, public_state

router = APIRouter(tags=["MongoDB Search", "AI Chat"])


# ---------------------------------------------------------------------------
# Request guards
# ---------------------------------------------------------------------------


def _client_key(host: str | None, headers: Any, user_id: str | None) -> str:
    if user_id:
        return f"user:{user_id}"
    # Only trust X-Real-IP when the request comes from the local reverse proxy.
    if host in {"127.0.0.1", "::1"} and headers.get("x-real-ip"):
        return f"ip:{headers.get('x-real-ip')}"
    return f"ip:{host or 'unknown'}"


def _guard(
    request: Request, kind: str, api_key: str | None, user_header: str | None
) -> str | None:
    if not api_key_valid(api_key or request.query_params.get("api_key")):
        metrics.incr("auth_rejected")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    user_id = trusted_user_id(user_header)
    limit = (
        settings.rate_limit_chat_per_minute
        if kind == "chat"
        else settings.rate_limit_search_per_minute
    )
    allowed, retry_after = rate_limiter.allow(
        f"{kind}:{_client_key(request.client.host if request.client else None, request.headers, user_id)}",
        limit,
    )
    if not allowed:
        metrics.incr("rate_limited")
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down.",
            headers={"Retry-After": str(max(1, int(retry_after)))},
        )
    return user_id


# ---------------------------------------------------------------------------
# Search (Phase 1/2) - unchanged contract
# ---------------------------------------------------------------------------


def _run_search(**kwargs: Any) -> SearchResponse:
    try:
        return SearchResponse(**get_container().search.search(**kwargs))
    except HTTPException:
        raise
    except Exception as exc:
        metrics.incr("search_errors")
        log_event("search_error", level=logging.ERROR, error=type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="MongoDB search unavailable"
        ) from exc


@router.get("/search", response_model=SearchResponse)
def search_get(
    request: Request,
    response: Response,
    q: str = Query(min_length=1, max_length=300),
    entity: str | None = None,
    city: str | None = Query(default=None, max_length=120),
    country: str | None = Query(default=None, max_length=120),
    status: str | None = Query(default=None, max_length=80),
    is_live: bool | None = None,
    limit: int = Query(default=5, ge=1, le=5),
    x_api_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> SearchResponse:
    with request_context(x_request_id) as rid:
        response.headers["X-Request-ID"] = rid
        _guard(request, "search", x_api_key, x_user_id)
        metrics.incr("search_requests")
        return _run_search(
            query=q,
            entity=entity,
            city=city,
            country=country,
            status=status,
            is_live=is_live,
            limit=min(limit, 5),
        )


@router.post("/search", response_model=SearchResponse)
def search_post(
    body: SearchRequest,
    request: Request,
    response: Response,
    x_api_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> SearchResponse:
    with request_context(x_request_id) as rid:
        response.headers["X-Request-ID"] = rid
        _guard(request, "search", x_api_key, x_user_id)
        metrics.incr("search_requests")
        return _run_search(
            query=body.q,
            entity=body.entity,
            city=body.city,
            country=body.country,
            status=body.status,
            is_live=body.is_live,
            limit=min(body.limit, 5),
        )


@router.get("/search/understand")
def search_understand(
    request: Request,
    q: str = Query(min_length=1, max_length=300),
    conversation_id: str | None = Query(default=None, max_length=128),
    x_api_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> dict:
    owner = _guard(request, "search", x_api_key, x_user_id)
    plan = understand(q)
    plan.clarification = clarification_for(plan)
    data = plan.as_dict()
    data["location"] = {
        "city": plan.city,
        "country": plan.country,
        "raw": plan.location_text,
    }
    data["is_new_search"] = True
    data["is_context_continuation"] = False
    if conversation_id:
        # Dry run against a conversation: how would this message change the
        # current search state? Nothing is saved.
        conversation = _conversation_or_404(conversation_id, owner)
        state = normalize_state(conversation.get("state") or {})
        intent = interpret(q, state)
        data["turn"] = intent.as_dict()
        if intent.action == "search":
            preview = apply_intent(state, intent)
            context = dict(preview.get("last_transition") or {})
            data["resulting_state"] = public_state(preview)
        else:
            context = {
                "transition": intent.transition or "continuation",
                "reason": intent.transition_reason,
            }
        data["context"] = context
        data["is_new_search"] = context.get("transition") in {
            "new_search",
            "faq",
            "facet",
        }
        data["is_context_continuation"] = context.get("transition") in {
            "modification",
            "continuation",
            "clarification_answer",
        }
    return data


@router.get("/search/health")
def search_health() -> dict:
    container = get_container()
    try:
        ping()
        count = container.repository.collection.estimated_document_count()
    except Exception as exc:
        log_event(
            "health_mongo_unavailable", level=logging.ERROR, error=type(exc).__name__
        )
        raise HTTPException(status_code=503, detail="MongoDB unavailable") from exc
    return {
        "ok": True,
        "service": "hozpitality-ai-search",
        "database": settings.mongodb_database,
        "collection": settings.mongodb_collection,
        "chat_collection": settings.chat_collection,
        "documents": count,
        "semantic_search_enabled": container.search.semantic.enabled,
        "llm_fallback_enabled": container.search.llm.enabled,
        "chat_enabled": container.llm.enabled,
        "chat_model": container.llm.model,
        "llm_circuit": container.llm.breaker.state,
    }


@router.get("/metrics")
def metrics_view(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> dict:
    if not api_key_valid(x_api_key or request.query_params.get("api_key")):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    container = get_container()
    return {**metrics.snapshot(), "llm_circuit": container.llm.breaker.state}


# ---------------------------------------------------------------------------
# Chat (Phase 3/4)
# ---------------------------------------------------------------------------


def _chat_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ConversationConflict):
        metrics.incr("chat_conflicts")
        return HTTPException(
            status_code=409,
            detail="The conversation was updated by another request. Please retry.",
        )
    if isinstance(exc, ConversationForbidden):
        return HTTPException(status_code=404, detail="Conversation not found")
    metrics.incr("chat_errors")
    log_event(
        "chat_error",
        level=logging.ERROR,
        error=type(exc).__name__,
        detail=str(exc)[:200],
    )
    return HTTPException(status_code=503, detail="AI chat is temporarily unavailable")


@router.post("/chat", response_model=ChatResponse)
def chat_post(
    body: ChatRequest,
    request: Request,
    response: Response,
    x_api_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> ChatResponse:
    with request_context(x_request_id, body.conversation_id) as rid:
        response.headers["X-Request-ID"] = rid
        owner = _guard(request, "chat", x_api_key, x_user_id)
        metrics.incr("chat_requests")
        try:
            result = get_container().chat.chat(
                message=body.message,
                conversation_id=body.conversation_id,
                limit=min(body.limit, 5),
                owner_id=owner,
            )
        except Exception as exc:
            raise _chat_error(exc) from exc
        return ChatResponse(**result, request_id=rid)


def _conversation_or_404(conversation_id: str, owner: str | None) -> dict[str, Any]:
    if not valid_conversation_id(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversation = get_container().conversations.get(conversation_id)
    if not conversation or (
        conversation.get("owner_id") and conversation.get("owner_id") != owner
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.get("/chat/{conversation_id}", response_model=ConversationView)
def chat_get(
    conversation_id: str,
    request: Request,
    x_api_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> ConversationView:
    owner = _guard(request, "search", x_api_key, x_user_id)
    conversation = _conversation_or_404(conversation_id, owner)
    return ConversationView(
        conversation_id=conversation_id,
        state=public_state(normalize_state(conversation.get("state") or {})),
        messages=jsonable_encoder(conversation.get("messages") or []),
        updated_at=conversation.get("updated_at"),
    )


@router.post("/chat/{conversation_id}/reset")
def chat_reset(
    conversation_id: str,
    request: Request,
    x_api_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> dict:
    owner = _guard(request, "chat", x_api_key, x_user_id)
    _conversation_or_404(conversation_id, owner)
    state = empty_state()
    get_container().conversations.update_state(conversation_id, state, "reset")
    metrics.incr("chat_action_reset")
    return {
        "conversation_id": conversation_id,
        "action": "reset",
        "answer": "Sure — starting a new search. What are you looking for?",
        "state": public_state(state),
    }


# ---------------------------------------------------------------------------
# Streaming (shared by WebSocket and SSE)
# ---------------------------------------------------------------------------


async def stream_turn(
    *,
    message: str,
    conversation_id: str | None,
    limit: int,
    owner: str | None,
    request_id: str,
    stop: asyncio.Event,
) -> AsyncIterator[dict[str, Any]]:
    """Yield protocol events for one chat turn.

    start -> results -> delta* -> final -> completion   (or error)
    """
    container = get_container()
    chat = container.chat
    try:
        turn: PreparedTurn = await run_in_threadpool(
            chat.prepare,
            message=message,
            conversation_id=conversation_id,
            limit=limit,
            owner_id=owner,
        )
    except Exception as exc:
        error = _chat_error(exc)
        yield {
            "type": "error",
            "request_id": request_id,
            "data": {"message": error.detail, "code": error.status_code},
        }
        return

    cid = turn.conversation_id
    set_conversation_id(cid)
    base = {"request_id": request_id, "conversation_id": cid}
    yield {**base, "type": "start"}
    preview = {k: v for k, v in turn.response.items() if k != "answer"}
    yield {**base, "type": "results", "data": jsonable_encoder(preview)}

    text: str | None = None
    stopped = False
    if turn.llm_messages:
        parts: list[str] = []
        stream = chat.llm.astream(turn.llm_messages, max_tokens=turn.max_tokens)
        try:
            async for delta in stream:
                parts.append(delta)
                yield {**base, "type": "delta", "data": {"text": delta}}
                if stop.is_set():
                    stopped = True
                    break
            text = None if stopped else "".join(parts)
        except LlmUnavailable as exc:
            chat.note_fallback(turn, exc.reason)
        finally:
            await stream.aclose()
        if stopped:
            chat.note_fallback(turn, "stopped")
    try:
        result = await run_in_threadpool(chat.finalize, turn, text)
    except Exception as exc:
        error = _chat_error(exc)
        yield {
            **base,
            "type": "error",
            "data": {"message": error.detail, "code": error.status_code},
        }
        return
    final = ChatResponse(**result, request_id=request_id)
    yield {**base, "type": "final", "data": jsonable_encoder(final)}
    yield {
        **base,
        "type": "completion",
        "data": {"status": "stopped" if stopped else "done"},
    }


def _parse_chat_payload(data: Any) -> ChatRequest:
    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")
    return ChatRequest(
        message=str(data.get("message") or ""),
        conversation_id=data.get("conversation_id") or None,
        limit=int(data.get("limit") or 5),
    )


@router.websocket("/chat/ws")
async def chat_websocket(websocket: WebSocket) -> None:
    api_key = websocket.query_params.get("api_key") or websocket.headers.get(
        "x-api-key"
    )
    if not api_key_valid(api_key):
        metrics.incr("auth_rejected")
        await websocket.close(code=4401, reason="unauthorized")
        return
    await websocket.accept()
    owner = trusted_user_id(websocket.headers.get("x-user-id"))
    host = websocket.client.host if websocket.client else None
    rate_key = f"chat:{_client_key(host, websocket.headers, owner)}"
    metrics.incr("ws_connections")

    async def send(event: dict[str, Any]) -> bool:
        try:
            await websocket.send_json(event)
            return True
        except Exception:
            return False

    while True:
        try:
            data = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except (ValueError, TypeError):
            await send(
                {
                    "type": "error",
                    "data": {"message": "Invalid JSON message", "code": 400},
                }
            )
            continue
        except Exception:
            return

        kind = data.get("type", "chat") if isinstance(data, dict) else "chat"
        if kind == "ping":
            await send({"type": "pong"})
            continue
        if kind == "stop":
            continue  # nothing in progress

        with request_context(
            data.get("request_id") if isinstance(data, dict) else None
        ) as rid:
            try:
                body = _parse_chat_payload(data)
            except (ValidationError, ValueError) as exc:
                await send(
                    {
                        "type": "error",
                        "request_id": rid,
                        "data": {
                            "message": f"Invalid request: {exc.__class__.__name__}",
                            "code": 422,
                        },
                    }
                )
                continue
            allowed, retry_after = rate_limiter.allow(
                rate_key, settings.rate_limit_chat_per_minute
            )
            if not allowed:
                metrics.incr("rate_limited")
                await send(
                    {
                        "type": "error",
                        "request_id": rid,
                        "data": {
                            "message": "Too many requests. Please slow down.",
                            "code": 429,
                            "retry_after": retry_after,
                        },
                    }
                )
                continue
            metrics.incr("chat_requests")
            metrics.incr("chat_ws_requests")

            connected = await _run_ws_turn(websocket, send, body, owner, rid)
            if not connected:
                return


async def _run_ws_turn(
    websocket: WebSocket, send, body: ChatRequest, owner: str | None, rid: str
) -> bool:
    """Stream one chat turn while listening for stop/ping. Returns False on disconnect."""
    stop = asyncio.Event()
    flags = {"disconnected": False}

    async def produce(events: AsyncIterator[dict[str, Any]]) -> None:
        async for event in events:
            if not flags["disconnected"]:
                await send(event)

    producer = asyncio.create_task(
        produce(
            stream_turn(
                message=body.message,
                conversation_id=body.conversation_id,
                limit=body.limit,
                owner=owner,
                request_id=rid,
                stop=stop,
            )
        )
    )
    while not producer.done():
        receiver = asyncio.create_task(websocket.receive_json())
        done, _ = await asyncio.wait(
            {producer, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        if receiver not in done:
            receiver.cancel()
            try:
                await receiver
            except (asyncio.CancelledError, Exception):
                pass
            continue
        try:
            incoming = receiver.result()
        except WebSocketDisconnect:
            # Client went away: stop generating but still persist the turn so
            # the conversation state matches what the user already saw.
            flags["disconnected"] = True
            stop.set()
            await producer
            return False
        except Exception:
            continue
        kind = incoming.get("type") if isinstance(incoming, dict) else None
        if kind == "stop":
            stop.set()
            metrics.incr("chat_stopped")
        elif kind == "ping":
            await send({"type": "pong"})
        else:
            await send(
                {
                    "type": "error",
                    "request_id": rid,
                    "data": {
                        "message": "A response is already in progress.",
                        "code": 409,
                    },
                }
            )
    try:
        producer.result()
    except Exception as exc:
        error = _chat_error(exc)
        await send(
            {
                "type": "error",
                "request_id": rid,
                "data": {"message": error.detail, "code": error.status_code},
            }
        )
    return True


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    x_api_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> StreamingResponse:
    """Server-Sent Events variant of the WebSocket protocol."""
    with request_context(x_request_id, body.conversation_id) as rid:
        owner = _guard(request, "chat", x_api_key, x_user_id)
    metrics.incr("chat_requests")
    stop = asyncio.Event()

    async def events() -> AsyncIterator[str]:
        with request_context(rid, body.conversation_id):
            async for event in stream_turn(
                message=body.message,
                conversation_id=body.conversation_id,
                limit=body.limit,
                owner=owner,
                request_id=rid,
                stop=stop,
            ):
                if await request.is_disconnected():
                    stop.set()
                yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": rid,
        },
    )


def register_mongo_search_routes(app) -> None:
    """Attach the AI search/chat routes to the existing V6 FastAPI application."""
    configure_logging()
    app.include_router(router)
    try:
        get_container().ensure_indexes()
    except Exception as exc:
        # Do not prevent the existing V6 API from booting if MongoDB is
        # temporarily unavailable. /search/health will report it.
        log_event("index_init_skipped", level=logging.WARNING, error=type(exc).__name__)
