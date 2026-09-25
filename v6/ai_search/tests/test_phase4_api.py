"""API contract tests: HTTP, WebSocket streaming and SSE."""

from __future__ import annotations

import json

import httpx
import pytest
from starlette.websockets import WebSocketDisconnect

from ai_search.app.chat_service import ChatService
from ai_search.app.ollama_client import OllamaChatClient

CID = "conversation-api-0001"


def stream_llm(container, chunks):
    def handler(request):
        body = json.loads(request.content)
        if body.get("stream"):
            lines = [{"message": {"content": c}, "done": False} for c in chunks]
            lines.append({"message": {"content": ""}, "done": True})
            return httpx.Response(
                200, content=("\n".join(json.dumps(x) for x in lines) + "\n").encode()
            )
        return httpx.Response(
            200, json={"message": {"content": "".join(chunks)}, "done": True}
        )

    transport = httpx.MockTransport(handler)
    container.llm = OllamaChatClient(
        base_url="http://ollama.test",
        model="qwen3:8b",
        timeout_seconds=5,
        enabled=True,
        transport=transport,
        async_transport=transport,
    )
    container.chat = ChatService(
        container.search, container.conversations, container.llm
    )


def test_health(client, monkeypatch):
    import ai_search.app.main as main_module
    import ai_search.app.router as router_module

    monkeypatch.setattr(main_module, "ping", lambda: True)
    monkeypatch.setattr(router_module, "ping", lambda: True)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["llm"]["model"]
    body = client.get("/search/health").json()
    assert body["documents"] > 0 and body["llm_circuit"] == "closed"

    def down():
        raise RuntimeError("mongodb://user:secret@host refused")

    monkeypatch.setattr(main_module, "ping", down)
    monkeypatch.setattr(router_module, "ping", down)
    for path in ("/health", "/search/health"):
        r = client.get(path)
        assert r.status_code == 503 and "secret" not in r.text


def test_search_get_and_post_contract(client):
    r = client.get("/search", params={"q": "chef jobs in Dubai"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {
        "query",
        "corrected_query",
        "total",
        "results",
        "related_results",
        "understanding",
    }
    assert body["total"] == 5
    first = body["results"][0]
    assert set(first) >= {
        "entity_type",
        "entity_id",
        "title",
        "url",
        "score",
        "matched_by",
        "location",
    }
    assert r.headers["X-Request-ID"]

    r = client.post(
        "/search", json={"q": "hotel companies", "entity": "company", "city": "Dubai"}
    )
    assert r.status_code == 200
    assert {x["entity_type"] for x in r.json()["results"]} == {"company"}


def test_search_validation(client):
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "chef", "limit": 50}).status_code == 422
    assert (
        client.post("/search", json={"q": "chef", "unexpected": 1}).status_code == 422
    )


def test_chat_http_flow_and_contract(client):
    r1 = client.post(
        "/chat", json={"message": "Find chef jobs in Dubai", "conversation_id": CID}
    )
    assert r1.status_code == 200
    body = r1.json()
    for key in (
        "conversation_id",
        "action",
        "answer",
        "results",
        "related_results",
        "understanding",
        "state",
        "references",
        "suggestions",
        "request_id",
    ):
        assert key in body
    assert body["conversation_id"] == CID
    assert body["action"] == "search"
    assert [ref["number"] for ref in body["references"]] == [1, 2, 3, 4, 5]
    assert body["references"][0]["url"] == body["results"][0]["url"]

    r2 = client.post(
        "/chat", json={"message": "Only management positions", "conversation_id": CID}
    )
    assert r2.json()["state"]["entity"] == "job"

    r3 = client.post(
        "/chat", json={"message": "Compare the first three", "conversation_id": CID}
    )
    comparison = r3.json()["comparison"]
    assert comparison["columns"][0] == "Field" and len(comparison["columns"]) == 4
    assert all(len(row) == 4 for row in comparison["rows"])


def test_chat_without_conversation_id_creates_one(client):
    r = client.post("/chat", json={"message": "hello"})
    assert r.status_code == 200
    assert len(r.json()["conversation_id"]) >= 8
    assert r.json()["action"] == "smalltalk"


def test_get_conversation_and_reset(client):
    client.post(
        "/chat", json={"message": "Find chef jobs in Dubai", "conversation_id": CID}
    )
    view = client.get(f"/chat/{CID}")
    assert view.status_code == 200
    data = view.json()
    assert data["state"]["entity"] == "job"
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]

    reset = client.post(f"/chat/{CID}/reset")
    assert reset.status_code == 200
    assert reset.json()["state"]["entity"] is None
    assert client.get(f"/chat/{CID}").json()["state"]["entity"] is None


def test_unknown_conversation_is_404(client):
    assert client.get("/chat/does-not-exist-123").status_code == 404
    assert client.post("/chat/does-not-exist-123/reset").status_code == 404
    assert client.get("/chat/bad id!").status_code == 404


def test_chat_validation(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 2001}).status_code == 422
    assert (
        client.post(
            "/chat", json={"message": "hi", "conversation_id": "../../etc"}
        ).status_code
        == 422
    )
    r = client.post("/chat", json={"message": "hi", "extra": True})
    assert r.status_code == 422
    assert "hi" not in r.text  # the submitted body is not echoed back


def test_websocket_streaming_protocol(client, container):
    stream_llm(container, ["I found ", "5 chef jobs ", "in Dubai."])
    with client.websocket_connect("/chat/ws") as ws:
        ws.send_json(
            {
                "message": "Find chef jobs in Dubai",
                "conversation_id": CID,
                "request_id": "req-000001",
            }
        )
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in {"completion", "error"}:
                break
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[1] == "results"
    assert types[-2:] == ["final", "completion"]
    assert types.count("delta") == 3
    assert all(e["request_id"] == "req-000001" for e in events)
    results_event = events[1]["data"]
    assert len(results_event["results"]) == 5 and "answer" not in results_event
    final = events[-2]["data"]
    assert final["answer"] == "I found 5 chef jobs in Dubai."
    assert final["conversation_id"] == CID
    assert events[-1]["data"]["status"] == "done"


def test_websocket_multiple_turns_on_one_socket(client):
    with client.websocket_connect("/chat/ws") as ws:
        for message in ("Find chef jobs in Dubai", "Only management positions"):
            ws.send_json({"message": message, "conversation_id": CID})
            while True:
                event = ws.receive_json()
                if event["type"] == "final":
                    final = event["data"]
                if event["type"] in {"completion", "error"}:
                    break
    assert final["state"]["entity"] == "job"
    assert final["state"]["filters"]["level"] == "manager"


def test_websocket_stop_generation_persists_turn(client, container):
    stream_llm(container, ["word "] * 50)
    with client.websocket_connect("/chat/ws") as ws:
        ws.send_json({"message": "Find chef jobs in Dubai", "conversation_id": CID})
        assert ws.receive_json()["type"] == "start"
        assert ws.receive_json()["type"] == "results"
        ws.send_json({"type": "stop"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in {"completion", "error"}:
                break
    assert events[-1]["type"] == "completion"
    assert events[-1]["data"]["status"] in {"stopped", "done"}
    final = [e for e in events if e["type"] == "final"][0]["data"]
    if events[-1]["data"]["status"] == "stopped":
        assert (
            final["answer"] == "I found 5 chef jobs in Dubai."
        )  # deterministic, complete answer
    assert client.get(f"/chat/{CID}").json()["state"]["entity"] == "job"


def test_websocket_invalid_messages(client):
    with client.websocket_connect("/chat/ws") as ws:
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"message": ""})
        err = ws.receive_json()
        assert err["type"] == "error" and err["data"]["code"] == 422
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


def test_websocket_accepts_legacy_frontend_payload(client):
    with client.websocket_connect("/chat/ws") as ws:
        ws.send_json(
            {
                "message": "Find chef jobs in Dubai",
                "conversation_id": "conv-1727300000-abc1234",
                "request_id": "req-1727300000-xyz",
                "metadata": {"client": "hozpitality-web"},
            }
        )
        types = []
        while True:
            event = ws.receive_json()
            types.append(event["type"])
            if event["type"] in {"completion", "error"}:
                break
    assert types[-1] == "completion"


def test_sse_stream(client):
    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "Find chef jobs in Dubai", "conversation_id": CID},
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")
        ]
    assert [e["type"] for e in events][-2:] == ["final", "completion"]


def test_api_key_enforced_when_configured(client):
    from ai_search.app import security

    # Settings is a frozen dataclass; tests toggle it explicitly and restore it.
    object.__setattr__(security.settings, "api_keys", ("secret-key-1",))
    try:
        assert client.post("/chat", json={"message": "hi"}).status_code == 401
        assert (
            client.post(
                "/chat", json={"message": "hi"}, headers={"X-API-Key": "wrong"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/chat", json={"message": "hi"}, headers={"X-API-Key": "secret-key-1"}
            ).status_code
            == 200
        )
        assert client.get("/search", params={"q": "chef"}).status_code == 401
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/chat/ws") as ws:
                ws.receive_json()
        with client.websocket_connect("/chat/ws?api_key=secret-key-1") as ws:
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"
    finally:
        object.__setattr__(security.settings, "api_keys", ())


def test_rate_limit(client):
    from ai_search.app import router as router_module

    original = router_module.settings.rate_limit_chat_per_minute
    object.__setattr__(router_module.settings, "rate_limit_chat_per_minute", 2)
    try:
        codes = [
            client.post("/chat", json={"message": "hello"}).status_code
            for _ in range(4)
        ]
        assert codes[:2] == [200, 200]
        assert codes[2:] == [429, 429]
        r = client.post("/chat", json={"message": "hello"})
        assert int(r.headers["Retry-After"]) >= 1
    finally:
        object.__setattr__(
            router_module.settings, "rate_limit_chat_per_minute", original
        )


def test_metrics_endpoint(client):
    client.post("/chat", json={"message": "Find chef jobs in Dubai"})
    data = client.get("/metrics").json()
    assert data["counters"]["chat_turns"] >= 1
    assert "search_ms" in data["latency_ms"] and "mongo_ms" in data["latency_ms"]


def test_mongo_failure_returns_503_without_leaking_details(
    client, container, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("mongodb://admin:hunter2@db:27017 connection refused")

    monkeypatch.setattr(container.conversations, "ensure", boom)
    r = client.post("/chat", json={"message": "Find chef jobs in Dubai"})
    assert r.status_code == 503
    assert "hunter2" not in r.text and "mongodb://" not in r.text
