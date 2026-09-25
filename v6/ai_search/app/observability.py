"""Structured logging, request correlation and lightweight in-process metrics.

Logs are single-line JSON so they can be shipped as-is to journald / Loki /
CloudWatch. Message text is never logged by default; only sizes, actions,
counts and latencies. Credentials never pass through this module.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator
from uuid import uuid4

from .config import settings

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
conversation_id_var: ContextVar[str | None] = ContextVar(
    "conversation_id", default=None
)
_timings_var: ContextVar[dict[str, float] | None] = ContextVar("timings", default=None)

logger = logging.getLogger("ai_search")

_SECRET_PATTERNS = [
    # mongodb://user:password@host -> mongodb://***@host
    (re.compile(r"(mongodb(?:\+srv)?://)[^@/\s]+@", re.I), r"\1***@"),
    (
        re.compile(r"(api[_-]?key|token|password|secret)(\s*[=:]\s*)\S+", re.I),
        r"\1\2***",
    ),
]


def redact(value: str) -> str:
    for pattern, replacement in _SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        rid = request_id_var.get()
        cid = conversation_id_var.get()
        if rid:
            payload["request_id"] = rid
        if cid:
            payload["conversation_id"] = cid
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                payload[key] = redact(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["error"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    logger.addHandler(handler)
    logger.setLevel(settings.log_level.upper())
    logger.propagate = False
    _configured = True


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": {"event": event, **fields}})


def new_request_id(candidate: str | None = None) -> str:
    if candidate and re.fullmatch(r"[A-Za-z0-9._:-]{6,128}", candidate):
        return candidate
    return uuid4().hex


@contextmanager
def request_context(
    request_id: str | None = None, conversation_id: str | None = None
) -> Iterator[str]:
    rid = new_request_id(request_id)
    token_r = request_id_var.set(rid)
    token_c = conversation_id_var.set(conversation_id)
    token_t = _timings_var.set({})
    try:
        yield rid
    finally:
        request_id_var.reset(token_r)
        conversation_id_var.reset(token_c)
        _timings_var.reset(token_t)


def set_conversation_id(conversation_id: str | None) -> None:
    conversation_id_var.set(conversation_id)


def add_timing(name: str, milliseconds: float) -> None:
    timings = _timings_var.get()
    if timings is not None:
        timings[name] = round(timings.get(name, 0.0) + milliseconds, 2)
    metrics.observe(name, milliseconds)


def ensure_timings() -> None:
    """Start collecting timings when called outside an HTTP request context."""
    if _timings_var.get() is None:
        _timings_var.set({})


def current_timings() -> dict[str, float]:
    return dict(_timings_var.get() or {})


@contextmanager
def timed(name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        add_timing(name, (time.perf_counter() - start) * 1000)


class Metrics:
    """Thread-safe counters and latency aggregates for /metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._latency: dict[str, list[float]] = {}

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe(self, name: str, milliseconds: float) -> None:
        with self._lock:
            agg = self._latency.setdefault(name, [0.0, 0.0, 0.0])  # count, total, max
            agg[0] += 1
            agg[1] += milliseconds
            agg[2] = max(agg[2], milliseconds)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "latency_ms": {
                    name: {
                        "count": int(agg[0]),
                        "avg": round(agg[1] / agg[0], 2) if agg[0] else 0.0,
                        "max": round(agg[2], 2),
                    }
                    for name, agg in self._latency.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latency.clear()


metrics = Metrics()
