"""Ollama /api/chat client for answer generation (Qwen3 8B on a T4).

* Pooled HTTP connections (httpx) with keep_alive so the model stays loaded.
* ``think: false`` + low temperature + bounded num_ctx/num_predict.
* Streaming (async) and non-streaming (sync) calls.
* Circuit breaker: after N consecutive failures the LLM is skipped for a
  cool-down period and callers answer deterministically. Every timeout,
  failure and fallback is logged and counted - never silently hidden.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx

from .config import settings
from .observability import add_timing, log_event, metrics


class LlmUnavailable(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class LlmResult:
    text: str
    duration_ms: float
    model_ms: float | None = None
    eval_tokens: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    def __init__(self, threshold: int, reset_seconds: float) -> None:
        self.threshold = max(1, threshold)
        self.reset_seconds = reset_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if time.monotonic() - self._opened_at >= self.reset_seconds:
                # Half-open: let one request probe the model.
                self._opened_at = None
                self._failures = self.threshold - 1
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold and self._opened_at is None:
                self._opened_at = time.monotonic()
                metrics.incr("llm_circuit_open")
                log_event(
                    "llm_circuit_open",
                    level=logging.ERROR,
                    consecutive_failures=self._failures,
                    cooldown_s=self.reset_seconds,
                )

    @property
    def state(self) -> str:
        return "open" if self._opened_at is not None else "closed"


class ThinkFilter:
    """Remove <think>...</think> blocks from a token stream."""

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        out: list[str] = []
        while self._buffer:
            if self._inside:
                end = self._buffer.find("</think>")
                if end < 0:
                    # Keep a tail in case the closing tag is split.
                    self._buffer = self._buffer[-8:]
                    return "".join(out)
                self._buffer = self._buffer[end + len("</think>") :]
                self._inside = False
                continue
            start = self._buffer.find("<think>")
            if start < 0:
                # Hold back a possible partial "<think" prefix.
                safe = len(self._buffer)
                tail = self._buffer.rfind("<")
                if tail >= 0 and "<think>".startswith(self._buffer[tail:]):
                    safe = tail
                out.append(self._buffer[:safe])
                self._buffer = self._buffer[safe:]
                return "".join(out)
            out.append(self._buffer[:start])
            self._buffer = self._buffer[start + len("<think>") :]
            self._inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._inside else self._buffer
        self._buffer = ""
        return rest


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


class OllamaChatClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        enabled: bool | None = None,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (
            settings.ollama_base_url if base_url is None else base_url
        ).rstrip("/")
        self.model = settings.ollama_chat_model if model is None else model
        self.timeout_seconds = (
            settings.ollama_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self._enabled = settings.chat_llm_enabled if enabled is None else enabled
        self.breaker = CircuitBreaker(
            settings.ollama_failure_threshold, settings.ollama_circuit_reset_seconds
        )
        self._transport = transport
        self._async_transport = async_transport
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self.model and self.base_url)

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            self.timeout_seconds,
            connect=min(settings.ollama_connect_timeout_seconds, self.timeout_seconds),
        )

    def _sync_client(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self._timeout(),
                    limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
                    transport=self._transport,
                )
            return self._client

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def payload(
        self, messages: list[dict[str, str]], *, max_tokens: int, stream: bool
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "stream": stream,
            "think": False,
            "keep_alive": settings.ollama_keep_alive,
            "messages": messages,
            "options": {
                "temperature": settings.ollama_temperature,
                "top_p": 0.9,
                "num_predict": max_tokens,
                "num_ctx": settings.ollama_num_ctx,
            },
        }

    def _precheck(self) -> None:
        if not self.enabled:
            raise LlmUnavailable("disabled")
        if not self.breaker.allow():
            metrics.incr("llm_skipped_circuit_open")
            raise LlmUnavailable("circuit_open")

    def _record_failure(
        self, exc: Exception, started: float, kind: str
    ) -> LlmUnavailable:
        duration = (time.perf_counter() - started) * 1000
        timed_out = isinstance(exc, httpx.TimeoutException)
        reason = (
            "timeout"
            if timed_out
            else (
                "connect_error"
                if isinstance(exc, httpx.ConnectError)
                else type(exc).__name__
            )
        )
        metrics.incr("llm_timeout" if timed_out else "llm_error")
        self.breaker.failure()
        add_timing("llm_ms", duration)
        log_event(
            "llm_failure",
            level=logging.WARNING,
            kind=kind,
            reason=reason,
            duration_ms=round(duration, 1),
            timeout_s=self.timeout_seconds,
            breaker=self.breaker.state,
        )
        return LlmUnavailable(reason)

    def _record_success(
        self, started: float, data: dict[str, Any], kind: str, chars: int
    ) -> LlmResult:
        duration = (time.perf_counter() - started) * 1000
        model_ms = data.get("total_duration")
        model_ms = (
            round(model_ms / 1e6, 1) if isinstance(model_ms, (int, float)) else None
        )
        self.breaker.success()
        metrics.incr("llm_success")
        add_timing("llm_ms", duration)
        if model_ms is not None:
            metrics.observe("ollama_model_ms", model_ms)
        log_event(
            "llm_complete",
            kind=kind,
            duration_ms=round(duration, 1),
            ollama_ms=model_ms,
            eval_tokens=data.get("eval_count"),
            prompt_tokens=data.get("prompt_eval_count"),
            output_chars=chars,
        )
        return LlmResult(
            text="",
            duration_ms=duration,
            model_ms=model_ms,
            eval_tokens=data.get("eval_count"),
        )

    # ------------------------------------------------------------------
    def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> LlmResult:
        self._precheck()
        started = time.perf_counter()
        try:
            response = self._sync_client().post(
                "/api/chat",
                json=self.payload(messages, max_tokens=max_tokens, stream=False),
            )
            response.raise_for_status()
            data = response.json()
            text = strip_think(str((data.get("message") or {}).get("content") or ""))
        except Exception as exc:
            raise self._record_failure(exc, started, "complete") from exc
        result = self._record_success(started, data, "complete", len(text))
        result.text = text
        return result

    async def astream(
        self, messages: list[dict[str, str]], *, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """Yield answer text deltas. Raises LlmUnavailable on failure."""
        self._precheck()
        started = time.perf_counter()
        think = ThinkFilter()
        chars = 0
        final: dict[str, Any] = {}
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout(),
                transport=self._async_transport,
            ) as client:
                async with client.stream(
                    "POST",
                    "/api/chat",
                    json=self.payload(messages, max_tokens=max_tokens, stream=True),
                ) as response:
                    response.raise_for_status()
                    deadline = started + self.timeout_seconds
                    async for line in response.aiter_lines():
                        if time.perf_counter() > deadline:
                            raise httpx.ReadTimeout(
                                "stream exceeded OLLAMA_TIMEOUT_SECONDS"
                            )
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("error"):
                            raise RuntimeError(str(data["error"])[:200])
                        delta = str((data.get("message") or {}).get("content") or "")
                        visible = think.feed(delta)
                        if visible:
                            chars += len(visible)
                            yield visible
                        if data.get("done"):
                            final = data
                            break
            tail = think.flush()
            if tail:
                chars += len(tail)
                yield tail
        except LlmUnavailable:
            raise
        except GeneratorExit:
            # Consumer stopped (user pressed Stop / disconnected).
            metrics.incr("llm_cancelled")
            log_event(
                "llm_cancelled",
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            raise
        except Exception as exc:
            raise self._record_failure(exc, started, "stream") from exc
        self._record_success(started, final, "stream", chars)
