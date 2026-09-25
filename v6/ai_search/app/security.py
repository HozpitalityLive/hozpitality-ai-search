"""Request hardening helpers: auth, rate limiting, URL safety, LLM data hygiene."""

from __future__ import annotations

import hmac
import re
import threading
import time
from collections import deque
from urllib.parse import urlsplit

from .config import settings

# ---------------------------------------------------------------------------
# Authentication (optional, backwards compatible)
# ---------------------------------------------------------------------------


def api_key_valid(candidate: str | None, keys: tuple[str, ...] | None = None) -> bool:
    """Return True when auth is disabled or the candidate matches a key."""
    configured = settings.api_keys if keys is None else keys
    if not configured:
        return True
    if not candidate:
        return False
    return any(hmac.compare_digest(candidate, key) for key in configured)


_USER_ID_RE = re.compile(r"^[A-Za-z0-9._@:-]{1,128}$")


def trusted_user_id(header_value: str | None) -> str | None:
    """X-User-Id is only honoured behind a trusted auth proxy."""
    if not settings.trust_user_header or not header_value:
        return None
    value = header_value.strip()
    return value if _USER_ID_RE.match(value) else None


CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def valid_conversation_id(value: str | None) -> bool:
    return bool(value and CONVERSATION_ID_RE.match(value))


# ---------------------------------------------------------------------------
# Rate limiting (per process sliding window; pair with nginx limit_req)
# ---------------------------------------------------------------------------


class RateLimiter:
    def __init__(self, max_keys: int = 50_000) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}
        self._max_keys = max_keys

    def allow(
        self, key: str, limit: int, window_seconds: float = 60.0
    ) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        if limit <= 0:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self._max_keys:
                    # Bounded memory: drop the stalest keys.
                    for stale in list(self._hits)[: self._max_keys // 10]:
                        self._hits.pop(stale, None)
                hits = self._hits[key] = deque()
            while hits and now - hits[0] >= window_seconds:
                hits.popleft()
            if len(hits) >= limit:
                return False, round(window_seconds - (now - hits[0]), 1)
            hits.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------


def safe_url(value: object, base_url: str | None = None) -> str | None:
    """Return an http(s) URL taken from real data, or None.

    Site-relative paths ("/jobs/1") are absolutized with the configured public
    base URL when present. Bare slugs, javascript:, data: and other schemes are
    rejected: we never guess a URL.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if (
        not candidate
        or len(candidate) > 2048
        or any(ch in candidate for ch in "\r\n\t <>\"'`")
    ):
        return None
    base = settings.public_site_base_url if base_url is None else base_url
    if candidate.startswith("//"):
        candidate = "https:" + candidate
    elif candidate.startswith("/"):
        if not base:
            return candidate
        candidate = base.rstrip("/") + candidate
    parts = urlsplit(candidate)
    if parts.scheme.casefold() not in {"http", "https"} or not parts.netloc:
        return None
    return candidate


# ---------------------------------------------------------------------------
# LLM data hygiene (prompt-injection defence in depth)
# ---------------------------------------------------------------------------

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ROLE_MARKER_RE = re.compile(
    r"<\|?/?(?:im_start|im_end|system|assistant|user|endoftext|think)\|?>"
    r"|</?\s*(?:search_data|system|instructions?|tool_call|think)\s*>"
    r"|^\s*(?:system|assistant|developer)\s*:",
    re.I | re.M,
)
_INJECTION_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:previous|prior|above|all|earlier|system)\b"
    r"[^.\n]{0,40}\b(?:instructions?|prompts?|rules?|messages?)\b"
    r"|\byou\s+are\s+now\b"
    r"|\bnew\s+instructions?\s*:"
    r"|\b(?:reveal|print|show)\b[^.\n]{0,30}\b(?:system\s+prompt|instructions|api\s*keys?|passwords?)\b",
    re.I,
)
_HTML_TAG_RE = re.compile(r"<[^>]{1,300}>")


def clean_untrusted_text(value: object, max_chars: int = 400) -> str | None:
    """Prepare retrieved document text for inclusion in an LLM prompt.

    Retrieved content is data, not instructions: strip markup, role markers
    and common injection phrasing, collapse whitespace and truncate.
    """
    if value is None:
        return None
    text = str(value)
    text = _CONTROL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _ROLE_MARKER_RE.sub(" ", text)
    text = _INJECTION_RE.sub("[removed]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def contains_injection(value: str) -> bool:
    return bool(
        _INJECTION_RE.search(value or "") or _ROLE_MARKER_RE.search(value or "")
    )


_URL_RE = re.compile(r"https?://[^\s)\]>\"']+", re.I)


def strip_unknown_urls(text: str, allowed: set[str]) -> str:
    """Remove any URL from model output that was not present in the data."""

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        url = raw.rstrip(".,;:")
        # Keep sentence punctuation that the URL pattern swallowed.
        return raw if url in allowed else raw[len(url) :]

    return _URL_RE.sub(replace, text)
