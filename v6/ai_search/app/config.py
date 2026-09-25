from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


AI_SEARCH_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = AI_SEARCH_ROOT / ".env"

load_dotenv(ENV_FILE)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _list(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


URL_TEMPLATE_ENTITIES = (
    "job",
    "professional",
    "company",
    "product",
    "article",
    "event",
    "award",
    "faq",
)


def _url_templates() -> dict[str, str]:
    templates: dict[str, str] = {}
    raw = os.getenv("PUBLIC_URL_TEMPLATES", "").strip()
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("PUBLIC_URL_TEMPLATES must be a JSON object")
        templates.update({str(k): str(v) for k, v in parsed.items() if v})
    for entity in URL_TEMPLATE_ENTITIES:
        value = os.getenv(f"URL_TEMPLATE_{entity.upper()}", "").strip()
        if value:
            templates[entity] = value
    for entity, template in templates.items():
        if entity not in URL_TEMPLATE_ENTITIES or not template.startswith(
            ("https://", "http://")
        ):
            raise ValueError(
                f"Invalid URL template for {entity!r}: must be an absolute http(s) URL"
            )
    return templates


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str = os.getenv(
        "MONGODB_URI",
        "mongodb://127.0.0.1:27017",
    )

    mongodb_database: str = os.getenv(
        "MONGODB_DATABASE",
        "mongoAdmin",
    )

    mongodb_collection: str = os.getenv(
        "MONGODB_COLLECTION",
        "search_documents",
    )

    chat_collection: str = os.getenv(
        "CHAT_COLLECTION",
        "ai_search_conversations",
    )

    mongodb_max_pool_size: int = _int(
        "MONGODB_MAX_POOL_SIZE",
        100,
    )

    mongodb_min_pool_size: int = _int(
        "MONGODB_MIN_POOL_SIZE",
        5,
    )

    mongodb_server_selection_timeout_ms: int = _int(
        "MONGODB_SERVER_SELECTION_TIMEOUT_MS",
        5000,
    )

    mongodb_connect_timeout_ms: int = _int(
        "MONGODB_CONNECT_TIMEOUT_MS",
        5000,
    )

    # Bounds every individual MongoDB query so a pathological regex can never
    # hold a worker thread indefinitely.
    mongodb_max_time_ms: int = _int("MONGODB_MAX_TIME_MS", 4000)

    max_results: int = min(
        _int("SEARCH_MAX_RESULTS", 5),
        5,
    )

    fuzzy_threshold: int = _int(
        "SEARCH_FUZZY_THRESHOLD",
        82,
    )

    fuzzy_min_token_length: int = _int(
        "SEARCH_FUZZY_MIN_TOKEN_LENGTH",
        3,
    )

    # Optional base URL used to absolutize site-relative result links such as
    # "/jobs/123". Slugs are never turned into URLs.
    public_site_base_url: str = os.getenv("PUBLIC_SITE_BASE_URL", "").rstrip("/")

    # Public page routes per module, e.g. {"job": "https://www.hozpitality.com/<route>/{slug}"}.
    # The migration stores slugs (and ids) but not the site's route patterns,
    # so they must be configured; placeholders: {slug}, {id}. A module without
    # a template gets url=null (never guessed). Awards use their real
    # links.detail URL regardless.
    url_templates: dict[str, str] = field(default_factory=lambda: _url_templates())

    # --- Ollama / Qwen3 -----------------------------------------------------
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_query_model: str = os.getenv("OLLAMA_QUERY_MODEL", "qwen3:8b")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
    # Chat answer generation timeout (whole request, including streaming).
    ollama_timeout_seconds: float = _float("OLLAMA_TIMEOUT_SECONDS", 60)
    # Query-understanding fallback is on the /search hot path, so it gets a
    # much shorter budget than answer generation.
    ollama_query_timeout_seconds: float = _float("OLLAMA_QUERY_TIMEOUT_SECONDS", 8)
    ollama_connect_timeout_seconds: float = _float("OLLAMA_CONNECT_TIMEOUT_SECONDS", 3)
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    ollama_num_ctx: int = _int("OLLAMA_NUM_CTX", 4096)
    ollama_temperature: float = _float("OLLAMA_TEMPERATURE", 0.1)
    ollama_max_answer_tokens: int = _int("OLLAMA_MAX_ANSWER_TOKENS", 320)
    ollama_max_compare_tokens: int = _int("OLLAMA_MAX_COMPARE_TOKENS", 480)
    # Circuit breaker: after N consecutive failures, skip the LLM for a while
    # and answer deterministically instead of making every user wait.
    ollama_failure_threshold: int = _int("OLLAMA_FAILURE_THRESHOLD", 3)
    ollama_circuit_reset_seconds: float = _float("OLLAMA_CIRCUIT_RESET_SECONDS", 30)
    chat_llm_enabled: bool = _bool("CHAT_LLM_ENABLED", True)

    # --- Semantic search ------------------------------------------------------
    semantic_search_enabled: bool = (
        os.getenv("SEMANTIC_SEARCH_ENABLED", "false").casefold() == "true"
    )
    semantic_model: str = os.getenv(
        "SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings_collection: str = os.getenv(
        "EMBEDDINGS_COLLECTION", "ai_search_embeddings"
    )

    # --- Conversation memory ---------------------------------------------------
    chat_ttl_days: int = _int("CHAT_TTL_DAYS", 30)
    chat_max_messages: int = _int("CHAT_MAX_MESSAGES", 20)
    chat_max_history: int = _int("CHAT_MAX_RESULT_HISTORY", 50)
    chat_max_shown: int = _int("CHAT_MAX_SHOWN_IDS", 100)
    chat_max_message_chars: int = _int("CHAT_MAX_MESSAGE_CHARS", 1000)

    # --- Production hardening -------------------------------------------------
    # Comma-separated API keys. Empty = open API (backwards compatible).
    api_keys: tuple[str, ...] = field(
        default_factory=lambda: _list("AI_SEARCH_API_KEYS")
    )
    # Only trust X-User-Id when a trusted proxy/auth gateway sets it.
    trust_user_header: bool = _bool("TRUST_USER_HEADER", False)
    rate_limit_chat_per_minute: int = _int("RATE_LIMIT_CHAT_PER_MINUTE", 30)
    rate_limit_search_per_minute: int = _int("RATE_LIMIT_SEARCH_PER_MINUTE", 120)
    cors_allow_origins: tuple[str, ...] = field(
        default_factory=lambda: _list(
            "CORS_ALLOW_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        )
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_json: bool = _bool("LOG_JSON", True)


settings = Settings()
