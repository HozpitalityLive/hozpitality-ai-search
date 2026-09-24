from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


AI_SEARCH_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = AI_SEARCH_ROOT / ".env"

load_dotenv(ENV_FILE)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


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

    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_timeout_seconds: float = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "4"))
    semantic_search_enabled: bool = os.getenv("SEMANTIC_SEARCH_ENABLED", "false").casefold() == "true"
    semantic_model: str = os.getenv("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


settings = Settings()