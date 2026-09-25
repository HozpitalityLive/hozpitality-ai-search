"""Test configuration.

Unit and integration tests never touch a real MongoDB or Ollama:
* MongoDB  -> mongomock collections seeded with realistic search_documents.
* Ollama   -> disabled by default; tests that need it inject httpx.MockTransport.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must run before any ai_search module reads settings.
os.environ.setdefault("OLLAMA_BASE_URL", "")
os.environ.setdefault("CHAT_LLM_ENABLED", "false")
os.environ.setdefault("SEMANTIC_SEARCH_ENABLED", "false")
os.environ.setdefault("MONGODB_URI", "mongodb://127.0.0.1:1/")
os.environ.setdefault("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "100")
os.environ.setdefault("MONGODB_CONNECT_TIMEOUT_MS", "100")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("RATE_LIMIT_CHAT_PER_MINUTE", "1000")
os.environ.setdefault("RATE_LIMIT_SEARCH_PER_MINUTE", "1000")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mongomock  # noqa: E402
import pytest  # noqa: E402

from ai_search.tests.fixtures_data import seed_documents  # noqa: E402


@pytest.fixture()
def mongo_db():
    client = mongomock.MongoClient()
    db = client["mongoAdmin"]
    db["search_documents"].insert_many(seed_documents())
    db["search_documents"].create_index(
        [("ai_search_text", "text")], name="idx_ai_search_text"
    )
    return db


@pytest.fixture()
def container(mongo_db):
    from ai_search.app.container import build_container, set_container
    from ai_search.app.ollama_client import OllamaChatClient
    from ai_search.app.observability import metrics
    from ai_search.app.security import rate_limiter

    metrics.reset()
    rate_limiter.reset()
    built = build_container(
        search_collection=mongo_db["search_documents"],
        chat_collection=mongo_db["ai_search_conversations"],
        llm=OllamaChatClient(enabled=False),
    )
    built.ensure_indexes()
    set_container(built)
    yield built
    set_container(None)


@pytest.fixture()
def chat(container):
    return container.chat


@pytest.fixture()
def client(container):
    from fastapi.testclient import TestClient

    from ai_search.app.main import app

    with TestClient(app) as test_client:
        yield test_client
