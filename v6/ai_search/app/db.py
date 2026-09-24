from __future__ import annotations

from pymongo import MongoClient
from pymongo.collection import Collection

from .config import settings

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(
            settings.mongodb_uri,
            maxPoolSize=settings.mongodb_max_pool_size,
            minPoolSize=settings.mongodb_min_pool_size,
            serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
            connectTimeoutMS=settings.mongodb_connect_timeout_ms,
            retryWrites=True,
            appname="hozpitality-ai-search-v1",
        )
    return _client


def get_collection() -> Collection:
    return get_client()[settings.mongodb_database][settings.mongodb_collection]


def ping() -> bool:
    get_client().admin.command("ping")
    return True


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
