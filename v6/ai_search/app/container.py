"""Service wiring. Built lazily so importing the router never touches MongoDB,
and tests can install a container backed by an in-memory collection."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .chat_service import ChatService
from .config import settings
from .conversation import ConversationRepository
from .db import get_collection
from .ollama_client import OllamaChatClient
from .repository import SearchDocumentsRepository
from .service import SearchService


@dataclass
class Container:
    repository: SearchDocumentsRepository
    search: SearchService
    conversations: ConversationRepository
    chat: ChatService
    llm: OllamaChatClient

    def ensure_indexes(self) -> None:
        self.repository.ensure_indexes()
        self.conversations.ensure_indexes()


def build_container(
    *,
    search_collection=None,
    chat_collection=None,
    llm: OllamaChatClient | None = None,
    search_service: SearchService | None = None,
) -> Container:
    repository = SearchDocumentsRepository(
        search_collection if search_collection is not None else get_collection()
    )
    search = search_service or SearchService(
        repository, fuzzy_threshold=settings.fuzzy_threshold
    )
    conversations = ConversationRepository(
        chat_collection
        if chat_collection is not None
        else get_collection(settings.chat_collection)
    )
    llm = llm or OllamaChatClient()
    chat = ChatService(search, conversations, llm)
    return Container(
        repository=repository,
        search=search,
        conversations=conversations,
        chat=chat,
        llm=llm,
    )


_container: Container | None = None
_lock = threading.Lock()


def get_container() -> Container:
    global _container
    if _container is None:
        with _lock:
            if _container is None:
                _container = build_container()
    return _container


def set_container(container: Container | None) -> None:
    global _container
    with _lock:
        _container = container
