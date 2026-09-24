from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo import ASCENDING

from .db import get_collection
from .config import settings


class ConversationRepository:
    """MongoDB-backed conversation state for the Phase 3 chat layer."""

    def __init__(self):
        self.collection = get_collection(settings.chat_collection)

    def ensure_indexes(self) -> None:
        self.collection.create_index(
            [("conversation_id", ASCENDING)],
            unique=True,
            name="idx_chat_conversation_id",
        )
        self.collection.create_index(
            [("updated_at", -1)],
            name="idx_chat_updated_at",
        )

    def create(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        conversation_id = uuid4().hex
        document = {
            "conversation_id": conversation_id,
            "state": state or {},
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        self.collection.insert_one(document)
        return document

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        return self.collection.find_one({"conversation_id": conversation_id})

    def ensure(self, conversation_id: str | None) -> dict[str, Any]:
        if conversation_id:
            existing = self.get(conversation_id)
            if existing:
                return existing
        return self.create()

    def save_turn(
        self,
        conversation_id: str,
        *,
        user_message: str,
        assistant_message: str,
        state: dict[str, Any],
        action: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        self.collection.update_one(
            {"conversation_id": conversation_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {
                                "role": "user",
                                "content": user_message,
                                "created_at": now,
                            },
                            {
                                "role": "assistant",
                                "content": assistant_message,
                                "created_at": now,
                            },
                        ],
                        "$slice": -20,
                    }
                },
                "$set": {
                    "state": state,
                    "updated_at": now,
                    "last_action": action,
                },
            },
        )

    def update_state(self, conversation_id: str, state: dict[str, Any], action: str) -> None:
        self.collection.update_one(
            {"conversation_id": conversation_id},
            {
                "$set": {
                    "state": state,
                    "updated_at": datetime.now(timezone.utc),
                    "last_action": action,
                }
            },
        )
