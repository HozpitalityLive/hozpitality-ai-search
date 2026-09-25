"""MongoDB-backed conversation persistence for the chat layer.

Collection: ``ai_search_conversations`` (CHAT_COLLECTION)

{
  "conversation_id": "3f2a...",          # unique, client- or server-generated
  "owner_id": "user-42" | null,          # bound when a trusted user id is known
  "version": 7,                          # optimistic concurrency counter
  "state": { ...see state.py... },
  "messages": [                          # bounded to CHAT_MAX_MESSAGES
     {"role": "user", "content": "...", "created_at": ISODate},
     {"role": "assistant", "content": "...", "action": "search",
      "result_refs": ["job:1", ...], "created_at": ISODate}
  ],
  "last_action": "search",
  "created_at": ISODate, "updated_at": ISODate,
  "expires_at": ISODate                  # TTL index; refreshed on every turn
}

Indexes:
  idx_chat_conversation_id  {conversation_id: 1} unique
  idx_chat_updated_at       {updated_at: -1}
  idx_chat_expires_at       {expires_at: 1} expireAfterSeconds=0 (TTL)
  idx_chat_owner_updated    {owner_id: 1, updated_at: -1} sparse
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from .config import settings
from .db import get_collection
from .observability import timed
from .security import valid_conversation_id


class ConversationConflict(Exception):
    """Raised when a concurrent request updated the conversation first."""


class ConversationForbidden(Exception):
    """Raised when a conversation belongs to a different user."""


class ConversationRepository:
    """MongoDB-backed conversation state for the chat layer."""

    def __init__(self, collection: Collection | None = None):
        self.collection = (
            collection
            if collection is not None
            else get_collection(settings.chat_collection)
        )
        self._locks: OrderedDict[str, threading.Lock] = OrderedDict()
        self._locks_guard = threading.Lock()

    def ensure_indexes(self) -> None:
        self.collection.create_index(
            [("conversation_id", ASCENDING)],
            unique=True,
            name="idx_chat_conversation_id",
        )
        self.collection.create_index(
            [("updated_at", DESCENDING)],
            name="idx_chat_updated_at",
        )
        self.collection.create_index(
            [("expires_at", ASCENDING)],
            expireAfterSeconds=0,
            name="idx_chat_expires_at",
        )
        self.collection.create_index(
            [("owner_id", ASCENDING), ("updated_at", DESCENDING)],
            name="idx_chat_owner_updated",
            sparse=True,
        )

    # ------------------------------------------------------------------
    def lock(self, conversation_id: str) -> threading.Lock:
        """Per-conversation lock: serializes turns of one conversation in a worker."""
        with self._locks_guard:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = self._locks[conversation_id] = threading.Lock()
                while len(self._locks) > 10_000:
                    self._locks.popitem(last=False)
            else:
                self._locks.move_to_end(conversation_id)
            return lock

    @staticmethod
    def _expiry(now: datetime) -> datetime:
        return now + timedelta(days=max(1, settings.chat_ttl_days))

    def create(
        self,
        state: dict[str, Any] | None = None,
        *,
        conversation_id: str | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        document = {
            "conversation_id": conversation_id
            if valid_conversation_id(conversation_id)
            else uuid4().hex,
            "owner_id": owner_id,
            "version": 0,
            "state": state or {},
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "expires_at": self._expiry(now),
        }
        with timed("mongo_ms"):
            self.collection.insert_one(dict(document))
        return document

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        if not valid_conversation_id(conversation_id):
            return None
        with timed("mongo_ms"):
            return self.collection.find_one(
                {"conversation_id": conversation_id}, {"_id": 0}
            )

    def ensure(
        self, conversation_id: str | None, owner_id: str | None = None
    ) -> dict[str, Any]:
        """Load a conversation, or create it (keeping a valid client-supplied id)."""
        if conversation_id and valid_conversation_id(conversation_id):
            existing = self.get(conversation_id)
            if existing:
                owner = existing.get("owner_id")
                if owner and owner_id and owner != owner_id:
                    raise ConversationForbidden(conversation_id)
                return existing
            try:
                return self.create(conversation_id=conversation_id, owner_id=owner_id)
            except DuplicateKeyError:
                # Created concurrently by another request.
                existing = self.get(conversation_id)
                if existing:
                    return existing
                raise
        return self.create(owner_id=owner_id)

    def save_turn(
        self,
        conversation_id: str,
        *,
        user_message: str,
        assistant_message: str,
        state: dict[str, Any],
        action: str,
        expected_version: int | None = None,
        result_refs: list[str] | None = None,
    ) -> int:
        """Append a turn and replace state. Returns the new version.

        With ``expected_version`` the write only succeeds if nobody else has
        written since the state was read (optimistic concurrency).
        """
        now = datetime.now(timezone.utc)
        query: dict[str, Any] = {"conversation_id": conversation_id}
        if expected_version is not None:
            query["version"] = (
                expected_version if expected_version else {"$in": [0, None]}
            )
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_message,
            "action": action,
            "created_at": now,
        }
        if result_refs:
            assistant_entry["result_refs"] = result_refs[:10]
        with timed("mongo_ms"):
            result = self.collection.update_one(
                query,
                {
                    "$push": {
                        "messages": {
                            "$each": [
                                {
                                    "role": "user",
                                    "content": user_message,
                                    "created_at": now,
                                },
                                assistant_entry,
                            ],
                            "$slice": -max(2, settings.chat_max_messages),
                        }
                    },
                    "$set": {
                        "state": state,
                        "updated_at": now,
                        "expires_at": self._expiry(now),
                        "last_action": action,
                    },
                    "$inc": {"version": 1},
                },
            )
        if result.matched_count == 0:
            raise ConversationConflict(conversation_id)
        return (expected_version or 0) + 1

    def update_state(
        self, conversation_id: str, state: dict[str, Any], action: str
    ) -> None:
        now = datetime.now(timezone.utc)
        with timed("mongo_ms"):
            self.collection.update_one(
                {"conversation_id": conversation_id},
                {
                    "$set": {
                        "state": state,
                        "updated_at": now,
                        "expires_at": self._expiry(now),
                        "last_action": action,
                    },
                    "$inc": {"version": 1},
                },
            )
