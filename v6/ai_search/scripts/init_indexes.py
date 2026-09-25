"""Verify search indexes and create the conversation-memory indexes.

    python -m ai_search.scripts.init_indexes            # verify + chat indexes
    python -m ai_search.scripts.init_indexes --browse   # also add the optional
                                                        # entity/recency index

search_documents is never modified except for the optional, non-text
{entity_type, created_at} index used by filter-only "browse" queries
("events in Dubai"). No additional text index is ever created.
"""

from __future__ import annotations

import argparse

from ai_search.app.config import settings
from ai_search.app.conversation import ConversationRepository
from ai_search.app.db import get_collection
from ai_search.app.repository import SearchDocumentsRepository

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--browse",
        action="store_true",
        help="create idx_entity_created_at on search_documents",
    )
    args = parser.parse_args()

    repo = SearchDocumentsRepository(get_collection())
    repo.ensure_indexes()
    if args.browse:
        repo.collection.create_index(
            [("entity_type", 1), ("created_at", -1)],
            name="idx_entity_created_at",
            background=True,
        )
    print("search_documents indexes:")
    for index in repo.collection.list_indexes():
        print(f"- {index['name']}")

    conversations = ConversationRepository()
    conversations.ensure_indexes()
    print(f"{settings.chat_collection} indexes:")
    for index in conversations.collection.list_indexes():
        print(f"- {index['name']}")
