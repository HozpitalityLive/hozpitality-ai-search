from __future__ import annotations

from ai_search.app.db import get_collection
from ai_search.app.repository import SearchDocumentsRepository


if __name__ == "__main__":
    repo = SearchDocumentsRepository(get_collection())
    repo.ensure_indexes()
    names = [index["name"] for index in repo.collection.list_indexes()]
    print("MongoDB Phase 1 search indexes verified:")
    for name in names:
        print(f"- {name}")
