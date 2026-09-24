from __future__ import annotations

from ai_search.app.db import get_collection
from ai_search.app.repository import SearchDocumentsRepository

if __name__ == "__main__":
    repo = SearchDocumentsRepository(get_collection())
    repo.ensure_indexes()
    print("MongoDB search_documents indexes created.")
