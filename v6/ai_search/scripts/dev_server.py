#!/usr/bin/env python3
"""Local development server backed by an in-memory, seeded MongoDB.

Runs the real API (search, chat, WebSocket streaming) without touching the
production database - useful for frontend work and demos.

    pip install mongomock uvicorn
    python -m ai_search.scripts.dev_server --port 8086

Point OLLAMA_BASE_URL at a local Ollama to get LLM answers; without it the
chat answers deterministically.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8086)
    args = parser.parse_args()

    import mongomock
    import uvicorn

    from ai_search.app.container import build_container, set_container
    from ai_search.app.main import app
    from ai_search.tests.fixtures_data import seed_documents

    db = mongomock.MongoClient()["mongoAdmin"]
    db["search_documents"].insert_many(seed_documents())
    db["search_documents"].create_index(
        [("ai_search_text", "text")], name="idx_ai_search_text"
    )
    set_container(
        build_container(
            search_collection=db["search_documents"],
            chat_collection=db["ai_search_conversations"],
        )
    )
    # The in-memory database has no ping(); report healthy.
    import ai_search.app.main as main_module
    import ai_search.app.router as router_module

    main_module.ping = router_module.ping = lambda: True
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
