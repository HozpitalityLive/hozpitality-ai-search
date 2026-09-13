# Hozpitality AI V5

AI-powered natural-language search and data assistant for the Hozpitality platform.

Built with **Vanna 2.0**, **FastAPI**, **Ollama**, **PostgreSQL**, and **ChromaDB**.

## Architecture

```text
Hozpitality V5
      │
      ▼
FastAPI
      │
      ▼
Vanna Agent
      │
 ┌────┴─────┐
 ▼          ▼
Ollama   PostgreSQL
Llama      │
           ▼
      Hozpitality Data

## Hozpitality V5 response behavior

- Common greetings/courtesy messages are handled locally by `ChatHandler` and bypass the LLM, database, ChromaDB, and tool loop for fast responses.
- Database questions are expected to execute `run_sql` immediately; the assistant should not ask for permission or expose SQL to the user.
- The Ollama adapter recognizes both native Qwen tool calls and read-only SQL returned as text, converting the latter into an internal `run_sql` call.

