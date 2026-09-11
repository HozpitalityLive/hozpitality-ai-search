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