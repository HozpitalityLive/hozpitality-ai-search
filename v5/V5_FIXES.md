# Hozpitality AI V5 fixes

## Included fixes

- Tool-call JSON / SQL is never rendered as a user-facing assistant message.
- Qwen-generated SQL is executed through `run_sql` and the user receives the query results.
- SQL permission prompts are explicitly prohibited.
- SQL results include a compact Markdown table fallback showing up to 10 rows.
- Common greetings use a local fast path and do not invoke Ollama, ChromaDB, or PostgreSQL.
- Fixed the greeting regex so punctuation does not break fast-path matching.
- Corrected job status mapping to `job_status`; start date remains `job_start_date`.

## Deployment

Keep your existing `.env`.

After replacing the V5 source:

```bash
sudo systemctl restart hozpitality-ai-v5
sudo systemctl status hozpitality-ai-v5
sudo journalctl -u hozpitality-ai-v5 -f
```
