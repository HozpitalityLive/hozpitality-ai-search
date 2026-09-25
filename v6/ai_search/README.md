# Hozpitality AI Search Chat

Conversational search over Hozpitality **jobs, professionals, companies, products,
articles, events, awards and FAQs**. MongoDB `search_documents` is the only source
of truth; Qwen3 8B (Ollama) only phrases answers about records that were actually
retrieved.

```text
Find chef jobs in Dubai          -> 5 real job results
Only management positions        -> same search, level = manager (jobs, still Dubai)
With accommodation               -> + accommodation = true (evidence-checked)
Show me more                     -> next jobs, never repeating what was shown (#6, #7…)
Compare the first three          -> table of the real #1-#3 records, fields from MongoDB
Actually show professionals instead -> entity = professional, keeps "chef" + Dubai
Show me chefs in Abu Dhabi       -> professionals, location changed
Start over                       -> clears the search state
```

---

## Architecture

```text
 Browser (Next.js + MUI)                          frontend/HozpitalityAIChat.tsx
   │  WebSocket /chat/ws  (fallback: POST /chat)
   ▼
 FastAPI  ai_search/app/router.py ── auth · rate limit · request id · validation
   │
   ▼
 ChatService.prepare()                           chat_service.py
   ├─ ConversationRepository ◄──► MongoDB ai_search_conversations   (state, history)
   ├─ dialogue.interpret(message, state)          deterministic: action + explicit changes
   ├─ state.apply_intent(...)                     only fields the user changed
   ├─ SearchService.execute_plan(state plan) ──► MongoDB search_documents   (top 5)
   │     lexical/exact/phrase/alias · typo · filters · (FAISS semantic) · rerank
   └─ answers.*                                   deterministic answer + grounded prompt
   │
   ▼  (0 or 1 LLM call)
 Ollama qwen3:8b   think=false · temperature 0.1 · num_ctx 4096 · streamed
   │
   ▼
 ChatService.finalize()   validate output (no unknown URLs/counts) → persist turn (versioned)
   │
   ▼
 events: start → results → delta* → final → completion

 Background (never on the request path):
 search_documents.ai_search_text → embedding_worker (batched, incremental, 384-d)
      → ai_search_embeddings → FAISS index file → hot-reloaded by API workers
```

**The LLM never decides** entity, location, keywords, filters, which records exist,
IDs, URLs or which result "the second one" is. Those are deterministic.

### Modules (`ai_search/app/`)

| Phase | Module | Responsibility |
|---|---|---|
| 1 | `repository.py` | MongoDB retrieval: exact title/alias/keyword, `$text` (`idx_ai_search_text`), token fallback, entity/city/country/status/live filters, browse, lookups |
| 1 | `ranking.py`, `typo.py`, `normalization.py` | deterministic ranking, conservative typo correction |
| 2 | `query_understanding.py` | intent/entity/keywords/location/experience/level/department/industry/category/date/filters; clarification rules |
| 2 | `llm.py` | optional Ollama fallback for ambiguous one-shot `/search` queries (short timeout) |
| 2 | `vector.py`, `embeddings.py` | optional FAISS semantic retrieval; background embedding pipeline |
| 2 | `service.py` | `plan_query()` (understanding) + `execute_plan()` (retrieval, hybrid rerank, strict-filter evidence, related results) |
| 3 | `dialogue.py` | deterministic turn interpretation (refine / remove / change / more / compare / reference / reset) |
| 3 | `state.py` | conversation state shape, merge rules, numbered result history, reference resolution |
| 3 | `conversation.py` | MongoDB persistence, TTL, optimistic concurrency, owner binding |
| 3 | `chat_service.py` | orchestration (`prepare` → LLM → `finalize`) |
| 3 | `answers.py`, `evidence.py` | deterministic answers, grounded prompts, comparison/detail views from real fields, output validation |
| 3 | `ollama_client.py` | pooled/streaming Ollama client, circuit breaker, `<think>` filtering |
| 4 | `router.py`, `schemas.py`, `main.py` | HTTP, WebSocket and SSE API |
| 5 | `security.py`, `observability.py`, `container.py` | auth, rate limit, URL/prompt hygiene, JSON logs, metrics, wiring |

---

## Conversation behaviour

### State (MongoDB `ai_search_conversations.state`)

```json
{
  "entity": "job",
  "entity_source": "text",
  "location": {"city": "Dubai", "country": "United Arab Emirates", "raw": null},
  "location_any": false,
  "keywords": ["chef"],
  "filters": {"level": "manager", "accommodation": true},
  "strict_filters": ["level", "accommodation"],
  "pending": null,
  "topic": "9d0c…",
  "shown": ["job:103", "job:106", "…"],
  "current_list": ["job:103", "job:106", "job:111", "job:112", "job:101", "job:108"],
  "last_results": ["job:108"],
  "result_history": [{"key": "job:103", "entity_type": "job", "entity_id": "103",
                      "doc_id": "job:103", "title": "Head Chef", "url": "https://…",
                      "company": "Marina Resorts", "city": "Dubai", "position": 1, "turn": 3}],
  "focus": "job:103",
  "previous_focus": null,
  "turn": 5
}
```

Bounds: `messages` ≤ `CHAT_MAX_MESSAGES` (20), `result_history`/`current_list` ≤ 50,
`shown` ≤ 100. Documents expire after `CHAT_TTL_DAYS` of inactivity (TTL index).

### Merge rules

* Only fields explicitly changed by the current message change. Refinements add
  (“only management positions”, “with accommodation”); nothing else resets.
* **The entity never changes implicitly.** A role word (“chef”) cannot turn a job
  search into a professional search. Strong module nouns (“professionals”,
  “companies”, “jobs”) change it; weak ones (“positions”, “roles”) never switch an
  existing entity. On a switch, filters that don't apply to the new entity are
  dropped (e.g. accommodation) and restrictions become preferences.
* Removals are detected first and consumed: “remove the accommodation
  requirement”, “don't restrict it to management”, “any level”, “anywhere, not
  just Dubai”, “remove the location”, “remove all filters”.
* Location: “actually Abu Dhabi”, “change that to Mumbai”, “anywhere in UAE”
  (country, no city). Unknown places (“Antarctica”) are explicit locations: no
  clarification, no exact results, related results only.
* A new complete request (“find sous chef jobs in Mumbai”) resets filters.

### Hard constraints vs ranking signals

| Hard (exact results must satisfy) | Ranking signals |
|---|---|
| entity, city/country, status/live, salary/employment type when requested | keyword relevance, semantic similarity |
| accommodation = true (**positive evidence** on the record) | level (preference) |
| level **when explicitly restricted** (“only management…”) — evidence in title or level field | experience, department, industry |

Candidates that fail a strict constraint are never shown as exact matches; when
no exact match exists they appear as clearly-labelled `related_results`.

### References, show more, compare

* Results are numbered per search; “show me more” continues the numbering
  (#6, #7…) and excludes everything already shown for the topic.
* “the second one”, “the first three”, “#4”, “the last one” resolve against that
  numbering; “these” = the page shown last; “that one/that company” = the result
  discussed last; “the previous job” = the one before it.
* Compare/detail re-fetch the referenced records from MongoDB by `_id`. The
  comparison table contains only real field values (“Not specified” otherwise).
* Asking for records that don't exist gets a clear answer
  (“I don't have three recent results to compare yet. Run a search first.”).

---

## API

All endpoints accept `X-Request-ID` (echoed back) and, when `AI_SEARCH_API_KEYS`
is set, `X-API-Key` (WebSocket: `?api_key=`).

| Method | Path | Purpose |
|---|---|---|
| GET/POST | `/search` | Phase 1/2 search (unchanged contract, max 5 results) |
| GET | `/search/understand?q=` | query understanding only |
| POST | `/chat` | one chat turn (non-streaming) |
| WS | `/chat/ws` | streaming chat turns |
| POST | `/chat/stream` | streaming chat turn over SSE |
| GET | `/chat/{conversation_id}` | state + recent messages |
| POST | `/chat/{conversation_id}/reset` | clear the search state |
| GET | `/health`, `/search/health` | liveness + MongoDB/LLM status |
| GET | `/metrics` | counters and latency aggregates (internal) |

### `POST /chat`

```json
{"message": "Find chef jobs in Dubai", "conversation_id": "optional-8-128-chars", "limit": 5}
```

Response (predictable fields; `comparison`/`detail` only when relevant):

```json
{
  "conversation_id": "3f2a…",
  "action": "search | more | compare | detail | related_entity | clarify | reset | smalltalk",
  "answer": "I found 5 chef jobs in Dubai.",
  "results": [{"number": 1, "entity_type": "job", "entity_id": "101", "doc_id": "job:101",
               "title": "Executive Chef", "company": "ABC Hospitality",
               "location": {"city": "Dubai", "country": "United Arab Emirates"},
               "snippet": "…", "url": "https://www.hozpitality.com/jobs/101", "image": null,
               "score": 377.0, "matched_by": ["exact_keyword", "location_filter"]}],
  "related_results": [],
  "message": null,
  "understanding": {"entity": "job", "keywords": ["chef"], "city": "Dubai", "…": "…", "turn": {"action": "search"}},
  "state": {"entity": "job", "location": {"city": "Dubai", "country": "United Arab Emirates"},
            "keywords": ["chef"], "filters": {}, "strict_filters": [], "last_results": ["job:101", "…"]},
  "references": [{"number": 1, "key": "job:101", "title": "Executive Chef", "url": "https://…"}],
  "comparison": {"columns": ["Field", "#1 Head Chef", "…"], "rows": [["Company", "Marina Resorts", "…"]],
                 "fields": [{"key": "company", "label": "Company"}], "items": ["…"]},
  "suggestions": ["Show me more", "Compare the first 3", "Tell me more about the first one"],
  "llm": {"used": true},
  "request_id": "…"
}
```

Errors: `422` validation, `401` API key, `404` unknown conversation, `409`
concurrent update of the same conversation (retry), `429` rate limit
(`Retry-After`), `503` MongoDB unavailable. Error bodies never include
credentials, connection strings or the submitted message.

### WebSocket `/chat/ws`

Client → server:

```json
{"type": "chat", "message": "Only management positions", "conversation_id": "…", "request_id": "…"}
{"type": "stop"}
{"type": "ping"}
```

Server → client, per turn:

```text
{"type":"start","conversation_id","request_id"}
{"type":"results","data":{results, related_results, comparison, state, suggestions, …}}   // before any LLM text
{"type":"delta","data":{"text":"I found "}}                                            // 0..n
{"type":"final","data":<ChatResponse>}                                                 // validated answer: authoritative
{"type":"completion","data":{"status":"done"|"stopped"}}
{"type":"error","data":{"message","code"}}
```

`stop` ends generation; the turn is still saved with a complete deterministic
answer. One socket can carry many turns; a request is complete only on
`completion` or `error`. The legacy client payload (`message`,
`conversation_id`, `request_id`, `metadata`) is accepted.

---

## LLM (Qwen3 8B on a Tesla T4)

* One call at most per turn; none for reset, clarification, “open the second
  one”, field questions (“what company…”), or when disabled.
* `think: false`, `temperature 0.1`, `num_ctx 4096`, `num_predict 320` (480 for
  comparisons), `keep_alive 30m`, pooled connections.
* `OLLAMA_TIMEOUT_SECONDS` (default 60) bounds the whole answer, including
  streaming. Every timeout/failure is logged (`llm_failure`, reason, duration)
  and counted; after `OLLAMA_FAILURE_THRESHOLD` consecutive failures a circuit
  breaker answers deterministically for `OLLAMA_CIRCUIT_RESET_SECONDS`.
* Prompt security: retrieved records are sanitized (markup, role tokens and
  injection phrasing removed, truncated) and sent as JSON inside
  `<search_data>`, declared as untrusted data. The model output is validated:
  URLs not present in the results are stripped, raw HTML removed, answers that
  echo injection phrasing or claim result counts we didn't return are replaced
  by the deterministic answer.

---

## Environment

See [`.env.example`](.env.example) for every variable. Key ones:

| Variable | Default | Notes |
|---|---|---|
| `MONGODB_URI` | — | credentials only here, never in code/logs |
| `CHAT_COLLECTION` | `ai_search_conversations` | conversation memory |
| `CHAT_TTL_DAYS` | 30 | inactive conversations expire |
| `OLLAMA_BASE_URL` / `OLLAMA_CHAT_MODEL` | `http://127.0.0.1:11434` / `qwen3:8b` | |
| `OLLAMA_TIMEOUT_SECONDS` | 60 | chat answer budget |
| `OLLAMA_QUERY_TIMEOUT_SECONDS` | 8 | `/search` ambiguity fallback budget |
| `CHAT_LLM_ENABLED` | true | false = deterministic answers only |
| `SEMANTIC_SEARCH_ENABLED` | false | needs the FAISS index from the embedding worker |
| `AI_SEARCH_API_KEYS` | empty | server-to-server keys; empty = open |
| `TRUST_USER_HEADER` | false | bind conversations to `X-User-Id` from a trusted proxy |
| `RATE_LIMIT_CHAT_PER_MINUTE` / `RATE_LIMIT_SEARCH_PER_MINUTE` | 30 / 120 | per client, per process (also use nginx `limit_req`) |
| `CORS_ALLOW_ORIGINS` | localhost:3000 | also read by the V6 `main.py` |
| `PUBLIC_SITE_BASE_URL` | empty | absolutizes site-relative links |

Frontend (`frontend/HozpitalityAIChat.tsx`):

| Variable | Default |
|---|---|
| `NEXT_PUBLIC_AI_SEARCH_URL` | falls back to `NEXT_PUBLIC_AI_V6_URL`, then `https://llm.hozpitality.com` |
| `NEXT_PUBLIC_AI_SEARCH_WS_URL` | `<URL>/chat/ws` with `ws(s)://` |

---

## Deployment

```bash
cd /home/dev/hozpitality-ai-search/v6
source .venv/bin/activate
pip install -r ai_search/requirements.txt          # adds httpx
cp ai_search/.env.example ai_search/.env            # fill in; keep out of Git

# Ollama on the T4
ollama pull qwen3:8b
sudo systemctl edit ollama   # Environment=OLLAMA_KEEP_ALIVE=30m  OLLAMA_NUM_PARALLEL=2

# Indexes (verifies idx_ai_search_text; creates conversation indexes incl. TTL)
python -m ai_search.scripts.init_indexes
python -m ai_search.scripts.init_indexes --browse   # optional entity/recency index

# Services
sudo cp deployment/systemd/hozpitality-ai-v6.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl restart hozpitality-ai-v6

# nginx: add the locations from deployment/nginx-websocket.conf (+ limit_req zones)
sudo nginx -t && sudo systemctl reload nginx

# Smoke test
curl -s localhost:8085/search/health | python3 -m json.tool
curl -s -X POST localhost:8085/chat -H 'Content-Type: application/json' \
  -d '{"message":"Find chef jobs in Dubai"}' | python3 -m json.tool
```

The chat routes are mounted into the V6 app (`main.py`, port 8085) by
`register_mongo_search_routes`. `deployment/systemd/hozpitality-ai-search.service`
runs the AI search API standalone on 8086 instead, if preferred.

### Semantic search (optional)

```bash
pip install -r ai_search/requirements-phase2-vector.txt
python -m ai_search.scripts.embedding_worker --once          # first full pass
sudo cp deployment/systemd/hozpitality-embeddings.{service,timer} /etc/systemd/system/
sudo systemctl enable --now hozpitality-embeddings.timer     # incremental every 30 min
# then SEMANTIC_SEARCH_ENABLED=true and restart the API
```

The worker only reads `search_documents`, embeds new/changed `ai_search_text`
in batches, stores vectors in `ai_search_embeddings`, and atomically publishes
the FAISS index; API workers hot-reload it.

### Migration from the Phase 3 prototype

No data migration is required.

* Existing `ai_search_conversations` documents load as-is: the prototype's
  `last_result_items` is converted into `result_history` on read, and missing
  `version` fields are treated as 0.
* Run `init_indexes` once to add the TTL (`idx_chat_expires_at`) and owner indexes.
  Old documents without `expires_at` are not expired until their next turn.
* `search_documents` is not modified.

### Frontend

`HozpitalityAIChat.tsx` (+ `chatProtocol.ts`, `safeHtml.ts`, `ShapeGrid.tsx`) is a
client component for an existing Next.js/MUI app: copy the four files next to each
other and render `<HozpitalityAIChat />`. Set `NEXT_PUBLIC_AI_SEARCH_URL`.

---

## Local development and tests

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; source .venv/bin/activate on Linux
pip install -r ai_search/requirements-dev.txt

python -m pytest ai_search/tests -q                  # 210 tests, no external services
ruff check ai_search && mypy ai_search/app --ignore-missing-imports

# Run the real API against an in-memory seeded database (no production data):
python -m ai_search.scripts.dev_server --port 8086
```

Test suites: `test_phase1_search_foundation.py`, `test_phase2_nlu.py`,
`test_phase3_dialogue.py`, `test_phase3_chat.py` (includes the critical regression
conversation), `test_phase3_llm.py`, `test_phase4_api.py` (HTTP, WebSocket, SSE),
`test_phase5_production.py`, plus the original Phase 1/2 tests.
Frontend protocol tests: `frontend/chatProtocol.test.ts` (Node test runner).

---

## Observability

JSON logs (one line per event) with `request_id` and `conversation_id`:
`http_request`, `chat_turn` (action, result counts, `search_ms`, `mongo_ms`,
`llm_ms`, `total_ms`, `llm_used`, `fallback_reason`), `llm_complete`
(`ollama_ms`, tokens), `llm_failure`, `llm_fallback`, `llm_circuit_open`,
`embedding_sync`. Message text, credentials and connection strings are never
logged (connection strings/keys are also redacted defensively). `/metrics`
exposes counters (`chat_turns`, `llm_timeout`, `llm_fallback_*`,
`rate_limited`, …) and latency aggregates.

## Known limitations

* Rate limiting and per-conversation locks are per process; use nginx
  `limit_req` for a global limit. Cross-worker writes are protected by versioned
  updates (HTTP 409 on a true race).
* Level/accommodation evidence depends on what the records contain (title,
  level fields, benefit fields, description text). Records without evidence are
  offered as related results, never as exact matches.
* Relative links are only absolutized when `PUBLIC_SITE_BASE_URL` is set; records
  without `url`/`links.detail` show no link (slugs are never guessed).
