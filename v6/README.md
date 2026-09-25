# Hozpitality AI Search V6

Advanced self-hosted AI search and data assistant for Hozpitality.

> **AI Search Chat** (MongoDB `search_documents` search, conversation memory,
> streaming Qwen3 answers, `/search`, `/chat`, `/chat/ws`) lives in
> [`ai_search/`](ai_search/README.md) and is mounted into this app. The chat UI is
> [`frontend/HozpitalityAIChat.tsx`](frontend/HozpitalityAIChat.tsx).

## Stack

- FastAPI
- Vanna 2.0 agent/tooling
- PostgreSQL
- Ollama/Qwen or another tool-capable model
- ChromaDB for agent memory/schema training
- PostgreSQL `pg_trgm`, `unaccent`, full-text search and optional pgvector
- Optional Sentence Transformers for 384-dimensional semantic embeddings

## V6 capabilities

- Global discovery through `master_search_mastersearchindex`
- Dynamic `content_type_id + object_id` resolution
- Dynamic source-table resolution from Django metadata and PostgreSQL schema
- Hybrid lexical search: stored FTS + trigram
- Optional vector/semantic retrieval and cosine HNSW search
- Exact-match and field-weighted relevance ranking
- Typo-tolerant short-field matching
- Dynamic category/country/content-type resolution
- Relationship-aware schema intelligence
- Bounded source enrichment; no `SELECT *` across the database
- Sensitive-column protection
- Bounded in-process search cache
- Deterministic query planner separating search from analytics
- SQL/Vanna fallback for aggregation and reasoning
- Read-only search APIs for testing/frontend integration
- Throttled FTS/vector backfill tools for 500K+ records

## API

```text
GET /health
GET /api/search?q=...
GET /api/search/health
GET /api/schema
POST /api/vanna/v2/chat_sse
```

## PostgreSQL setup

The supplied Hozpitality schema contains 343 public base tables and 373 foreign-key constraints. It also defines `master_search_mastersearchindex.embedding vector(384)` and an existing HNSW cosine index. Always verify the live database with the included scripts before changing production.

Run the V6 foundation first:

```bash
psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -f sql/010_v6_search_foundation.sql
```

Backfill stored FTS vectors in controlled batches:

```bash
python scripts/backfill_search_vector_v6.py --limit 1000 --batch-size 500 --sleep 0.20 --confirm
```

Then, after checking load:

```bash
python scripts/backfill_search_vector_v6.py --batch-size 1000 --sleep 0.05 --confirm
psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -f sql/011_v6_search_indexes.sql
```

Verify:

```bash
python scripts/check_search_stack.py
python scripts/verify_vectors.py
```

## Vector generation

Vectors are optional for the first production rollout. FTS + trigram can run without embeddings.

If semantic search is desired:

```bash
pip install sentence-transformers
python scripts/backfill_embeddings.py --limit 1000 --batch-size 64 --db-batch 256 --sleep 0.20 --confirm
```

After verification, continue in controlled batches. V6 only fills `NULL` embeddings and requires exactly 384 dimensions.

Enable semantic query search in `.env`:

```env
SEARCH_VECTOR_ENABLED=true
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Local development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[ollama,postgres,chromadb,fastapi,search]"
python main.py
```

## Production notes

- Keep `.env` out of Git.
- Use a PostgreSQL read-only AI role with `SELECT` privileges only.
- Apply statement timeouts and connection limits.
- Do FTS/vector backfills off-peak.
- Do not create redundant HNSW indexes; run `scripts/verify_vectors.py` first.
- Do not regenerate vectors on every search request.
- Keep V6 as an independent service behind Nginx rather than mounting it into the legacy V2/V3/V4 process.

See `docs/V6_DEPLOYMENT.md` and `docs/V6_ARCHITECTURE.md`.


## V6 Query Understanding: spaCy EntityRuler + SymSpell

V6 does not use a hardcoded typo-alias table. Query understanding is deterministic and local:

1. **spaCy EntityRuler** extracts Hozpitality entity types (`job`, `professional`,
   `company`, `article`, `event`, `product`, `faq`, `award`) without downloading
   a language model.
2. **SymSpell** corrects misspelled semantic tokens against a vocabulary generated
   from the live Hozpitality master-search corpus.
3. PostgreSQL FTS + pg_trgm performs retrieval; an explicitly supplied location is
   a hard filter when `location_text` is populated.

Build the vocabulary after database migrations/backfills:

```bash
python scripts/build_symspell_dictionary.py
```

Optional environment variables:

```bash
SYMSpell_DICTIONARY=data/symspell_dictionary.txt
SYMSpell_MAX_EDIT_DISTANCE=2
```

If the dictionary is absent, V6 still works; SymSpell simply remains disabled and
PostgreSQL trigram retrieval is used as the fallback.


## V6.5 Canonical AI Search Documents

V6.5 keeps one master-search row per entity and builds a deterministic
canonical search document in `ai_search_text`. The document includes the
entity's identity, taxonomy, location, profile/content fields, and important
database relationships. `metadata` stores the corresponding structured
relationship attributes as JSONB.

For professionals this includes job role, department, job level, education,
current company, skills, languages, industries and experience. Articles include
category and article countries. Jobs include company, country, roles,
departments, levels, industries and job/employment types. Similar relationship
enrichment is provided for companies, events, products, FAQs, awards, posts and
categories.

Run the migration first:

```bash
psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE"   -f sql/013_v6_ai_search_document.sql
```

Then backfill in controlled batches:

```bash
python scripts/backfill_ai_search_text.py --limit 1000 --batch-size 500 --sleep 0.10 --confirm
python scripts/backfill_ai_search_text.py --batch-size 500 --sleep 0.05 --confirm
```

After the backfill, rebuild the V6 FTS vector once so existing rows include
`ai_search_text`:

```bash
python scripts/backfill_ai_search_text.py --limit 1 --rebuild-fts --confirm
```

For semantic search, embeddings now prefer `ai_search_text` automatically:

```bash
python scripts/backfill_embeddings.py --limit 1000 --batch-size 64 --db-batch 256 --sleep 0.20 --confirm
```

The V6 lexical index intentionally uses the existing GIN index on
`search_vector_v6`; a large trigram index on `ai_search_text` is not created.
This keeps disk/CPU costs bounded while FTS searches the complete document.

## AI Search — MongoDB Phase 1

See `ai_search/README.md` for the MongoDB-first Phase 1 search service.
