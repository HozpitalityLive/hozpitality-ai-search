# Hozpitality AI Search V6

Advanced self-hosted AI search and data assistant for Hozpitality.

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
