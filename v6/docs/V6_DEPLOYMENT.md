# Hozpitality AI Search V6 — production deployment

## 1. Install

```bash
cd ~/hozpitality-ai-search/v6
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[ollama,postgres,chromadb,fastapi,search]"
```

Keep `.env` outside Git. Use `.env.example` as the template.

## 2. Verify PostgreSQL search capabilities

```bash
python scripts/check_search_stack.py
python scripts/verify_vectors.py
```

The supplied schema contains PostgreSQL 18.6, `vector`, `pg_trgm`, `unaccent`, a `vector(384)` embedding column, and an HNSW cosine index on the existing master search table. Confirm the live database rather than assuming the dump is identical to production.

## 3. Install V6 FTS foundation

Run:

```bash
psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -f sql/010_v6_search_foundation.sql
```

This creates `search_vector_v6` and a trigger. It does not populate all historical rows yet.

## 4. Backfill FTS in throttled batches

Start small:

```bash
python scripts/backfill_search_vector_v6.py --limit 1000 --batch-size 500 --sleep 0.20 --confirm
```

Check database CPU/IO. If healthy, continue:

```bash
python scripts/backfill_search_vector_v6.py --batch-size 1000 --sleep 0.05 --confirm
```

This is keyset-paginated and commits every batch. It does not load 500K rows into memory.

## 5. Create V6 indexes

After the backfill:

```bash
psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -f sql/011_v6_search_indexes.sql
```

The indexes are created concurrently. Run them outside a transaction.

The migration removes the older broad content/keywords trigram and expression indexes created by V5, because V6 replaces those scans with the stored `search_vector_v6` GIN index.

## 6. Verify pgvector

```bash
python scripts/verify_vectors.py
```

If HNSW is missing:

```bash
python scripts/ensure_vector_index.py
```

### Confirm whether vectors exist

Look at:

- `Vectors total/non-null/null`
- `Wrong-dimension vectors`
- `HNSW`

The database column is `vector(384)`. V6's default semantic model is `sentence-transformers/all-MiniLM-L6-v2`, which produces 384 dimensions.

## 7. Generate missing vectors — optional

Do **not** generate 500K embeddings blindly during peak traffic.

Install:

```bash
pip install sentence-transformers
```

Test 1,000 rows:

```bash
python scripts/backfill_embeddings.py --limit 1000 --batch-size 64 --db-batch 256 --sleep 0.20 --confirm
```

Verify:

```bash
python scripts/verify_vectors.py
```

Then continue in controlled batches:

```bash
python scripts/backfill_embeddings.py --batch-size 128 --db-batch 512 --sleep 0.10 --confirm
```

Embedding generation is mostly CPU/GPU work on the worker, while PostgreSQL receives batched updates. It is much safer than embedding rows inside a user search request.

## 8. Important: do not regenerate existing vectors

The script only selects rows where `embedding IS NULL`. Existing vectors are left untouched.

If you change the embedding model, do not mix dimensions/models. Build a new embedding column/index and migrate deliberately.

## 9. Test search

```bash
python scripts/check_search_query.py "who is Raj Bhatt"
python scripts/check_search_query.py "find articles on hotel opening"
python scripts/check_search_query.py "chef jobs in Dubai"
python scripts/check_search_query.py "people who manage hotel operations"
```

## 10. API smoke test

```bash
curl "http://127.0.0.1:8084/api/search?q=Raj%20Bhatt&limit=10"
curl "http://127.0.0.1:8084/api/search/health"
```

## 11. Start service

```bash
sudo systemctl restart hozpitality-ai-v6
sudo systemctl status hozpitality-ai-v6
sudo journalctl -u hozpitality-ai-v6 -f
```

## Recommended production order

1. Foundation SQL
2. 1,000-row FTS backfill test
3. Inspect CPU/IO/query latency
4. Full FTS backfill off-peak
5. Concurrent FTS/trigram indexes
6. Verify pgvector/HNSW
7. Generate embeddings in a separate worker only if needed
8. Start V6
9. Benchmark with `EXPLAIN (ANALYZE, BUFFERS)`

Never run a full embedding backfill and a large index build at the same time on a busy production database.
