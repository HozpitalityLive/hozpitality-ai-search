# V6 rollout checklist

## Before production

- [ ] `.env` is not committed.
- [ ] PostgreSQL AI account is read-only.
- [ ] `vector`, `pg_trgm`, `unaccent` verified.
- [ ] `master_search_mastersearchindex.embedding` verified as `vector(384)`.
- [ ] Existing HNSW index verified; do not create duplicates.
- [ ] `search_vector_v6` foundation applied.
- [ ] FTS backfill completed in controlled batches.
- [ ] V6 FTS indexes created concurrently.
- [ ] Search smoke tests pass.
- [ ] Analytics queries still route to Vanna SQL.
- [ ] Nginx points the V6 hostname/path to port 8084.
- [ ] CORS contains the actual frontend origin.

## Performance acceptance

Run:

```bash
python scripts/benchmark_search.py "chef jobs Dubai"
```

Inspect:

- `Execution Time`
- index scan vs sequential scan
- shared hit/read buffers
- rows removed by filter

Do not enable semantic vector search for all traffic until lexical search is healthy.

## Vector acceptance

```bash
python scripts/verify_vectors.py
```

A partially populated vector column is acceptable. V6 automatically falls back to lexical search when query embeddings are unavailable.
