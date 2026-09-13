# V6 database migrations

Run in this order against the Hozpitality database:

1. `010_v6_search_foundation.sql`
2. Backfill `search_vector_v6` with `scripts/backfill_search_vector_v6.py`
3. `011_v6_search_indexes.sql`
4. `012_v6_vector_hnsw.sql` for read-only vector verification
5. `scripts/ensure_vector_index.py` only if the verification says HNSW is missing

Do not run migrations in a transaction. The index migration uses `CONCURRENTLY`.
