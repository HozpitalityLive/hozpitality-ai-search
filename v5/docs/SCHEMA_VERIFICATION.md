# Hozpitality schema verification

The V6 implementation was prepared against the supplied PostgreSQL schema dump.

Verified from the dump:

- PostgreSQL 18.6
- 343 public base tables
- 373 foreign-key constraints
- `master_search_mastersearchindex.embedding` is `vector(384)`
- `master_search_mastersearchindex.search_vector` exists
- an HNSW cosine index already exists on `embedding`
- `pg_trgm`, `unaccent`, and `vector` extensions are present in the dump
- `base_article.category_id -> base_category.id`
- `base_article_location.article_id -> base_article.id`
- `base_article_location.country_id -> countries.id`
- `master_search_mastersearchindex.content_type_id -> django_content_type.id`

The live database must still be verified because a dump and production can differ.

Use:

```bash
python scripts/check_search_stack.py
python scripts/verify_vectors.py
```

For exact production row coverage:

```sql
SELECT
  COUNT(*) AS total,
  COUNT(search_vector_v6) AS fts_ready,
  COUNT(embedding) AS vectors_ready,
  COUNT(*) - COUNT(embedding) AS vectors_missing
FROM public.master_search_mastersearchindex;
```

For extension versions:

```sql
SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('vector','pg_trgm','unaccent')
ORDER BY extname;
```
