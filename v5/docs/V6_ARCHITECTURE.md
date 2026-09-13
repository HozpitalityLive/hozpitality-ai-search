# Hozpitality AI Search V6

V6 is a database-first, hybrid global retrieval system. The LLM is not the database search engine.

## Principles

- `master_search_mastersearchindex` answers **where the relevant object is**.
- `content_type_id + object_id` identifies the object.
- Django content types and live schema metadata resolve the source model/table.
- Source tables answer **what the authoritative data is**.
- PostgreSQL FTS handles fast lexical retrieval.
- `pg_trgm` handles names, titles, categories, locations and spelling variations.
- pgvector handles semantic retrieval when vectors are available.
- A deterministic ranker combines exact, lexical, fuzzy and semantic signals.
- Vanna/LLM is reserved for analytics, reasoning and questions that require SQL.
- Search results are bounded; the application never fetches every row from every table.
- Sensitive source columns are filtered from generic enrichment.

## Query examples

`Raj Bhatt` -> global entity discovery.

`find articles on hotel opening` -> article content-type filter + hybrid search.

`hotel opening` -> global search without incorrectly forcing the query into jobs.

`chef jobs in Dubai` -> job filter + global retrieval.

`how many chef jobs are in Dubai?` -> SQL analytics path.

`people who manage hotel operations` -> semantic retrieval when vectors are enabled.

## Scaling model

For 500K+ master-index rows, the normal request path is:

1. normalize query
2. determine cheap search plan
3. retrieve up to ~100 lexical/fuzzy/vector candidates
4. combine and rank
5. return 10–20
6. fetch only those source objects
7. format the answer

The FTS vector is stored and indexed; it is not recomputed for every search.

## Index maintenance

New/updated master-index records are maintained by the V6 trigger for `search_vector_v6`. Embeddings should be generated asynchronously for new/changed records rather than during the user request.

## Vector strategy

The supplied database already has `embedding vector(384)` and an HNSW cosine index. V6 uses `sentence-transformers/all-MiniLM-L6-v2` by default because its output dimension is 384. If the production embedding model changes, use a new vector column/index rather than mixing models.
