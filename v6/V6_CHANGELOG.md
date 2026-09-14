# V6 changelog

## Search engine
- Replaced V5's broad multi-`ILIKE` global search with candidate retrieval using stored FTS + short-field trigram search.
- Added optional pgvector semantic candidate retrieval.
- Added deterministic field-weighted relevance scoring.
- Added exact title/name/category matches and typo tolerance.
- Added dynamic content-type filtering only when an actual Django content type can be resolved.
- Removed the incorrect behavior where generic words such as `opening` forced job searches.

## Database performance
- Added `search_vector_v6` stored by trigger.
- Added a single GIN FTS index for the V6 search vector.
- Added trigram indexes only to short searchable identity/taxonomy fields.
- Removed the old broad content/keywords trigram indexes from the V5 migration path.
- Added composite type/live/object indexes.
- Added throttled keyset-paginated backfill tools.

## Semantic search
- Added optional `sentence-transformers/all-MiniLM-L6-v2` query embedding.
- Added 384-dimension validation.
- Added an embedding backfill worker script that only fills NULL vectors.
- Reuses the existing HNSW index when present and provides a safe checker/creator.

## Reliability
- Added bounded per-process cache.
- Added read-only `/api/search` and `/api/search/health` endpoints.
- Kept source enrichment bounded to matched object IDs.
- Kept sensitive columns out of generic source enrichment.
- Preserved Vanna SQL path for analytics and complex database reasoning.


## V6.1 Final — Entity + Location Retrieval

- Entity words are removed from lexical keyword retrieval after routing.
- Locations are dynamically classified from indexed location data.
- Keyword and location candidate pools are ranked independently and merged.
- Trigram retrieval is token-aware for typo-tolerant search.
- Default V6 service port is 8085 so V5 can remain on 8084.
