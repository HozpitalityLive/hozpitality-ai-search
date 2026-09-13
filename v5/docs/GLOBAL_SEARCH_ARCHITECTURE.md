# Hozpitality AI V6 — Global Search Architecture

## Goal

Natural-language discovery across Hozpitality without adding a new hard-coded handler every time a module, category, or content type changes.

## Retrieval pipeline

1. Normalize the user request.
2. Detect an explicit content type only when the wording is clear.
3. Search `master_search_mastersearchindex` first.
4. Rank title, category, location, person/owner, keywords, content and full-text matches.
5. Resolve `content_type_id` through `django_content_type`.
6. Resolve `object_id` against the actual source model/table using live database metadata.
7. Fetch safe authoritative source fields from that table.
8. Return a compact result set (default 10).
9. Fall back to Vanna SQL/LLM only for analytics, unsupported retrieval, or when the master index has no match.

## Why this is faster

The LLM is no longer required to decide which table to search for ordinary entity/document discovery. PostgreSQL performs the first-pass retrieval, and only the matching objects are enriched.

## Why this scales

The resolver uses `django_content_type`, `information_schema`, primary-key metadata and the master index's `content_type_id/object_id`. It does not require an individual Python branch for every article category, professional name, company, event type, or product category.

## Security

Source enrichment removes credential/session/secret fields. The database account used by V6 should still be read-only and restricted to the intended schemas.

## Recommended indexes

Run `sql/001_global_search_indexes.sql` once on the production PostgreSQL database. The supplied schema already contains `pg_trgm`, `unaccent`, `vector`, and a GIN index for the existing `search_vector`.
