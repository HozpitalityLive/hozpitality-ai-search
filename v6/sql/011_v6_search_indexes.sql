-- Hozpitality AI Search V6 indexes.
-- IMPORTANT: CREATE/DROP INDEX CONCURRENTLY must be run outside a transaction.
CREATE INDEX CONCURRENTLY IF NOT EXISTS master_search_msi_v6_fts_idx
ON public.master_search_mastersearchindex USING gin (search_vector_v6);

CREATE INDEX CONCURRENTLY IF NOT EXISTS master_search_msi_v6_title_trgm_idx
ON public.master_search_mastersearchindex USING gin (title gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS master_search_msi_v6_user_trgm_idx
ON public.master_search_mastersearchindex USING gin (user_name gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS master_search_msi_v6_category_trgm_idx
ON public.master_search_mastersearchindex USING gin (category_text gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS master_search_msi_v6_location_trgm_idx
ON public.master_search_mastersearchindex USING gin (location_text gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS master_search_msi_v6_slug_trgm_idx
ON public.master_search_mastersearchindex USING gin (slug gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS master_search_msi_v6_type_live_created_idx
ON public.master_search_mastersearchindex (content_type_id, is_live, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS master_search_msi_v6_type_object_idx
ON public.master_search_mastersearchindex (content_type_id, object_id);

-- Remove the broad V5 content-trigram/expression indexes if they were created
-- by the previous V5 migration. V6 replaces them with one stored FTS vector.
DROP INDEX CONCURRENTLY IF EXISTS master_search_msi_content_trgm_idx;
DROP INDEX CONCURRENTLY IF EXISTS master_search_msi_keywords_trgm_idx;
DROP INDEX CONCURRENTLY IF EXISTS master_search_msi_global_fts_idx;
