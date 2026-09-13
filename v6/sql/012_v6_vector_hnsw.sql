-- V6 vector verification.
-- The supplied Hozpitality schema already contains embedding vector(384) and
-- an HNSW cosine index named embedding_index. Do not create a second HNSW
-- index unless the verification script says one is missing.
CREATE EXTENSION IF NOT EXISTS vector;

SELECT
    current_setting('server_version') AS postgres_version,
    EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') AS vector_extension,
    EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema='public'
          AND table_name='master_search_mastersearchindex'
          AND column_name='embedding'
          AND udt_name='vector'
    ) AS embedding_column_present,
    EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname='public'
          AND tablename='master_search_mastersearchindex'
          AND indexdef ILIKE '%USING hnsw%embedding%'
    ) AS hnsw_index_present;
