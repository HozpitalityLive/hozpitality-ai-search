-- Hozpitality AI Search V6.5
-- Canonical search document + structured metadata.
--
-- Safe to run repeatedly. This migration intentionally does NOT add a
-- trigram index to ai_search_text because it can become very large.
-- ai_search_text is indexed through search_vector_v6 (GIN).
--
-- metadata is structured JSONB for exact/future faceted filtering.

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

ALTER TABLE public.master_search_mastersearchindex
    ADD COLUMN IF NOT EXISTS ai_search_text text;

ALTER TABLE public.master_search_mastersearchindex
    ADD COLUMN IF NOT EXISTS metadata jsonb;

UPDATE public.master_search_mastersearchindex
SET metadata = '{}'::jsonb
WHERE metadata IS NULL;

ALTER TABLE public.master_search_mastersearchindex
    ALTER COLUMN metadata SET DEFAULT '{}'::jsonb;

-- Keep V6 FTS authoritative and include the complete canonical search
-- document. The old columns remain useful as separate structured signals.
CREATE OR REPLACE FUNCTION public.hozpitality_v6_search_vector_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.search_vector_v6 :=
          setweight(to_tsvector('simple', unaccent(coalesce(NEW.title, ''))), 'A')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.entity_name, ''))), 'A')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.category_text, ''))), 'A')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.company_name, ''))), 'A')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.user_name, ''))), 'A')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.subcategory_text, ''))), 'B')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.country_text, ''))), 'B')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.city_text, ''))), 'B')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.location_text, ''))), 'B')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.ai_keywords, ''))), 'B')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.slug, ''))), 'C')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.ai_summary, ''))), 'C')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.content, ''))), 'C')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.ai_search_text, ''))), 'B');
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS hozpitality_v6_search_vector_trigger
ON public.master_search_mastersearchindex;

CREATE TRIGGER hozpitality_v6_search_vector_trigger
BEFORE INSERT OR UPDATE OF
    title, entity_name, category_text, company_name, user_name,
    subcategory_text, country_text, city_text, location_text,
    ai_keywords, slug, ai_summary, content, ai_search_text
ON public.master_search_mastersearchindex
FOR EACH ROW
EXECUTE FUNCTION public.hozpitality_v6_search_vector_update();

-- FTS index. Re-use the existing index name if the V6 foundation already
-- created it; CREATE INDEX IF NOT EXISTS is safe for fresh installations.
CREATE INDEX IF NOT EXISTS msi_search_v6_gin
ON public.master_search_mastersearchindex
USING gin (search_vector_v6);

-- Structured metadata index. Useful for future exact filters without
-- bloating the lexical search index.
CREATE INDEX IF NOT EXISTS msi_metadata_gin
ON public.master_search_mastersearchindex
USING gin (metadata jsonb_path_ops);
