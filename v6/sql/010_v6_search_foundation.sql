-- Hozpitality AI Search V6 search foundation.
-- Run outside a transaction. Safe to re-run.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE public.master_search_mastersearchindex
    ADD COLUMN IF NOT EXISTS search_vector_v6 tsvector;

CREATE OR REPLACE FUNCTION public.hozpitality_v6_search_vector_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.search_vector_v6 :=
          setweight(to_tsvector('simple', unaccent(coalesce(NEW.title, ''))), 'A')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.category_text, ''))), 'A')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.user_name, ''))), 'A')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.location_text, ''))), 'B')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.ai_keywords, ''))), 'B')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.slug, ''))), 'C')
        || setweight(to_tsvector('simple', unaccent(coalesce(NEW.content, ''))), 'C');
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS hozpitality_v6_search_vector_trigger
ON public.master_search_mastersearchindex;

CREATE TRIGGER hozpitality_v6_search_vector_trigger
BEFORE INSERT OR UPDATE OF title, category_text, user_name, location_text,
ai_keywords, slug, content
ON public.master_search_mastersearchindex
FOR EACH ROW
EXECUTE FUNCTION public.hozpitality_v6_search_vector_update();

-- V6 deliberately does not use a trigram index on the large content/keywords
-- fields. Those indexes can consume substantial disk and CPU on 500K+ rows.
-- Keep trigram indexes for short identity/taxonomy fields instead.
