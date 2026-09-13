"""Read-only verification for pgvector readiness and vector coverage."""
from __future__ import annotations
import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()

def main():
    conn=psycopg2.connect(host=os.getenv('POSTGRES_HOST'),port=os.getenv('POSTGRES_PORT','5432'),dbname=os.getenv('POSTGRES_DATABASE'),user=os.getenv('POSTGRES_USER'),password=os.getenv('POSTGRES_PASSWORD'))
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','pg_trgm','unaccent') ORDER BY extname")
            print('Extensions:', cur.fetchall())
            cur.execute("SELECT udt_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema='public' AND table_name='master_search_mastersearchindex' AND column_name='embedding'")
            print('Embedding column:', cur.fetchone())
            cur.execute("SELECT COUNT(*), COUNT(embedding), COUNT(*)-COUNT(embedding) FROM public.master_search_mastersearchindex")
            print('Vectors total/non-null/null:', cur.fetchone())
            cur.execute("SELECT COUNT(*) FROM public.master_search_mastersearchindex WHERE embedding IS NOT NULL AND vector_dims(embedding) <> 384")
            print('Wrong-dimension vectors:', cur.fetchone()[0])
            cur.execute("""SELECT indexname,indexdef FROM pg_indexes WHERE schemaname='public' AND tablename='master_search_mastersearchindex' AND indexdef ILIKE '%hnsw%'""")
            print('HNSW:', cur.fetchall())
    finally: conn.close()
if __name__=='__main__': main()
