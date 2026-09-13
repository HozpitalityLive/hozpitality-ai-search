"""Create the pgvector HNSW index only when one is actually missing."""
from __future__ import annotations
import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()

def main():
    conn=psycopg2.connect(host=os.getenv('POSTGRES_HOST'),port=os.getenv('POSTGRES_PORT','5432'),dbname=os.getenv('POSTGRES_DATABASE'),user=os.getenv('POSTGRES_USER'),password=os.getenv('POSTGRES_PASSWORD'))
    conn.autocommit=True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector')")
            if not cur.fetchone()[0]:
                raise SystemExit("pgvector extension is not installed")
            cur.execute("""SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='master_search_mastersearchindex' AND column_name='embedding' AND udt_name='vector')""")
            if not cur.fetchone()[0]:
                raise SystemExit("master_search_mastersearchindex.embedding vector column is missing")
            cur.execute("""SELECT indexname FROM pg_indexes WHERE schemaname='public' AND tablename='master_search_mastersearchindex' AND indexdef ILIKE '%USING hnsw%embedding%' LIMIT 1""")
            existing=cur.fetchone()
            if existing:
                print(f"HNSW already present: {existing[0]}")
                return
            print("Creating HNSW index concurrently...")
            cur.execute("""CREATE INDEX CONCURRENTLY master_search_msi_v6_embedding_hnsw_idx ON public.master_search_mastersearchindex USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)""")
            print("HNSW index created.")
    finally:
        conn.close()
if __name__=='__main__': main()
