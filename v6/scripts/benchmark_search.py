"""Read-only PostgreSQL benchmark for the V6 FTS path.

Use on a staging/production replica first. EXPLAIN ANALYZE executes the SELECT
but does not mutate data. It reports actual execution time and buffer usage.
"""
from __future__ import annotations
import argparse, os, json
import psycopg2
from dotenv import load_dotenv
load_dotenv()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('query')
    args=p.parse_args()
    conn=psycopg2.connect(host=os.getenv('POSTGRES_HOST'),port=os.getenv('POSTGRES_PORT','5432'),dbname=os.getenv('POSTGRES_DATABASE'),user=os.getenv('POSTGRES_USER'),password=os.getenv('POSTGRES_PASSWORD'))
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout='30s'")
            cur.execute("""EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                SELECT id, object_id, content_type_id, title
                FROM public.master_search_mastersearchindex
                WHERE (is_live = TRUE OR is_live IS NULL)
                  AND (expires_at IS NULL OR expires_at >= CURRENT_TIMESTAMP)
                  AND search_vector_v6 @@ plainto_tsquery('simple', unaccent(%s))
                ORDER BY ts_rank_cd(search_vector_v6, plainto_tsquery('simple', unaccent(%s))) DESC, id DESC
                LIMIT 20""", (args.query,args.query))
            plan=cur.fetchone()[0][0]
            print(json.dumps(plan, indent=2))
    finally: conn.close()
if __name__=='__main__': main()
