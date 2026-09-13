"""Run a safe global-search query and print ranking/debug metadata."""
from __future__ import annotations
import argparse, os, json
from dotenv import load_dotenv
import psycopg2
from vanna.hozpitality.schema_intelligence import HozpitalitySchemaIntelligence
from vanna.hozpitality.global_search import GlobalSearchService

load_dotenv()

def connect():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'), port=os.getenv('POSTGRES_PORT','5432'),
        dbname=os.getenv('POSTGRES_DATABASE'), user=os.getenv('POSTGRES_USER'), password=os.getenv('POSTGRES_PASSWORD'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('query')
    p.add_argument('--limit', type=int, default=10)
    args=p.parse_args()
    schema=HozpitalitySchemaIntelligence(connect)
    service=GlobalSearchService({
        'host':os.getenv('POSTGRES_HOST'),'port':int(os.getenv('POSTGRES_PORT','5432')),
        'dbname':os.getenv('POSTGRES_DATABASE'),'user':os.getenv('POSTGRES_USER'),'password':os.getenv('POSTGRES_PASSWORD')}, schema)
    hits=service.search_hits(args.query,args.limit)
    print(json.dumps({'query':args.query,'stats':service.last_stats,'hits':hits}, default=str, indent=2))

if __name__ == '__main__': main()
