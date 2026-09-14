#!/usr/bin/env python3
"""Smoke-test V6 search parsing and retrieval against PostgreSQL."""
import os, sys
from pathlib import Path
BASE_DIR=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from dotenv import load_dotenv
load_dotenv(BASE_DIR/'.env')
from src.vanna.hozpitality.schema_intelligence import HozpitalitySchemaIntelligence
from src.vanna.hozpitality.global_search import GlobalSearchService
import psycopg2

cfg={
 "host":os.getenv("POSTGRES_HOST","127.0.0.1"),
 "port":int(os.getenv("POSTGRES_PORT","5432")),
 "dbname":os.getenv("POSTGRES_DATABASE"),
 "user":os.getenv("POSTGRES_USER"),
 "password":os.getenv("POSTGRES_PASSWORD"),
}
schema=HozpitalitySchemaIntelligence(lambda: psycopg2.connect(**cfg))
svc=GlobalSearchService(cfg,schema)
for q in sys.argv[1:] or ["chef jobs Dubai","restarant jobs Dubai","Marriott"]:
    print("\n===",q,"===")
    hits=svc.search_hits(q,10)
    print("stats:",svc.last_stats)
    for h in hits:
        print(h["id"],h["content_type_id"],h["title"],"|",h["location_text"])
