# Hozpitality AI Search — Phase 1

MongoDB-first search service for Hozpitality V6.

## Scope

Phase 1 provides:

- FastAPI `/search` endpoint
- MongoDB connection pooling through `pymongo.MongoClient`
- `search_documents` collection
- MongoDB text and filter indexes
- Jobs, professionals, companies, products, articles, events, awards and FAQs
- exact, phrase and keyword ranking
- alias matching
- location/entity/status filters
- typo correction using `rapidfuzz`
- strict maximum of 5 results
- deterministic search; no LLM calls

## Run

From the `v6` directory:

```bash
pip install -r ai_search/requirements.txt
cp ai_search/.env.example ai_search/.env
uvicorn ai_search.app.main:app --host 0.0.0.0 --port 8090 --reload
```

Health:

```bash
curl http://127.0.0.1:8090/health
```

Search:

```bash
curl "http://127.0.0.1:8090/search?q=excutive%20chef&entity=job&city=Dubai"
```

## Document shape

The service expects one normalized MongoDB document per searchable entity:

```json
{
  "entity_type": "job",
  "entity_id": "123",
  "title": "Executive Chef",
  "description": "Leading hotel kitchen...",
  "keywords": ["chef", "culinary", "hotel"],
  "aliases": ["executive cook"],
  "location": {
    "city": "Dubai",
    "country": "UAE"
  },
  "category": "Culinary",
  "status": "active",
  "is_live": true,
  "url": "/jobs/executive-chef-123",
  "image": null
}
```

`search_text` is optional; the sync script creates it when loading from the existing V6 PostgreSQL master search index.

## PostgreSQL -> MongoDB bootstrap

The existing V6 project is PostgreSQL-first. This phase intentionally moves retrieval to MongoDB while preserving the canonical V6 `master_search_mastersearchindex` as a source for the initial index build.

```bash
python ai_search/scripts/sync_from_postgres.py --confirm
```

The sync reads only the searchable master-index fields and writes normalized `search_documents`. It does not copy passwords, emails, phone numbers or other sensitive account fields.

For incremental operation, run the script on a schedule or replace it later with source-event/change-stream ingestion.

## Search behavior

1. Normalize query.
2. Remove stopwords.
3. Correct likely misspellings against known vocabulary.
4. Search exact/phrase/keyword matches in MongoDB.
5. Apply hard entity/location/status filters.
6. Rank exact title/phrase/alias/keyword/location matches above general content matches.
7. Return no more than five results.

No LLM is involved in Phase 1.
