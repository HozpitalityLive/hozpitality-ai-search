# Hozpitality AI Search — Phase 1

MongoDB-first lexical search foundation for V6.

## Data source

`master_search_mastersearchindex.ai_search_text` is the canonical flattened
search document. Relationship-aware fields are already materialized by the V6
backfill into `ai_search_text` and `metadata`.

The MongoDB bootstrap is therefore model-agnostic. It does not re-query Jobs,
Professionals, Companies, Products, Articles, Events, Awards and FAQs
individually.

## Runtime

The preferred production integration is the existing V6 FastAPI service on
port `8085`.

Routes added:

- `GET /search`
- `POST /search`
- `GET /search/health`

A standalone FastAPI app is also available:

```bash
uvicorn ai_search.app.main:app --host 127.0.0.1 --port 8090
```

## MongoDB

Configure `ai_search/.env`:

```env
MONGODB_URI=mongodb://mongoAdmin:PASSWORD@HOST:27017/hozpitality?authSource=admin
MONGODB_DATABASE=hozpitality
MONGODB_COLLECTION=search_documents
```

## Indexes

```bash
python -m ai_search.scripts.init_indexes
```

## Build Mongo documents from canonical V6 search documents

Do not run this until `ai_search_text` and `metadata` are populated:

```bash
python -m ai_search.scripts.sync_from_master_index --confirm
```

Test with a small sample first:

```bash
python -m ai_search.scripts.sync_from_master_index --confirm --limit 100
```

## Search examples

```bash
curl "http://127.0.0.1:8085/search?q=chef"
curl "http://127.0.0.1:8085/search?q=excutive%20chef&entity=job&city=Dubai"
curl "http://127.0.0.1:8085/search?q=hotel&entity=company"
curl "http://127.0.0.1:8085/search/health"
```

The API never returns more than five results.

Phase 1 does not use an LLM, vector search, conversation memory, or frontend
rendering.
