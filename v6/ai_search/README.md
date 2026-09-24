# Hozpitality AI Search — Phase 1

MongoDB-first lexical search foundation integrated into the existing V6 FastAPI service on port `8085`.

## Production data source

Phase 1 searches the existing MongoDB collection:

- Database: `mongoAdmin`
- Collection: `search_documents`
- Documents: approximately `326,596`
- Search corpus: `ai_search_text`
- Existing text index: `idx_ai_search_text`

The collection already contains Jobs, Professionals, Companies, Products, Articles, Events, Awards and FAQs. No PostgreSQL synchronization or new search-document build is required for Phase 1.

## Endpoints

- `GET /search`
- `POST /search`
- `GET /search/health`

The routes are registered into the existing V6 FastAPI app, so there is no separate Phase 1 server or port.

## Examples

```bash
curl "http://127.0.0.1:8085/search?q=chef"
curl "http://127.0.0.1:8085/search?q=excutive%20chef&entity=job&city=Dubai"
curl "http://127.0.0.1:8085/search?q=hotel&entity=company"
curl "http://127.0.0.1:8085/search?q=chef&country=UAE"
curl "http://127.0.0.1:8085/search?q=chef&is_live=true"
curl "http://127.0.0.1:8085/search/health"
```

## Phase 1 capabilities

- Keyword and phrase search
- Exact title matching
- Alias matching using `search_aliases`
- Keyword matching using `search_keywords` / `keywords`
- MongoDB `$text` retrieval from the existing `ai_search_text` index
- Typo correction with RapidFuzz
- Entity filtering
- City filtering
- Country filtering, including common aliases such as UAE/USA/UK
- Live filtering and expiry protection
- Status filtering across common document/status fields
- Deterministic ranking after MongoDB retrieval
- Maximum 5 API results

## Environment

Create `ai_search/.env` on the server and keep it out of Git:

```env
MONGODB_URI=mongodb://mongoAdmin:PASSWORD@10.5.140.74:27017/mongoAdmin?authSource=admin
MONGODB_DATABASE=mongoAdmin
MONGODB_COLLECTION=search_documents
MONGODB_MAX_POOL_SIZE=100
MONGODB_MIN_POOL_SIZE=5
MONGODB_SERVER_SELECTION_TIMEOUT_MS=5000
MONGODB_CONNECT_TIMEOUT_MS=5000
SEARCH_MAX_RESULTS=5
SEARCH_FUZZY_THRESHOLD=82
SEARCH_FUZZY_MIN_TOKEN_LENGTH=3
```

## Important

Do not run the old PostgreSQL/master-index synchronization scripts for Phase 1. Do not create a second MongoDB text index on this collection; MongoDB supports one text index per collection and the production `idx_ai_search_text` index is already the Phase 1 text-search index.
