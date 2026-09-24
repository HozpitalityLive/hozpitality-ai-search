# Hozpitality AI Search — Phase 2

Phase 2 adds natural-language query understanding on top of the Phase 1 MongoDB retrieval engine.

## Data source

The search path remains MongoDB-only:

- Database: `mongoAdmin`
- Collection: `search_documents`
- Search corpus: `ai_search_text`
- Existing MongoDB text index: `idx_ai_search_text`
- Maximum API results: 5

The existing V6 PostgreSQL/Vanna functionality remains separate. Phase 2 does not synchronize PostgreSQL data into the MongoDB search collection.

## Phase 2 flow

```text
Natural-language query
        |
        v
Deterministic query parser
        |
        +--> clear query ------------------+
        |                                  |
        +--> ambiguous/low confidence --> optional Gemini fallback
                                           |
                                           v
                                  Structured SearchPlan
                                           |
                                           v
                                  typo correction
                                           |
                                           v
                              MongoDB lexical retrieval
                                           |
                              +------------+------------+
                              |                         |
                         optional FAISS             lexical
                         semantic retrieval        candidates
                              |                         |
                              +------------+------------+
                                           |
                                           v
                                  hybrid reranking
                                           |
                                           v
                                      top 5
```

## Extracted fields

The parser can extract:

- intent
- entity
- keywords
- city
- country
- experience
- job level
- department
- industry
- category
- date range
- salary
- employment type
- verified
- featured
- currently working
- other structured filters

Example:

```text
I need a senior chef job in Dubai with 5 years experience
```

becomes:

```json
{
  "intent": "search",
  "entity": "job",
  "keywords": ["chef"],
  "city": "Dubai",
  "country": "United Arab Emirates",
  "experience": 5,
  "level": "senior"
}
```

## Clarification

The API does not execute an under-specified search when a useful clarification can be asked.

Examples:

```text
Find me a job
-> What type of job are you looking for?

Find chef jobs
-> Which location would you prefer?

Find chef jobs in Dubai
-> Search directly.
```

Clarification rules exist for:

- jobs
- professionals
- companies
- products
- articles
- events
- awards
- FAQs

## Spelling correction

Typos are corrected against the MongoDB title/alias/keyword vocabulary.

Example:

```text
excutive chef jobs in Dubai
```

becomes:

```text
executive chef
```

The original keyword query is also searched so a correction can never hide an exact match.

## Entity detection

Supported entities:

- job
- professional
- company
- product
- article
- event
- award
- faq

Natural aliases such as `vacancies`, `candidates`, `suppliers`, `news`, `marketplace`, and `FAQs` are recognized.

## Location detection

Common hospitality locations and country aliases are normalized, including:

- Dubai -> Dubai + United Arab Emirates
- UAE -> United Arab Emirates
- Mumbai -> Mumbai
- Bengaluru/Bangalore -> Bengaluru
- Abu Dhabi -> Abu Dhabi
- Riyadh -> Riyadh
- Doha -> Doha
- Singapore -> Singapore
- London -> London
- New York -> New York

Structured city/country filters are enforced after MongoDB retrieval as well as in the MongoDB query.

## Hybrid ranking

The existing deterministic lexical ranker remains authoritative for exact identity matches.

When the optional FAISS semantic index is enabled, semantic similarity is added as another ranking signal and candidates are marked with:

```text
semantic
```

Strong exact title/alias/keyword matches continue to outrank weak semantic similarity.

## Optional Gemini fallback

Gemini is NOT called for every query.

Set:

```env
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT_SECONDS=4
```

The deterministic parser handles clear searches first. Gemini is used only when the parser considers the query ambiguous or incomplete. If no key is configured, the deterministic parser remains fully functional.

## Optional semantic/vector search

The vector layer uses:

- `sentence-transformers/all-MiniLM-L6-v2`
- FAISS inner-product search with normalized embeddings
- MongoDB `search_documents` as the source corpus

Install:

```bash
source .venv/bin/activate
pip install -r ai_search/requirements-phase2-vector.txt
```

Build the index once:

```bash
python -m ai_search.scripts.build_vector_index
```

Then enable it:

```env
SEMANTIC_SEARCH_ENABLED=true
SEMANTIC_MODEL=sentence-transformers/all-MiniLM-L6-v2
SEMANTIC_INDEX_PATH=ai_search/data/search.faiss
SEMANTIC_IDS_PATH=ai_search/data/search_ids.json
```

The vector index is a retrieval accelerator; MongoDB remains the source of truth. Rebuild it after large changes to `ai_search_text`.

## API

### Search

```bash
curl --get 'http://127.0.0.1:8085/search' \
  --data-urlencode 'q=I need a senior chef job in Dubai with 5 years experience'
```

### Query understanding

```bash
curl --get 'http://127.0.0.1:8085/search/understand' \
  --data-urlencode 'q=I need a senior chef job in Dubai with 5 years experience'
```

### Clarification

```bash
curl --get 'http://127.0.0.1:8085/search' \
  --data-urlencode 'q=Find me a job'
```

### Health

```bash
curl http://127.0.0.1:8085/search/health
```

Health reports MongoDB connectivity, document count, and whether optional semantic/LLM layers are enabled.

## Environment

Keep credentials out of Git and do not put a production password in `.env.example`.

```env
MONGODB_URI=mongodb://USERNAME:PASSWORD@HOST:27017/mongoAdmin?authSource=admin
MONGODB_DATABASE=mongoAdmin
MONGODB_COLLECTION=search_documents

MONGODB_MAX_POOL_SIZE=100
MONGODB_MIN_POOL_SIZE=5
MONGODB_SERVER_SELECTION_TIMEOUT_MS=5000
MONGODB_CONNECT_TIMEOUT_MS=5000

SEARCH_MAX_RESULTS=5
SEARCH_FUZZY_THRESHOLD=82
SEARCH_FUZZY_MIN_TOKEN_LENGTH=3

GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT_SECONDS=4

SEMANTIC_SEARCH_ENABLED=false
SEMANTIC_MODEL=sentence-transformers/all-MiniLM-L6-v2
SEMANTIC_INDEX_PATH=ai_search/data/search.faiss
SEMANTIC_IDS_PATH=ai_search/data/search_ids.json
```

## Phase 2 boundaries

- No PostgreSQL synchronization is added.
- MongoDB `search_documents` remains the retrieval source of truth.
- No second MongoDB text index is created.
- Maximum API output remains 5.
- Semantic retrieval is optional and does not replace MongoDB.
- Gemini is optional and is not invoked for every request.
