# Hozpitality AI Search V6 — Final Release

## Release
V6.1 final global-search release.

### Critical fixes
- Entity words (`jobs`, `articles`, `professionals`, etc.) are removed from semantic keyword retrieval once an entity is detected.
- Location detection is data-driven from `master_search_mastersearchindex.location_text`.
- Location classification is conservative so content words such as `chef` are not incorrectly treated as geography because of a few malformed rows.
- Keyword and location retrieval use separate candidate pools.
- PostgreSQL FTS handles normal terms.
- PostgreSQL `pg_trgm` handles misspellings such as `restarant`.
- Optional pgvector remains supported but is disabled by default.
- Final ranking combines title, keyword, location, FTS, trigram, vector, and entity signals.
- Source records continue to resolve dynamically through Django `ContentType` metadata.
- Sensitive source columns remain excluded from enrichment.
- V5 remains independent; V6 uses port 8085 by default.

## Expected query parsing

`chef jobs Dubai`

```json
{
  "entity": "job",
  "keywords": ["chef"],
  "location": ["dubai"]
}
```

`restarant jobs Dubai`

```json
{
  "entity": "job",
  "keywords": ["restarant"],
  "location": ["dubai"]
}
```

The second query intentionally keeps the misspelled keyword so trigram retrieval can match the correctly spelled indexed content.

## Deployment

Keep the production `.env` outside the release archive. Do not copy credentials from this archive.

```bash
cd /home/dev/hozpitality-ai-search/v6
python -m py_compile main.py src/vanna/hozpitality/global_search.py src/vanna/hozpitality/query_planner.py
python scripts/check_search_stack.py
```

Start V6 on 8085 and keep V5 on 8084.

```bash
python main.py
```

Health check:

```bash
curl http://127.0.0.1:8085/api/search/health
```

Functional checks:

```bash
curl --get --data-urlencode "q=chef jobs Dubai" http://127.0.0.1:8085/api/search
curl --get --data-urlencode "q=restarant jobs Dubai" http://127.0.0.1:8085/api/search
curl --get --data-urlencode "q=Marriott" http://127.0.0.1:8085/api/search
```

Do not stop or replace the V5 service on port 8084 while validating V6.
\n## Content Type Resolution Fix\n\nWhen multiple django_content_type rows share the same model name, V6 now prefers the `base` app for public master-search records. This prevents `job` from resolving to a secondary `job.job` content type when the master index uses `base.job`.\n