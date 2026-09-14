# Hozpitality AI Search V6 — Final

This release fixes the two production search failures found during validation:
- duplicate Django content types: canonical `base` content types are preferred.
- explicit locations are hard constraints in the bounded retrieval pool.
- trigram retrieval is token-aware for typo tolerance such as `restarant`.
- V6 cache namespace is bumped to `v6.3`.
- V5 on port 8084 is not modified.

Smoke test:
`python scripts/test_search_cases.py "chef jobs Dubai" "restarant jobs Dubai" "Marriott"`

Expected parsing for `chef jobs Dubai`:
entity=`job`, keywords=`chef`, location=`dubai`, content_type_id=`18`.
