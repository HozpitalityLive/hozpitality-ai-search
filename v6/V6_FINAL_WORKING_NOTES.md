# Hozpitality AI Search V6 — Final Working Fix

## Search behavior
- `chef jobs Dubai` parses as entity=`job`, keyword=`chef`, location=`dubai`.
- Job resolves to canonical `base.job` content type when duplicate Django model names exist.
- Explicit locations are hard-filtered against `location_text`; title/content fallback is only allowed when location_text is empty.
- Common restaurant typos (`restarant`, `restraunt`, `resturant`, etc.) normalize to `restaurant` before lexical/trigram retrieval.
- Global searches such as `Marriott` remain unrestricted by content type.
- Cache namespace is `v6.4` to invalidate prior bad search results.
