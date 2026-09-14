# V6 Final Fix Notes

## Search parsing fix

The previous V6 location classifier used pg_trgm similarity against `location_text`.
That could incorrectly classify semantic words such as `chef` as locations when malformed
or noisy location values existed.

The final version classifies a token as a location only when it occurs as a whole word
inside normalized `location_text`.

Examples:

- `chef jobs Dubai` -> entity `job`, keyword `chef`, location `dubai`
- `restarant jobs Dubai` -> entity `job`, keyword `restarant`, location `dubai`
- `Marriott` -> global search, keyword `marriott`, no location

## Retrieval

- FTS searches semantic keywords only.
- Trigram retrieval supports typo-tolerant keywords and includes title, user_name,
  category_text, slug, content and ai_keywords.
- Location retrieval is separate.
- Entity filtering remains dynamic through Django content types.
- Cache namespace is bumped to `v6.2` so stale V6.1 zero-result responses are not reused.

## Production

V6 remains on port 8085. Do not stop the existing V5 service on port 8084.
