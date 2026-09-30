# Total Match Count Update

## Behavior
For a search that returns a visible page of up to 5 ranked records, the assistant now distinguishes the authoritative total match count from the visible top results.

Example:

> I found 240 waiter jobs. Here are the top 5 results.

The total is calculated independently of the UI result limit using MongoDB's existing text index and the same entity/location/status/live/structured filters. The ranked `results` array remains the top page.

## API
`ChatResponse.total` now contains the total lexical match count. `answer_presentation` and `suggestions` remain unchanged.

## LLM grounding
The LLM receives:
- `total_matches`
- `shown_result_count`
- `shown_results_are_top_ranked: true`
- the top ranked result records

The prompt explicitly prohibits treating the displayed page size as the total.

## Fallback
If the LLM is unavailable, the deterministic fallback uses the same distinction:
`I found 240 waiter jobs. Here are the top 5 results.`

## Validation
Python compilation passes for the modified modules. Full pytest execution requires the project's missing `mongomock` dependency in this environment.
