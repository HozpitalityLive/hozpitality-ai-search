# Hozpitality AI — UI & Chat Fixes

## Included in this release

### Chat layout
- Conversation scroll is controlled by the chat panel itself instead of `scrollIntoView()` on a page-level anchor.
- New/streaming messages auto-scroll to the latest message inside the conversation viewport.
- Short conversations are bottom-aligned above the composer instead of leaving a large unused gap below the messages.
- The fixed composer remains attached to the bottom of the panel.

### FAQ behavior
- FAQ answers are answer-only in the chat UI; FAQ result cards/links are not rendered.
- Removed the `Other FAQs that may help are listed below.` text.
- Exact FAQ matches continue to use the database answer.
- If no exact FAQ exists, the assistant can provide deterministic Hozpitality platform guidance for common workflows instead of showing loosely related FAQ cards.

### Generic platform guidance
Supported fallback guidance includes:
- applying for jobs
- finding jobs
- professional registration/profile
- posting jobs
- finding professionals
- finding companies/employers
- finding suppliers
- finding marketplace products
- finding articles
- finding events
- finding awards

The guidance is deliberately procedural and does not invent record-specific facts. When a specific database record is required, the assistant still uses the search/FAQ data.

### Intent detection
Imperative information requests such as:
- `give me steps to apply for job`
- `show me the process to post a job`
- `tell me the steps to register`

are classified as information/FAQ requests rather than record searches.

### Greetings
Greetings rotate without immediately repeating the previous greeting within the same process.
