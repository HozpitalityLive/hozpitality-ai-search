# Hozpitality AI — Final Light Premium v3

This package contains the updated Hozpitality AI chat frontend and the backend source used for the AI search/chat service.

## Changes in this version

### 1. Dynamic greetings
Greeting responses are rotated so repeated "Hello"/"Hi" messages do not always receive the same introduction.

### 2. Clear AI identity, audience, role and limitations
Hozpitality AI now explains:
- who it is
- who it is intended to help
- what information it can search/answer
- that it works from available Hozpitality information
- that it does not invent missing details
- that it does not apply for jobs, contact people, or guarantee outcomes

### 3. FAQ answer-only experience
FAQ questions are answered directly. FAQ result cards and their links are hidden in the chat UI, so the user sees the answer instead of a list of FAQ links.

### 4. Light Hozpitality branding
- Light/white mode is the default.
- Purple + gold Hozpitality-inspired accents.
- Dark mode remains available as an optional toggle.
- Existing conversation/search functionality is preserved.

## Frontend files

`frontend/HozpitalityAIChat.tsx`
`frontend/chatProtocol.ts`
`frontend/safeHtml.ts`

Replace the existing chat component with `HozpitalityAIChat.tsx` and keep the protocol/sanitizer files aligned with the existing component imports.

## Backend files

The `ai_search/` directory is the backend source from the supplied project archive, with the updated:
- `ai_search/app/answers.py`

No production `.env` or credentials are included.

## Validation

- Backend Python source compiled successfully with `python -m compileall`.
- Frontend TSX parsed successfully with the TypeScript parser.
- The supplied backend test suite could not be executed in this isolated environment because `mongomock` is not installed here; this package does not change the project's dependency requirements.
