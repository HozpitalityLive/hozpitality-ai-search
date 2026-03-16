import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

LLM_URL = "http://127.0.0.1:8080/completion"


class IntentRequest(BaseModel):
    query: str


class SummaryRequest(BaseModel):
    query: str
    titles: list = []


def generate(prompt, tokens=200):

    payload = {
        "prompt": f"[INST] {prompt} [/INST]",
        "n_predict": tokens,
        "temperature": 0.2,
        "top_p": 0.9
    }

    try:
        r = requests.post(LLM_URL, json=payload, timeout=60)

        if r.status_code != 200:
            return ""

        data = r.json()

        return data.get("content", "").strip()

    except Exception:
        return ""


def safe_json(text):

    if not text:
        return None

    text = text.replace("```json", "").replace("```", "").strip()

    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group())
    except:
        return None


def detect_intent(query):

    prompt = f"""
You are an AI intent classifier for a hospitality platform.

User query:
{query}

Tasks:
1. Detect intent (SEARCH, FAQ, CHAT)
2. Detect content type (job, article, useraccount, faq, company, event, supplier, product)
3. Rewrite the query removing stop words like: in, the, for, at, near, jobs, job
4. Extract location if present (city, state, or country)

Examples:

Query: chef jobs in mumbai
Output:
{{
"intent":"SEARCH",
"type":"job",
"keywords":"chef",
"location":"mumbai"
}}

Query: hotel manager jobs in dubai
Output:
{{
"intent":"SEARCH",
"type":"job",
"keywords":"hotel manager",
"location":"dubai"
}}

Query: hospitality news
Output:
{{
"intent":"SEARCH",
"type":"article",
"keywords":"hospitality news",
"location":""
}}

Return JSON only.
"""

    text = generate(prompt, 140)

    data = safe_json(text)

    if data:
        return data

    return {
        "intent": "SEARCH",
        "type": "job",
        "keywords": query.lower(),
        "location": ""
    }


@app.post("/intent")
def intent(req: IntentRequest):
    return detect_intent(req.query)


def generate_summary(query, titles):

    prompt = f"""
You are the AI assistant for Hozpitality.com.

User search:
{query}

Top search result titles:
{titles}

Write:

1 short introduction about the results.

Rules:
- Maximum 4 lines
- Use ONE <p> tag
- Do not list jobs
- Mention hospitality careers if relevant

Then write suggestions.

Format:

<p>intro text</p>

<p>You may also want to explore:</p>
<p>Question</p>
<p>Question</p>
<p>Question</p>

Return JSON:

{{
"intro_html":"<p>text</p>",
"suggestions_html":"<p>suggestions</p>"
}}
"""

    text = generate(prompt, 180)

    data = safe_json(text)

    if data:
        return data

    return {
        "intro_html": "<p>Explore hospitality opportunities related to your search on Hozpitality.</p>",
        "suggestions_html": ""
    }


@app.post("/summary")
def summary(req: SummaryRequest):
    return generate_summary(req.query)


@app.get("/")
def health():
    return {"status": "ok"}