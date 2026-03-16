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

Categories:
job, article, useraccount, faq, company, event, supplier, product

Rules:
- If the user searches for jobs, careers, hiring, vacancies → type = job
- If the query contains job titles (chef, waiter, manager, bartender etc) → type = job
- If the query contains location + job → type = job
- If user asks a question → intent = FAQ
- Otherwise → SEARCH

Return ONLY JSON.

Example:

{{
"intent":"SEARCH",
"type":"job",
"keywords":"chef jobs mumbai"
}}
"""

    text = generate(prompt, 120)

    data = safe_json(text)

    if data:
        return data

    return {
        "intent": "SEARCH",
        "type": "job",
        "keywords": query.lower()
    }


@app.post("/intent")
def intent(req: IntentRequest):
    return detect_intent(req.query)


def generate_summary(query):

    prompt = f"""
You are the official AI assistant for Hozpitality.com.

User search:
{query}

Write HTML.

Rules:
- First write ONE intro paragraph
- Then write suggestions
- Use <p> tags only
- Do not list jobs

Example:

<p>Here are some hospitality opportunities related to chef jobs in Mumbai.</p>

<p>You may also want to explore:</p>
<p>Chef jobs in Dubai</p>
<p>Hotel chef careers in India</p>
<p>Sous chef positions in luxury hotels</p>

Return JSON:

{{
"intro_html":"<p>intro text</p>",
"suggestions_html":"<p>suggestion block</p>"
}}
"""

    text = generate(prompt, 220)

    data = safe_json(text)

    if data:
        return data

    return {
        "intro_html": "<p>Here are some relevant hospitality opportunities.</p>",
        "suggestions_html": ""
    }


@app.post("/summary")
def summary(req: SummaryRequest):
    return generate_summary(req.query)


@app.get("/")
def health():
    return {"status": "ok"}