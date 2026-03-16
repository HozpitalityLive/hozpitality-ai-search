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

    text = text.replace("```json", "").replace("```", "")

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except:
        return None


def detect_intent(query):

    prompt = f"""
You are an AI assistant for a hospitality platform.

User Query:
"{query}"

Categories:
job, article, useraccount, faq, company, event, supplier, product

Return JSON:

{{
"intent":"SEARCH|FAQ|CHAT",
"type":"job|article|useraccount|faq|company|event|supplier|product",
"keywords":"..."
}}
"""

    text = generate(prompt, 120)

    data = safe_json(text)

    if data:
        return data

    return {
        "intent": "SEARCH",
        "type": "article",
        "keywords": query.lower()
    }


@app.post("/intent")
def intent(req: IntentRequest):
    return detect_intent(req.query)


def generate_summary(query):

    prompt = f"""
You are the official AI assistant for Hozpitality.com.

User query:
{query}

Write HTML only.

Rules:
- Intro must be inside <p>
- Suggestions must use <p> tags
- Do not include job listings

Format example:

<p>Short introduction text</p>

<p>You may also want to explore:</p>
<p>Question 1</p>
<p>Question 2</p>
<p>Question 3</p>

Return JSON:

{{
"intro_html":"<p>text</p>",
"suggestions_html":"<p>text</p>"
}}
"""

    text = generate(prompt, 200)

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