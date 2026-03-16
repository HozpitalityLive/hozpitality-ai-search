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

class KeywordGenRequest(BaseModel):
    title: str
    content: str


def generate(prompt, tokens=200):

    payload = {
        "prompt": f"[INST] {prompt} [/INST]",
        "n_predict": tokens,
        "temperature": 0.2,
        "top_p": 0.9
    }

    print("\n================ LLM REQUEST ================")
    print("PROMPT:")
    print(prompt)
    print("TOKENS:", tokens)
    print("=============================================\n")

    try:

        r = requests.post(LLM_URL, json=payload, timeout=60)

        print("LLM STATUS:", r.status_code)

        if r.status_code != 200:
            print("LLM ERROR:", r.text)
            return ""

        data = r.json()

        print("\n============= LLM RAW RESPONSE =============")
        print(data)
        print("============================================\n")

        content = data.get("content", "").strip()

        print("LLM CONTENT:", content)

        return content

    except Exception as e:

        print("LLM REQUEST FAILED:", e)

        return ""


def safe_json(text):

    print("\n----- SAFE JSON INPUT -----")
    print(text)
    print("---------------------------")

    if not text:
        print("JSON ERROR: empty text")
        return None

    text = text.replace("```json", "").replace("```", "").strip()

    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        print("JSON ERROR: no JSON object found")
        return None

    try:

        parsed = json.loads(match.group())

        print("JSON PARSED:", parsed)

        return parsed

    except Exception as e:

        print("JSON PARSE ERROR:", e)

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

1 short introduction paragraph about the search results.

Rules:
- Maximum 4 lines
- Use <p> tag
- Do not list jobs

Then write suggestions.

Format exactly like this:

<p>intro text</p>

<p>You may also want to explore:</p>
<p>Suggestion</p>
<p>Suggestion</p>
<p>Suggestion</p>
"""

    text = generate(prompt, 180)

    print("SUMMARY RAW:", text)

    if not text:
        return "", ""

    parts = text.split("You may also want to explore:")

    if len(parts) > 1:
        intro_html = parts[0].strip()
        suggestions_html = "<p>You may also want to explore:</p>" + parts[1]
    else:
        intro_html = text
        suggestions_html = ""

    return intro_html, suggestions_html


@app.post("/generate_keywords")
def generate_keywords(req: KeywordGenRequest):
    prompt = f"Generate 10 relevant SEO keywords for hospitality content. Return comma-separated list.\nTitle: {req.title}\nContent: {req.content[:500]}"
    keywords_text = generate(prompt, tokens=100)
    return {"keywords": [k.strip() for k in keywords_text.split(",") if k.strip()]}


@app.post("/summary")
def summary(req: SummaryRequest):
    return generate_summary(req.query , req.titles)


@app.get("/")
def health():
    return {"status": "ok"}