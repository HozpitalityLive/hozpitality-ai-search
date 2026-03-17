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


class HtmlRequest(BaseModel):
    results: list


class GenerateRequest(BaseModel):
    query: str
    results: list




def generate(prompt, tokens=300):

    payload = {
        "prompt": f"[INST] {prompt} [/INST]",
        "n_predict": tokens,
        "temperature": 0.2,
        "top_p": 0.9
    }

    try:

        r = requests.post(LLM_URL, json=payload, timeout=60)

        print("LLM status:", r.status_code)

        if r.status_code != 200:
            print("LLM ERROR:", r.text)
            return ""

        data = r.json()

        return data.get("content", "").strip()

    except Exception as e:

        print("LLM request failed:", e)
        return ""


def run_llm(prompt, tokens=400):

    payload = {
        "prompt": f"[INST] {prompt} [/INST]",
        "n_predict": tokens,
        "temperature": 0.2,
        "top_p": 0.9
    }

    try:

        r = requests.post(LLM_URL, json=payload, timeout=60)

        print("LLM status:", r.status_code)

        if r.status_code != 200:
            print("LLM ERROR:", r.text)
            return ""

        data = r.json()

        return data.get("content", "").strip()

    except Exception as e:

        print("LLM request failed:", e)
        return ""


def safe_json(text):

    if not text:
        return None

    text = text.replace("```json", "").replace("```", "")

    try:

        start = text.index("{")
        end = text.rindex("}") + 1

        return json.loads(text[start:end])

    except Exception as e:

        print("JSON parse error:", e)
        print("RAW OUTPUT:", text)

    return None



def detect_intent(query):

    prompt = f"""
You are an AI assistant for a hospitality platform.

User Query:
"{query}"

Categories:
job, article, useraccount, faq, company, event, supplier, product

Tasks:
1 Detect intent: SEARCH, FAQ or CHAT
2 Extract keywords
3 Choose category

Return JSON:

{{
"intent":"SEARCH|FAQ|CHAT",
"type":"job|article|useraccount|faq|company|event|supplier|product",
"keywords":"..."
}}
"""

    text = generate(prompt, 120)

    print("INTENT RAW:", text)

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
You are an AI assistant for a hospitality search platform.

User query:
{query}

Write:
1 Short professional introduction
2 Suggest 3 follow-up questions

Return JSON:

{{
"intro":"text",
"suggestions":["q1","q2","q3"]
}}
"""

    text = generate(prompt, 200)

    print("SUMMARY RAW:", text)

    data = safe_json(text)

    if data:
        return data

    return {
        "intro": "Here are some relevant hospitality results.",
        "suggestions": []
    }


@app.post("/summary")
def summary(req: SummaryRequest):
    return generate_summary(req.query)


def generate_html(results):

    results = results[:5]

    prompt = f"""
You are formatting hospitality search results.

Results JSON:
{json.dumps(results, indent=2)}

Task:

Convert the results into HTML.

Rules:

- Use ONLY title and url from results
- Do NOT invent data
- Do NOT use placeholders like {{ }}
- Insert the real URL inside <a href="">
- Show maximum 5 results

HTML format example:

<p>Here are some job results:</p>

<ul>
<li>
<strong>
<a href="/jobs/123">Housekeeping Attendant</a>
</strong>
</li>

<li>
<strong>
<a href="/jobs/456">Room Attendant</a>
</strong>
</li>
</ul>

Return JSON:

{{
"html":"generated html"
}}
"""

    text = generate(prompt, 500)

    print("HTML RAW:", text)

    data = safe_json(text)

    if data:
        return data

    return {
        "html": "<p>No results found.</p>"
    }

@app.post("/generate")
def generate_answer(req: GenerateRequest):

    results = req.results[:5]

    jobs_html = ""
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "#")
        jobs_html += f'<li><a href="{url}">{title}</a></li>\n'

    results_block = f"<ul>\n{jobs_html}</ul>"

    intro_prompt = f"""
You are the official AI assistant for Hozpitality.com.

User query:
{req.query}

Write ONE short introduction paragraph about the search results.

Rules:
- Maximum 4 lines
- Do NOT list jobs
- Do NOT ask questions
- Use only <p> tag

Return only HTML.
"""

    intro_html = run_llm(intro_prompt, 120)

    if not intro_html:
        intro_html = "<p>Here are some hospitality opportunities that match your search on Hozpitality.</p>"


    follow_prompt = f"""
User searched for:
{req.query}

Write 3 short follow-up questions related to hospitality jobs.

Rules:
- Use <p> tags only
- Do NOT mention other job portals
- Do NOT list jobs

Example:

<p>You may also want to explore:</p>
<p>Question</p>
<p>Question</p>
<p>Question</p>

Return only HTML.
"""

    follow_html = run_llm(follow_prompt, 420)

    if not follow_html:
        follow_html = """
<p>You may also want to explore:</p>
<p>Would you like to see similar hospitality jobs in other locations?</p>
<p>Are you interested in management or entry-level roles?</p>
<p>Would you like to explore jobs from top hotel brands?</p>
"""


    html = f"""
{intro_html}

<p>Here are some relevant opportunities on Hozpitality:</p>

{results_block}

{follow_html}
"""

    return {"html": html}


@app.post("/html")
def html(req: HtmlRequest):
    return generate_html(req.results)


@app.get("/")
def health():
    return {"status": "ok"}