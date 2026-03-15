# import json
# import re
# import torch

# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# app = FastAPI()

# torch.set_grad_enabled(False)

# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# print("Loading model...")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
#     llm_int8_enable_fp32_cpu_offload=True
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     quantization_config=bnb_config,
#     device_map="auto",
#     dtype=torch.float16,
#     offload_folder="offload"
# )

# model.eval()

# print("Model ready")


# class IntentRequest(BaseModel):
#     query: str
#     last_suggestion: str | None = None


# class FormatRequest(BaseModel):
#     query: str
#     results: list


# def generate(prompt, tokens=160):

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=tokens,
#             temperature=0.2,
#             do_sample=True,
#             top_p=0.9,
#             repetition_penalty=1.1,
#             pad_token_id=tokenizer.eos_token_id
#         )

#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     return decoded[len(prompt):].strip()


# def safe_json(text):

#     matches = re.findall(r"\{.*?\}", text, re.S)

#     for m in matches:
#         try:
#             return json.loads(m)
#         except:
#             continue

#     return None



# def detect_intent(query):

#     prompt = f"""
# You are an AI assistant for a hospitality platform.

# User Query:
# "{query}"

# Categories:
# job, article, useraccount, faq, company, event, supplier, product

# Tasks:
# 1 Detect intent: SEARCH, FAQ or CHAT
# 2 Extract keywords
# 3 Choose category

# Return JSON:

# {{
# "intent":"SEARCH|FAQ|CHAT",
# "type":"job|article|useraccount|faq|company|event|supplier|product",
# "keywords":"..."
# }}
# """

#     text = generate(prompt,120)

#     data = safe_json(text)

#     if data:
#         return data

#     return {
#         "intent":"SEARCH",
#         "type":"article",
#         "keywords":query.lower()
#     }



# def build_response(query, results):

#     prompt = f"""
# User query:
# {query}

# Search results JSON:
# {json.dumps(results)}

# Tasks:
# 1 Write short professional intro
# 2 Suggest 3 follow-up questions
# 3 Format final answer in HTML

# Allowed HTML:
# <p><ul><li><strong><a>

# Return JSON:

# {{
# "intro":"...",
# "suggestions":["...","...","..."],
# "html":"..."
# }}
# """

#     text = generate(prompt,220)

#     data = safe_json(text)

#     if data:
#         return data

#     return {
#         "intro":"Hospitality industry information.",
#         "suggestions":[],
#         "html":"<p>Hospitality industry information.</p>"
#     }


# @app.get("/")
# def health():
#     return {"status":"ok"}


# @app.post("/intent")
# def intent(req: IntentRequest):

#     return detect_intent(req.query)


# @app.post("/format")
# def format(req: FormatRequest):

#     return build_response(req.query, req.results)


import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

LLM_URL = "http://127.0.0.1:8080/completion"


# -----------------------------
# REQUEST MODELS
# -----------------------------

class IntentRequest(BaseModel):
    query: str


class SummaryRequest(BaseModel):
    query: str


class HtmlRequest(BaseModel):
    results: list


class GenerateRequest(BaseModel):
    query: str
    results: list


# -----------------------------
# LLM GENERATION FUNCTION
# -----------------------------

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

# -----------------------------
# SAFE JSON PARSER
# -----------------------------

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


# ------------------------------------------------
# INTENT API
# ------------------------------------------------

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


# ------------------------------------------------
# SUMMARY API
# ------------------------------------------------

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


# ------------------------------------------------
# HTML GENERATION API
# ------------------------------------------------

def generate_html(results):

    # limit results to 5
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

    # Build real HTML list for the LLM to reuse
    jobs_html = ""
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "#")
        jobs_html += f'<li><a href="{url}">{title}</a></li>\n'

    prompt = f"""
You are an AI assistant for a hospitality platform.

User query:
{req.query}

Below are the REAL search results.

You MUST use these results exactly and DO NOT change job titles.

Results:

<ul>
{jobs_html}
</ul>

Instructions:

1. Do NOT invent job titles.
2. Do NOT modify URLs.
3. Do NOT create template syntax like {{}} or {{% %}}.
4. Use the exact results shown above.

Output HTML format:

<p>Short introduction explaining the results.</p>

<p>Here are some relevant opportunities:</p>

<ul>
{jobs_html}
</ul>

<p>You may also want to explore:</p>

<ul>
<li>Follow-up question</li>
<li>Follow-up question</li>
<li>Follow-up question</li>
</ul>

Return ONLY HTML.
"""

    html = run_llm(prompt, 500)

    if not html:
        html = f"""
<p>Here are some relevant hospitality opportunities:</p>
<ul>
{jobs_html}
</ul>
"""

    # cleanup safety
    html = html.replace("{{", "").replace("}}", "")
    html = html.replace("{%", "").replace("%}", "")

    return {"html": html}


@app.post("/html")
def html(req: HtmlRequest):
    return generate_html(req.results)




# ------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok"}