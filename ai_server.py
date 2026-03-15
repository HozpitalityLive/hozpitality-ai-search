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
import re
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

LLM_URL = "http://127.0.0.1:8080/completion"


class IntentRequest(BaseModel):
    query: str
    last_suggestion: str | None = None


class FormatRequest(BaseModel):
    query: str
    results: list


def generate(prompt, tokens=160):

    payload = {
        "prompt": f"[INST] {prompt} [/INST]",
        "n_predict": tokens,
        "temperature": 0.2,
        "top_p": 0.9
    }

    try:
        response = requests.post(LLM_URL, json=payload, timeout=60)

        print("LLM status:", response.status_code)

        if response.status_code != 200:
            print("LLM ERROR:", response.text)
            return ""

        data = response.json()

        return data.get("content","").strip()

    except Exception as e:
        print("LLM request failed:", str(e))
        return ""


def safe_json(text):

    try:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end != -1:
            return json.loads(text[start:end])

    except:
        pass

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

    text = generate(prompt,120)
    data = safe_json(text)

    if data:
        return data

    return {
        "intent": "SEARCH",
        "type": "article",
        "keywords": query.lower()
    }


def build_response(query, results):

    prompt = f"""
You are an AI assistant for a hospitality platform.

User query:
{query}

Search results JSON:
{json.dumps(results, indent=2)}

Instructions:

1. Write a short professional introduction about the results.
2. Generate an HTML list of the results.
3. Each result must contain a clickable link using the provided "url".
4. Use the provided "title" as the link text.
5. Do NOT invent URLs or titles.
6. Show maximum 5 results.
7. Format like a professional search answer similar to ChatGPT or Google.

Allowed HTML tags:
<p><ul><li><strong><a>

Example HTML format:

<p>Here are some hospitality jobs related to your search:</p>

<ul>
<li><strong><a href="/jobs/123">Job title</a></strong></li>
<li><strong><a href="/jobs/456">Another job title</a></strong></li>
</ul>

Return ONLY valid JSON:

{{
"intro":"short intro",
"suggestions":[
"question1",
"question2",
"question3"
],
"html":"generated html"
}}
"""

    text = generate(prompt, 800)

    print("\n========= RAW LLM OUTPUT =========")
    print(text)
    print("==================================\n")

    data = safe_json(text)

    if data:
        return data

    return {
        "intro": "Hospitality industry information.",
        "suggestions": [],
        "html": "<p>Hospitality industry information.</p>"
    }


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/intent")
def intent(req: IntentRequest):
    return detect_intent(req.query)


@app.post("/format")
def format(req: FormatRequest):
    return build_response(req.query, req.results)