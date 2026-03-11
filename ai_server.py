import json
import re
import time
import torch

from fastapi import FastAPI
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

app = FastAPI()

print("Loading Mistral 7B...")

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)

print("Model Loaded Successfully")


TYPE_MAPPING = [
    "job",
    "article",
    "useraccount",
    "faq",
    "company",
    "event",
    "supplier",
    "product"
]


class Query(BaseModel):
    query: str


class FormatRequest(BaseModel):
    query: str
    intro: str
    results: list
    suggested_query: str


def ask_llm(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text



def generate_ai_response(query):

    prompt = f"""
You are AI assistant of Hozpitality.com.

User query:
"{query}"

Tasks:

1 Detect intent:
FAQ
SEARCH
CHAT

2 Detect content type from list:
{TYPE_MAPPING}

3 Extract search keywords

4 Write short professional intro

5 Create suggested follow-up question.

Return ONLY JSON:

{{
"intent":"",
"keywords":"",
"type":"",
"intro":"",
"suggested_query":""
}}
"""

    try:

        text = ask_llm(prompt)

        matches = re.findall(r'\{[^{}]*\}', text)

        for m in matches:
            try:
                data = json.loads(m)

                if data.get("intent") and data.get("type"):
                    return data

            except:
                continue

    except Exception as e:
        print("LLM ERROR:", str(e))

    return {
        "intent": "SEARCH",
        "keywords": query,
        "type": "article",
        "intro": f"{query} is an important topic in the hospitality industry.",
        "suggested_query": f"Would you like to know more about {query}?"
    }



def generate_html_response(query, intro, results, suggested_query):

    prompt = f"""
You are AI assistant of Hozpitality.

Create professional HTML response.

Rules:

Use only HTML tags:
<p>
<ul>
<li>
<strong>
<a>

Structure:

Intro paragraph

Then list results

Then suggested question

Return ONLY HTML.

User Query:
{query}

Intro:
{intro}

Results:
{json.dumps(results)}

Suggested Question:
{suggested_query}
"""

    text = ask_llm(prompt)

    html_match = re.search(r'<p>.*', text, re.DOTALL)

    if html_match:
        return html_match.group()

    return f"<p>{intro}</p>"


@app.get("/")
def home():
    return {"message": "Hozpitality AI running"}


@app.post("/intent")
def detect_intent(data: Query):

    return generate_ai_response(data.query)


@app.post("/format")
def format_chat(data: FormatRequest):

    html = generate_html_response(
        data.query,
        data.intro,
        data.results,
        data.suggested_query
    )

    return {"html": html}