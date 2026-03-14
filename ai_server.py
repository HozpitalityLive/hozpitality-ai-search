import json
import re
import torch

from fastapi import FastAPI
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

app = FastAPI()

torch.set_grad_enabled(False)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16
)

model.config.use_cache = True
model.eval()

print("Model ready on GPU")


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


class IntentRequest(BaseModel):
    query: str
    last_suggestion: str | None = None


class FormatRequest(BaseModel):
    query: str
    intro: str
    results: list
    suggested_query: str


def generate(prompt, tokens=120):

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded[len(prompt):]


def safe_json(text):

    matches = re.findall(r"\{.*?\}", text, re.S)

    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    return None


def detect_intent(query, last_suggestion=None):

    prompt = f"""
Classify hospitality search query.

Query: "{query}"

Return JSON only:

{{
"intent":"FAQ|SEARCH|CHAT",
"keywords":"...",
"type":"{','.join(TYPE_MAPPING)}",
"intro":"...",
"suggested_query":"..."
}}
"""

    text = generate(prompt)

    data = safe_json(text)

    if data:
        return data

    return {
        "intent": "SEARCH",
        "keywords": query.lower(),
        "type": "article",
        "intro": "Hospitality industry information.",
        "suggested_query": f"Would you like to know more about {query}?"
    }


def format_html(query, intro, results, suggested):

    prompt = f"""
Create HTML response.

Intro:
{intro}

Results JSON:
{json.dumps(results)}

Suggested question:
{suggested}

Allowed HTML tags:
<p><ul><li><strong><a>

Return HTML only.
"""

    text = generate(prompt, 160)

    match = re.search(r"<p>.*", text, re.S)

    if match:
        return match.group()

    return f"<p>{intro}</p>"


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/intent")
def intent(req: IntentRequest):

    return detect_intent(req.query, req.last_suggestion)


@app.post("/format")
def format(req: FormatRequest):

    html = format_html(
        req.query,
        req.intro,
        req.results,
        req.suggested_query
    )

    return {"html": html}