import json
import re
import time
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

print("Loading model...")

model_id = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Model loaded")


TYPE_MAPPING = [
    "job", "article", "useraccount",
    "faq", "company", "event", "supplier", "product"
]


class Query(BaseModel):
    query: str


class FormatRequest(BaseModel):
    query: str
    intro: str
    results: list
    suggested_query: str



def generate_ai_response(query):

    prompt = f"""
Hozpitality Portal Expert Query: "{query}"

Rules:
1. Identify intent: FAQ, SEARCH, CHAT.
2. Identify type from: {TYPE_MAPPING}
3. Provide professional intro
4. Suggested query must be:
"Would you like to know more about [Topic]?"

Respond ONLY JSON:

{{
 "intent":"",
 "keywords":"",
 "type":"",
 "intro":"",
 "suggested_query":""
}}
"""

    try:

        print("STEP 1: Tokenizing")

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        print("STEP 2: Generating")

        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )

        print("Generation time:", time.time() - start)

        print("STEP 3: Decoding")

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("MODEL OUTPUT:")
        print(text)

        print("STEP 4: Extracting JSON")

        matches = re.findall(r'\{[^{}]*\}', text)

        for m in matches:
            try:
                data = json.loads(m)

                if (
                    isinstance(data, dict)
                    and data.get("intent")
                    and data.get("type")
                ):
                    print("Valid JSON Found:", data)
                    return data

            except Exception:
                continue

        print("STEP 5: JSON not found, returning fallback")

    except Exception as e:
        print("LLM ERROR:", str(e))

    return {
        "intent": "SEARCH",
        "keywords": query,
        "type": "article",
        "intro": f"{query} is an important topic within the hospitality industry.",
        "suggested_query": f"Would you like to know more about {query}?"
    }



def generate_html_response(query, intro, results, suggested_query):

    prompt = f"""
You are the official AI assistant of Hozpitality.

Create a professional chat response in HTML.

Rules:
- Use simple HTML only: <p>, <ul>, <li>, <strong>, <a>
- Do not use markdown
- Start with intro paragraph
- Then list results
- End with suggested question
- Return ONLY HTML

User Query:
{query}

Intro:
{intro}

Results:
{json.dumps(results)}

Suggested Query:
{suggested_query}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    html_match = re.search(r'<p>.*', text, re.DOTALL)

    if html_match:
        return html_match.group()

    return f"<p>{intro}</p>"



@app.get("/")
def home():
    return {"message": "Hozpitality AI Assistant running"}


@app.post("/intent")
def detect_intent(data: Query):

    result = generate_ai_response(data.query)

    return result


@app.post("/format")
def format_chat(data: FormatRequest):

    html = generate_html_response(
        data.query,
        data.intro,
        data.results,
        data.suggested_query
    )

    return {"html": html}