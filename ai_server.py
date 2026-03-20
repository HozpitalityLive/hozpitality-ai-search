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


def generate(prompt, tokens=300):

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
    import json, re

    if not text:
        return None

    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{.*", text, re.DOTALL)
    if not match:
        return None

    partial = match.group()

    try:
        partial = partial.rstrip(", \n")
        if not partial.endswith("}"):
            partial += '"}'  
        return json.loads(partial)
    except:
        return None

import re

def detect_intent(query):
    categories = ['job', 'article', 'professional', 'faq', 'company', 'event', 'supplier', 'product', 'awards']

    prompt = f"""
You are an AI intent classifier for a hospitality platform.
Hospitality Portal Expert. 
User Query: "{query}"
Available Categories: {categories}

STRICT RULES:
0. TYPO CORRECTION: Fix minor typos (e.g., 'kitchn' -> 'kitchen', 'dishwashr' -> 'dishwasher'). Normalize to industry terms.

1. INTENT LOGIC (CRITICAL):
    - FAQ RULE (TOP PRIORITY): If the query starts with "how to", "how do i", "how can i", "steps to", or "process", intent MUST be 'FAQ' and type MUST be 'faq'. DO NOT classify as 'job' even if 'job' or 'apply' is mentioned.
    - PROFILE RULE: If the query is a person's name (e.g., 'Yuni Hunter', 'Raj Bhatt') OR starts with "who is" or "who's", intent MUST be 'SEARCH' and type MUST be 'professional'.
    
    - COMPANY RULE: If query contains "what is", "define", or "hozpitality", intent MUST be 'SEARCH' and type MUST be 'company'.
    - Default: Intent 'SEARCH', type 'article'.

2. KEYWORD EXTRACTION (STRICT):
    - Output keywords in LOWERCASE only.
    - REMOVE category words from keywords if they are used as 'type' (e.g., if type is 'job', REMOVE 'jobs', 'job', 'openings', 'opening', 'vacancies' from keywords).
    
    - For FAQ: Extract only the core topic (e.g., 'account deletion'). REMOVE filler words like "how to", "how do i", "apply", "job" from FAQ keywords.
    
    - For SEARCH: REMOVE category words like 'articles', 'article', 'events' from keywords if they match the 'type'.

User query:
{query}

Tasks:
1. Detect intent (SEARCH, FAQ, CHAT)
2. Detect content type (Ensure 'professional' for names, 'faq' for procedures)
3. Rewrite the query removing stop words.
4. Extract location if present.

Examples:

Query: Yuni Hunter
Output:
{{
"intent":"SEARCH",
"type":"professional",
"keywords":"yuni hunter",
"location":""
}}

# UPDATED: Added explicit "How to apply" example to guide the model
Query: How to apply for a job on Hozpitality?
Output:
{{
"intent":"FAQ",
"type":"faq",
"keywords":"job application",
"location":""
}}

Query: waiter job openings in Dubai
Output:
{{
"intent":"SEARCH",
"type":"job",
"keywords":"waiter",
"location":"dubai"
}}

Query: how to delete my account
Output:
{{
"intent":"FAQ",
"type":"faq",
"keywords":"account deletion",
"location":""
}}

Return JSON only.
"""

    text = generate(prompt, 140)

    def extract_json(text):
        try:
            import re
            import json
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return None
        except Exception:
            return None

    data = extract_json(text) if text else None

    if data and isinstance(data, dict):
        intent = data.get("intent", "SEARCH")
        dtype = data.get("type", "article")
        keywords = data.get("keywords", query.lower())

        if isinstance(keywords, list):
            keywords = " ".join(map(str, keywords))

        if intent == "FAQ":
            dtype = "faq"

        return {
            "intent": intent,
            "type": dtype,
            "keywords": str(keywords).lower(),
            "location": data.get("location", "")
        }

    # Final Fallback
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

TASK:

1. Write a short introduction about the search results.
   - Max 4 lines
   - Use clean HTML formatting:
     - <p> for paragraph
     - <b> for important keywords
     - <u> optional emphasis
   - Max 3 highlights


2. Generate 5 follow-up search suggestions.
   - MAX 6 words each
   - Plain text only
   - No punctuation at end
   - No long phrases

IMPORTANT:
    - Ensure valid JSON is completed
    - Do NOT cut output mid-sentence
    - Keep response within token limit

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "intro": "<p>text</p>",
  "suggestions": [
    "suggestion 1",
    "suggestion 2",
    "suggestion 3",
    "suggestion 4",
    "suggestion 5"
  ]
}}
"""

    text = generate(prompt, 300)

    print("SUMMARY RAW:", text)

    data = safe_json(text)

    if not data:
        return "", []

    intro = data.get("intro", "").strip()
    suggestions = data.get("suggestions", [])

    if not isinstance(suggestions, list):
        suggestions = []

    suggestions = [str(s).strip() for s in suggestions if s]

    return intro, suggestions


# @app.post("/generate_keywords")
# def generate_keywords(req: KeywordGenRequest):

#     print(f"DEBUG: LLM ko bheja jane wala Data:")
#     print(f"Title: {req.title}")
#     print(f"Content: {req.content}")

#     prompt = f"""Generate 21 relevant SEO keywords for the hospitality industry based on the following. 
#         Return ONLY a comma-separated list. Do not use numbers, bullet points, or any prefixes.
#         Focus on the Title for core role and Content for location/details.
        
#         Title: {req.title}
#         Content: {req.content}"""

#     keywords_text = generate(prompt, tokens=150)
    
#     return {"keywords": [k.strip() for k in keywords_text.split(",") if k.strip()]}

@app.post("/generate_keywords")
def generate_keywords(req: KeywordGenRequest):

    print(f"DEBUG: Generating for Type: {req.title} | Data: {req.content}")

    prompt = f"""
    You are an AI SEO Specialist. Your task is to generate exactly 21 highly relevant, professional keywords.
    
    CONTEXT:
    - Entity Type: {req.title} (This defines the domain, e.g., Job, Product, Supplier, Article)
    - Input Data: {req.content} (This contains the specific details)

    STRICT GUIDELINES:
    1. STRICT RELEVANCE: Keywords must ONLY relate to the specific 'Entity Type' and 'Input Data'. 
       - If it's a 'Furniture Product', DO NOT include 'kitchen', 'food', or 'hotel management'.
       - If it's a 'Job', focus on skills, role, and industry.
       - If it's an 'Article', focus on the topic and category.
    2. NO GENERIC FILLERS: Avoid words like 'hospitality', 'service', or 'global' unless they are explicitly in the Input Data.
    3. NO HALLUCINATIONS: Do not assume related categories. Stay within the boundary of the provided content.
    4. FORMAT: Return ONLY a comma-separated list of strings. No bullets, no numbering, no introductory text.

    Data for Keyword Generation:
    {req.content}

    Keywords:"""

    # LLM Call
    keywords_text = generate(prompt, tokens=150)
    
    if not keywords_text:
        return {"keywords": []}

    raw_keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
    
    return {"keywords": raw_keywords[:21]}

@app.post("/summary")
def summary(req: SummaryRequest):
    intro, suggestions = generate_summary(req.query, req.titles)

    return {
        "intro": intro,
        "suggestions": suggestions
    }


@app.get("/")
def health():
    return {"status": "ok"}