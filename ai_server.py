# previous code 

import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

LLM_URL = "http://127.0.0.1:8080/completion"

SITE_CONTEXT = """
Hozpitality.com is a global hospitality platform.

It includes:
- Jobs (hospitality careers, hotel jobs, chef jobs)
- Articles (industry news, insights, trends)
- Professionals (user profiles, candidates, experts)
- Companies (hospitality brands, hotels, suppliers)
- Events (upcoming hospitality events, expos)
- Suppliers (vendors, services, B2B providers)
- Products (equipment, services)
- Awards (voting-based hospitality awards)

FIELD MEANINGS:

- title → main heading
- content → full searchable text (includes dates, roles, departments)
- company → associated company or brand
- location → city/country
- model_type → content type (job, article, professional, etc.)

IMPORTANT RULES:

- "professional" = person profile, not article
- "article" = blog/news content
- "event" = has start/end date (future = upcoming)
- "job" = hiring role
- "awards" = voting or competition

SEARCH BEHAVIOR:

- If user asks "today" → match recent or current content
- If "latest" → prefer newest
- If "upcoming" → prefer future events
- If "active" → prefer currently active items

NEVER GUESS:
Only use provided Data. If no match, return empty results.
"""

json_grammar = r"""
root ::= object

object ::= "{" ws "\"intro\"" ws ":" ws string ws "," ws "\"results\"" ws ":" ws array ws "," ws "\"followup\"" ws ":" ws string ws "}"

array ::= "[" ws (item (ws "," ws item)*)? ws "]"

item ::= "{" ws
    "\"title\"" ws ":" ws string ws "," ws
    "\"url\"" ws ":" ws string ws "," ws
    "\"company\"" ws ":" ws string ws "," ws
    "\"location\"" ws ":" ws string ws "," ws
    "\"model_type\"" ws ":" ws string
ws "}"

string ::= "\"" ([^"\\'] | "\\" .)* "\""

ws ::= [ \t\n\r]*
"""


class IntentRequest(BaseModel):
    query: str

class SynonymRequest(BaseModel):
    text: str

class SummaryRequest(BaseModel):
    query: str
    context: list = []
    type: str = ""


class KeywordGenRequest(BaseModel):
    title: str
    content: str

class ChatRequest(BaseModel):
    query: str
    context: list = []
    history: list = []
    google_links: list = []
    type: str = ""

def generate(prompt, tokens=300):

    payload = {
        "prompt": f"[INST] {prompt} [/INST]",
        "n_predict": tokens,
        "temperature": 0.0,
        "top_p": 0.9,
        "grammar": json_grammar
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
    try:
        return json.loads(text)
    except Exception as e:
        print("JSON FAIL:", e)
        print(text)

        text = text.replace("'", "")
        try:
            return json.loads(text)
        except:
            return None
    

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

Query: How to apply for a job on Hozpitality?
Output:
{{
"intent":"FAQ",
"type":"faq",
"keywords":"faq",
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


Query: Jivesh Kumar professional dubai united arab emirates
Output:
{{
"intent":"SEARCH",
"type":"professional",
"keywords":"jivesh kumar",
"location":"dubai"
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


def generate_summary(query, context, intent_type):

    prompt = f"""
You are the AI assistant for Hozpitality.com.

User Query:
{query}

Intent Type:
{intent_type}

Search Results Context:
{context}

TASK:

1. Write a short intro (max 4 lines)
   - Use HTML (<p>, <b>, <u>)
   - Highlight key relevant terms

2. Generate 5 follow-up suggestions.

STRICT RULES:
- MUST be based ONLY on provided results
- MUST be short (max 10 words each)
- MUST match the intent type ({intent_type})
- MUST be directly useful to the user
- NO generic suggestions
- NO punctuation at end
- Each suggestion must be unique

TYPE GUIDELINES:
- If type = faq → generate question-style suggestions
- If type = job → generate job search related suggestions
- If type = company → generate company-related queries
- If type = article → generate topic exploration queries

IMPORTANT:
- Ensure valid JSON output
- Do NOT cut response
- Always return randomly 1-3 suggestions not more than 3

OUTPUT FORMAT:

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

    text = generate(prompt, 320)

    print("SUMMARY RAW:", text)

    data = safe_json(text)

    if not data:
        return "", []

    intro = data.get("intro", "").strip()
    suggestions = data.get("suggestions", [])

    if not isinstance(suggestions, list):
        suggestions = []

    suggestions = [
        str(s).strip().lower()[:60]
        for s in suggestions if s
    ][:5]

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
    intro, suggestions = generate_summary(
        req.query,
        req.context,
        req.type
    )

    return {
        "intro": intro,
        "suggestions": suggestions
    }

@app.post("/chat")
def chat(req: ChatRequest):

    prompt = f"""
You are a hospitality AI assistant for Hozpitality.com.

{SITE_CONTEXT}

Query: {req.query}
History: {req.history}
Data: {req.context}


STRICT RULES:

1. If query mentions:
   - "today" → prefer today's content
   - "latest" → prefer newest content
   - "upcoming" → prefer future events
   - "active" → prefer active items

2. RETURN JSON ONLY
3. DO NOT RETURN HTML
4. DO NOT INVENT DATA
5. USE ONLY VALUES FROM "Data"
6. SELECT BEST MATCHING RESULTS
7. DO NOT use apostrophes (') in output
8. Use only double quotes (") for JSON
9. Keep followup simple (max 8 words, no punctuation)

EACH RESULT MUST HAVE:
- title (exact match from Data)
- url (exact)
- company (exact)
- location (exact)
- model_type (exact)

MAX 5 RESULTS

OUTPUT:

{{
  "intro": "short helpful intro",
  "results": [
    {{
      "title": "exact title from data",
      "url": "exact url",
      "company": "exact company",
      "location": "exact location",
      "model_type": "job"
    }}
  ],
  "followup": "short next suggestion"
}}
"""

    text = generate(prompt, 400)
    data = safe_json(text)

    if not data:
        return {
            "intro": "No results found",
            "results": [],
            "followup": "Try another search"
        }

    return data



@app.post("/synonyms")
def get_synonyms(req: SynonymRequest):

    prompt = f"""
You are an AI that generates job-related synonyms.

INPUT:
"{req.text}"

TASK:
- Extract core keywords
- Generate related job role keywords and synonyms
- Keep it relevant to hiring / job titles
- Return ONLY a JSON list

RULES:
- No explanation
- No sentence
- Only list
- Max 15 keywords
- Lowercase only

EXAMPLE:

Input: finance
Output:
["finance", "accounting", "audit", "billing", "payroll", "accounts payable", "accounts receivable"]

Input: IT
Output:
["it", "software", "developer", "engineer", "backend", "frontend", "technology"]

OUTPUT:
"""

    text = generate(prompt, 120)

    data = safe_json(text)

    if isinstance(data, list):
        return {"keywords": data}

    return {"keywords": [req.text.lower()]}



@app.get("/")
def health():
    return {"status": "ok"}