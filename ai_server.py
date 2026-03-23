import requests
import psycopg2
import numpy as np
import faiss
import json
import re
import os

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool


FAISS_INDEX_FILE = "faiss.index"
VECTOR_DATA_FILE = "vector_data.json"
ID_MAP_FILE = "id_map.json"

id_map = {} 

TYPE_MAPPING = {
    'job': 'job',
    'article': 'article',
    'professional': 'professional',
    'faq': 'faq',
    'company': 'company',
    'event': 'event',
    'supplier': 'supplier',
    'product': 'product',
    'awards': 'awards'
}

ACTION_MAP = {
    "job": "apply_job",
    "faq": "apply_job",
    "article": "read_article",
    "professional": "view_profile",
    "company": "view_company",
    "event": "view_event",
    "supplier": "view_supplier",
    "product": "view_product",
    "awards": "view_award"
}


class KeywordGenRequest(BaseModel):
    title: str
    content: str

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

LLM_URL = os.getenv("LLM_URL")

app = FastAPI()

db_pool = SimpleConnectionPool(1, 10, **DB_CONFIG)

def detect_primary_type(context):
    if not context:
        return None

    scores = {}

    for c in context:
        t = (c.get("category") or "").lower()

        if "question" in (c.get("content") or "").lower():
            t = "faq"

        scores[t] = scores.get(t, 0) + 1

    return max(scores, key=scores.get) if scores else None

def save_faiss():
    if vector_index is not None:
        faiss.write_index(vector_index, FAISS_INDEX_FILE)

        with open(VECTOR_DATA_FILE, "w") as f:
            json.dump(vector_data, f)

        with open(ID_MAP_FILE, "w") as f:
            json.dump(id_map, f)

        print("💾 FAISS SAVED")
    
def load_faiss():
    global vector_index, vector_data, id_map

    if os.path.exists(FAISS_INDEX_FILE):
        print("⚡ Loading FAISS from disk...")

        vector_index = faiss.read_index(FAISS_INDEX_FILE)

        with open(VECTOR_DATA_FILE) as f:
            vector_data.extend(json.load(f))

        with open(ID_MAP_FILE) as f:
            id_map.update(json.load(f))

        print("✅ FAISS LOADED:", len(vector_data))
        return True

    return False

def analyze_query(query):
    prompt = f"""
You are a STRICT search query intent classifier.

User Query:
{query}

Return ONLY valid JSON:

{{
  "intent": "job/article/professional/faq/company/event/supplier/product/awards",
  "query": "clean improved search query"
}}

INTENT DEFINITIONS:

- job → job search, hiring, vacancies
- article → general reading, news, blogs, information
- faq → how-to, help, instructions, questions
- company → company, brand, employer
- event → conferences, summits, expos, scheduled events
- awards → awards, nominations, winners, contests, rankings
- product → items, marketplace, buy/sell
- supplier → vendors, suppliers
- professional → people, profiles

STRICT RULES:

1. If query contains:
   - "award", "awards", "nomination", "winner", "ranking"
   → intent MUST be "awards"

2. If query contains:
   - "date", "deadline", "upcoming", "schedule"
   AND also contains "award"
   → intent MUST be "awards"

3. If query is about event timing WITHOUT awards
   → intent = "event"

4. NEVER classify award-related queries as "article"

5. query must be corrected and meaningful (e.g., "upcoming awards dates 2026")

6. NO explanation, ONLY JSON

Output:
"""

    try:
        r = requests.post(LLM_URL, json={
            "prompt": f"[INST] {prompt} [/INST]",
            "n_predict": 100,
            "temperature": 0.0
        }, timeout=5)

        text = r.json().get("content", "")

        data = safe_json_parse(text)

        if data:
            intent = data.get("intent", "article")
            new_query = data.get("query", query)

            if intent not in TYPE_MAPPING:
                intent = "article"

            return intent, new_query

    except Exception as e:
        print("Analyze error:", e)

    return "article", query

def generate(prompt, tokens=500):

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


def get_db():
    return db_pool.getconn()

def release_db(conn):
    db_pool.putconn(conn)

def safe_db_execute(query):
    print("\n[DB] Executing Query:\n", query)

    conn = None
    cur = None
    try:
        conn = get_db()
        print("[DB] Connection acquired")

        cur = conn.cursor()
        cur.execute(query)

        rows = cur.fetchall()
        print(f"[DB] Rows fetched: {len(rows)}")

        return rows

    except Exception as e:
        print("❌ DB ERROR:", e)
        return []

    finally:
        if cur:
            cur.close()
        if conn:
            release_db(conn)
            print("[DB] Connection released")

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text).astype("float32")

vector_index = None
vector_data = []

def build_vector_index():
    global vector_index, vector_data

    print("\n[FAISS] Building vector index...")

    rows = safe_db_execute("""
        SELECT id, title, category_text, ai_keywords , content
        FROM master_search_mastersearchindex
        WHERE is_live = TRUE
    """)

    print(f"[FAISS] Rows received: {len(rows)}")

    embeddings = []
    vector_data = []
    id_map.clear()

    for i, r in enumerate(rows):
        text = f"{r[1]} {r[2]} {r[3]} {r[4]}"
        print(f"[FAISS] Processing row {i}: {r[1]}")

        emb = get_embedding(text)
        embeddings.append(emb)

        vector_data.append({
            "id": r[0],
            "title": r[1],
            "category": r[2],
            "keywords": r[3],
            "content": r[4]
        })

        id_map[r[0]] = i

    if not embeddings:
        print("⚠️ No embeddings found — check DB data or query")
        return

    vectors = np.array(embeddings)
    print("[FAISS] Vector shape:", vectors.shape)

    vector_index = faiss.IndexFlatL2(vectors.shape[1])
    vector_index.add(vectors)

    print("✅ FAISS READY:", len(vector_data))

    save_faiss()

def semantic_search(query, k=5):
    print(f"\n[SEARCH] Semantic search for: {query}")

    if vector_index is None:
        build_vector_index()

    q_vec = np.array([get_embedding(query)])
    print("[SEARCH] Query embedding shape:", q_vec.shape)

    distances, indices = vector_index.search(q_vec, k)

    print("[SEARCH] Indices:", indices)
    print("[SEARCH] Distances:", distances)

    return [vector_data[i] for i in indices[0] if i < len(vector_data)]

def fetch_db_context():
    rows = safe_db_execute("""
        SELECT DISTINCT ON (title) title, category_text, ai_keywords, content
        FROM master_search_mastersearchindex
        WHERE is_live = TRUE
        ORDER BY title, id DESC
        LIMIT 5
    """)

    return [{
        "title": r[0],
        "category": r[1],
        "keywords": r[2],
        "content": r[3]
    } for r in rows]

def hybrid_search(query):
    print("\n[HYBRID] Running hybrid search")

    intent, improved_query = analyze_query(query)

    print("[HYBRID] Intent:", intent)
    print("[HYBRID] Improved Query:", improved_query)

    sem = semantic_search(improved_query)
    db = fetch_db_context()

    print(f"[HYBRID] Semantic results: {len(sem)}")
    print(f"[HYBRID] DB fallback results: {len(db)}")

    combined = {}

    for r in sem:
        score = 0.7

        r_type = (r.get("category") or "").lower()
        text = (r.get("content") or "").lower()

        if intent and intent in r_type:
            score += 2.0
        elif intent and r_type.startswith(intent):
            score += 2.0

        if "question:" in text or "how" in r.get("title","").lower():
            score += 0.5

        combined[r["title"]] = {"data": r, "score": score}

    for r in db:
        score = 0.3

        text = (r.get("content") or "").lower()

        if "featured" in text:
            score += 0.2
        if "premium" in text:
            score += 0.2

        if r["title"] in combined:
            combined[r["title"]]["score"] += score
        else:
            combined[r["title"]] = {"data": r, "score": score}

    final = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

    if intent:
        filtered = [
            x for x in final
            if intent in (x["data"].get("category","").lower())
        ]

        if filtered:
            final = filtered[:5]
        else:
            final = final[:5]
    else:
        final = final[:5]

    print(f"[HYBRID] Final results: {len(final)}")

    return [x["data"] for x in final]

def safe_json_parse(text):
    if not text:
        return None

    # 🔥 FIX ESCAPED CHARACTERS
    text = text.replace("\\_", "_")

    try:
        return json.loads(text)
    except:
        pass

    try:
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            cleaned = text[start:end+1]
            cleaned = cleaned.replace("\\_", "_")
            return json.loads(cleaned)
    except:
        pass

    try:
        fixed = text.strip()

        if fixed.count("{") > fixed.count("}"):
            fixed += "}"

        if fixed.count("[") > fixed.count("]"):
            fixed += "]"

        fixed = fixed.replace("\\_", "_")

        return json.loads(fixed)
    except:
        pass

    return None

def generate_full_ai(query, context, history):

    prompt = f"""
You are Hozpitality AI Assistant.

INPUT
User Query:
{query}

Context:
{context}

History:
{history}

OBJECTIVE

Your job is to:
1. Understand user intent
2. Filter and select ONLY relevant context
3. Generate structured HTML response

DECISION LOGIC

STEP 1: Detect intent

STEP 2: Check if FAQ exists
- If any context item:
  - looks like a question/answer OR
  - contains instructions OR
  - starts with "how", "what", "can I"

THEN:
- Treat it as FAQ
- Extract its CONTENT
- Convert into step-by-step guide
- IGNORE all other items

STEP 3: If NO FAQ
- Select top 3 relevant items
- Show them as list/cards

FAQ OUTPUT RULE

- Convert answer into 4–6 steps
- Each step must be short and actionable
- Use STRICT HTML:

CRITICAL LINKING RULES:

- Every result MUST be a clickable <a> tag using slug
- Extract "Slug: xyz" from context

URL FORMAT RULES:

- job → https://www.hozpitality.com/jobs/details/{slug}/
- company → https://www.hozpitality.com/profile/{slug}/
- professional → https://www.hozpitality.com/profile/{slug}/
- supplier → https://www.hozpitality.com/profile/{slug}/
- article → https://www.hozpitality.com/articles/details/{slug}/
- event → https://www.hozpitality.com/events/details/{slug}/
- awards → https://www.hozpitality.com/awards

IMPORTANT:

- ALWAYS wrap title in <a href="URL">TITLE</a>
- If slug is missing → DO NOT create link
- NEVER hallucinate slug
- Use ONLY slug from context

<div>
  <ol>
    <li>Step 1</li>
    <li>Step 2</li>
  </ol>
</div>

NON-FAQ OUTPUT RULE

- Use max 3 items
- Use <ul><li> format
- Each <li> MUST contain clickable link:

<li>
  <a href="URL">Title</a>
</li>
- Keep titles short

OUTPUT FORMAT (STRICT JSON)

Return ONLY valid JSON:

{{
  "intro_html": "<p>Short engaging intro</p>",
  "results_html": "<div>HTML content</div>",
  "followup": "One relevant follow-up question",
}}



CRITICAL RULES

- ONLY use "Hozpitality" (no other platforms)
- DO NOT hallucinate
- Use ONLY given context
- DO NOT invent data
- ALWAYS return valid JSON
- ALWAYS close all HTML tags
- Keep response concise
- Do NOT decide action
- Action is handled by system

EXAMPLE

Context:
TYPE: faq
TITLE: How do I apply for a job?
CONTENT: Create profile. Search jobs. Click apply. Submit application.

Output:
{{
  "intro_html": "<p>Here’s how you can apply for a job on Hozpitality:</p>",
  "results_html": "<div><ol><li>Create your profile</li><li>Search for jobs</li><li>Select a job</li><li>Click Apply</li><li>Submit your application</li></ol></div>",
  "followup": "Would you like to see available jobs now?",
  "action": "apply_job"
}}
"""

    print("\n[LLM] Sending request")
    print("[LLM] Query:", query)
    print("[LLM] Context:", context)
    print("[LLM] History:", history)

    try:
        r = requests.post(LLM_URL, json={
            "prompt": f"[INST] {prompt} [/INST]",
            "n_predict": 500,
            "temperature": 0.3
        }, timeout=15)

        print("[LLM] Status Code:", r.status_code)

        text = r.json().get("content", "")
        print("[LLM] Raw Response:", text)

        data = safe_json_parse(text)

        if data:
            print("[LLM] Parsed JSON:", data)
            return data
        else:
            print("⚠️ JSON parsing failed")

    except Exception as e:
        print("❌ LLM ERROR:", e)

    return {
        "answer": f"Fallback response for: {query}",
        "action": None
    }

def extract_slug(content):
    if not content:
        return ""
    match = re.search(r"Slug:\s*(\S+)", content)
    return match.group(1) if match else ""


def build_context(context):
    if not context:
        return "No strong results."

    formatted = []

    for i, c in enumerate(context):
        formatted.append(f"""
ITEM {i+1}:
TYPE: {c.get("category")}
TITLE: {c.get("title")}
KEYWORDS: {c.get("keywords")}
SLUG: {extract_slug(c.get("content"))}
CONTENT: {c.get("content")}
""")

    return "\n\n".join(formatted)

def build_history(history):
    return "\n".join([f"{h['role']}: {h['content']}" for h in history[-5:]]) if history else ""

class ChatRequest(BaseModel):
    query: str
    context: list = []
    history: list = []

@app.post("/chat")
def chat(req: ChatRequest):

    print("\n========== NEW REQUEST ==========")
    print("[API] Incoming query:", req.query)

    query = req.query.strip()

    context = req.context if req.context else hybrid_search(query)

    print("[API] Context selected:", len(context))

    context_text = build_context(context)
    history_text = build_history(req.history)

    ai = generate_full_ai(query, context_text, history_text)

    print("[API] Final response:", ai)

    primary_type = detect_primary_type(context)
    final_action = ACTION_MAP.get(primary_type, "none")

    return {
        "intro_html": ai.get("intro_html", ""),
        "results_html": ai.get("results_html", ""),
        "followup": ai.get("followup", ""),
        "action": final_action,
        "context": context
    }


@app.on_event("startup")
def startup():
    if not load_faiss():
        print("🚀 First build")
        build_vector_index()
        save_faiss()


@app.post("/add-to-index")
def add_to_index(data: dict):
    global vector_index, vector_data, id_map

    if vector_index is None:
        build_vector_index()

    text = f"{data['title']} {data['category']} {data['keywords']} {data['content']}"
    emb = get_embedding(text)

    if data["id"] in id_map:
        idx = id_map[data["id"]]

        vector_data[idx] = {
            "id": data["id"],
            "title": data["title"],
            "category": data["category"],
            "keywords": data["keywords"],
            "content": data["content"]
        }

        print("♻️ Updated existing ID:", data["id"])

    else:
        vector_index.add(np.array([emb]))

        vector_data.append({
            "id": data["id"],
            "title": data["title"],
            "category": data["category"],
            "keywords": data["keywords"],
            "content": data["content"]
        })

        id_map[data["id"]] = len(vector_data) - 1

        print("✅ Added new ID:", data["id"])

        save_faiss()

    return {"status": "ok"}


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



@app.get("/reset-index")
def reset_index():
    import os

    for f in ["faiss.index", "vector_data.json", "id_map.json"]:
        if os.path.exists(f):
            os.remove(f)

    return {"status": "FAISS RESET"}


@app.get("/")
def health():
    return {"status": "FINAL OPTIMIZED AI READY"}