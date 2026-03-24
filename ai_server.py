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
    q = query.strip()

    # 🔹 SOFT GUARDRAILS (precision boost, not hardcoding)
    q_lower = q.lower()

    job_signals = ["job", "jobs", "vacancy", "vacancies", "hiring", "career", "apply"]
    award_signals = ["award", "awards", "nomination", "winner", "ranking"]

    if any(w in q_lower for w in job_signals):
        return "job", q

    if any(w in q_lower for w in award_signals):
        return "awards", q

    # 🔹 FEW-SHOT PROMPT (THIS IS THE REAL FIX)
    prompt = f"""
You are an intent classifier for a hospitality platform (jobs, companies, news, events).

Classify the query into ONE of:
job, article, professional, faq, company, event, supplier, product, awards

Examples:

"chef jobs" → job
"hotel vacancies dubai" → job
"apply for waiter job" → job
"latest hotel news" → article
"dubai hospitality trends" → article
"how to apply for job" → faq
"what is hospitality industry" → faq
"marriott hotel company" → company
"top chefs in dubai" → professional
"upcoming hospitality events dubai" → event
"hotel awards 2025" → awards
"kitchen equipment supplier" → supplier
"buy hotel furniture" → product

Now classify:

Query: {q}

Return ONLY valid JSON:
{{
  "intent": "...",
  "query": "clean improved query"
}}
"""

    try:
        r = requests.post(LLM_URL, json={
            "prompt": f"[INST] {prompt} [/INST]",
            "n_predict": 80,
            "temperature": 0.0
        }, timeout=5)

        text = r.json().get("content", "")
        data = safe_json_parse(text)

        if data:
            intent = data.get("intent", "").strip().lower()
            new_query = data.get("query", q)

            if intent not in TYPE_MAPPING:
                intent = "article"

            if any(w in q_lower for w in job_signals):
                intent = "job"

            return intent, new_query

    except Exception as e:
        print("Analyze error:", e)

    return "article", q

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

def normalize_type(category_text):
    if not category_text:
        return "article"

    c = category_text.lower()

    if "event" in c:
        return "event"
    if "award" in c:
        return "awards"
    if "job" in c:
        return "job"
    if "supplier" in c:
        return "supplier"
    if "professional" in c:
        return "professional"
    if "company" in c:
        return "company"
    if "faq" in c or "question" in c or "how" in c:
        return "faq"
    if "product" in c:
        return "product"

    return "article"

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
            "category": normalize_type(r[2]),
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

def fetch_db_context(intent):
    query = f"""
        SELECT DISTINCT ON (title) title, category_text, ai_keywords, content
        FROM master_search_mastersearchindex
        WHERE is_live = TRUE
        AND LOWER(category_text) LIKE '%{intent}%'
        ORDER BY title, id DESC
        LIMIT 5
    """

    rows = safe_db_execute(query)

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
    if intent:
        sem = [r for r in sem if intent in (r.get("category") or "").lower()]
    db = fetch_db_context(intent)

    print(f"[HYBRID] Semantic results: {len(sem)}")
    print(f"[HYBRID] DB fallback results: {len(db)}")

    combined = {}

    for r in sem:
        score = 0.7

        r_type = (r.get("category") or "").strip().lower()
        text = (r.get("content") or "").lower()

        if intent and (
            intent in r_type or
            r_type.startswith(intent)
        ):
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
            if intent in (x["data"].get("category") or "").lower()
        ]

        final = filtered[:5] if filtered else []
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

# def generate_full_ai(query, context, history):

#     prompt = f"""
# You are Hozpitality AI Assistant.

# You behave like ChatGPT but specialized for hospitality (jobs, companies, news, events).

# ====================
# USER QUERY:
# {query}

# CONTEXT (JSON):
# {context}

# HISTORY:
# {history}

# 🎯 YOUR TASK:

# 1. Understand user intent deeply
# 2. Decide response style:
#    - If informational → explain clearly
#    - If job search → show relevant jobs
#    - If mixed → explain + suggest
# 3. Use context ONLY if relevant
# 4. If relevant items exist in context:
# - ALWAYS show them (max 3–5)
# - Even if data is partial

# If no items exist → then say no results

# 🧠 RESPONSE RULES:

# - You are NOT a search engine
# - You are an assistant

# ✔ You MAY:
# - explain
# - summarize
# - suggest
# - list items (if needed)

# ❌ DO NOT:
# - force list always
# - show irrelevant items
# - dump raw data

# 🌐 HTML RULES:

# - Use clean HTML only
# - Use:
#   <p> for explanation
#   <ul><li> ONLY if listing helps
#   <strong> for highlights

# - Links format:

# job:
# https://www.hozpitality.com/jobs/details/{{slug}}/

# article:
# https://www.hozpitality.com/articles/details/{{slug}}/

# company/professional/supplier:
# https://www.hozpitality.com/profile/{{slug}}/

# event:
# https://www.hozpitality.com/events/details/{{slug}}/

# awards:
# https://www.hozpitality.com/awards

# - If slug exists → use clickable link  
# - If slug missing → show plain text title
# - NEVER hallucinate links

# 🎯 OUTPUT FORMAT (STRICT JSON):

# {{
#   "intro_html": "<p>natural human-like intro</p>",
#   "results_html": "<div>clean structured html</div>",
#   "followup": "relevant follow-up question"
# }}

# 💡 EXAMPLES:

# User: chef jobs

# → GOOD:
# <p>Here are some chef job opportunities you can explore:</p>
# <ul>
#   <li><a href="...">Chef - Dubai</a></li>
# </ul>

# User: what is hospitality industry

# → GOOD:
# <p>The hospitality industry includes hotels, restaurants...</p>

# """

#     try:
#         r = requests.post(LLM_URL, json={
#             "prompt": f"[INST] {prompt} [/INST]",
#             "n_predict": 500,
#             "temperature": 0.4   # 🔥 slightly creative (important)
#         }, timeout=20)

#         text = r.json().get("content", "")
#         data = safe_json_parse(text)

#         if data:
#             return data

#     except Exception as e:
#         print("❌ LLM ERROR:", e)

#     return {
#         "intro_html": f"<p>Here’s what I found for '{query}':</p>",
#         "results_html": "<div><p>No strong results available right now.</p></div>",
#         "followup": "Would you like me to refine your search?"
#     }

def generate_full_ai(query, context, history):

    prompt = f"""
You are Hozpitality AI Assistant.

You MUST act as a job search assistant.

USER QUERY:
{query}

CONTEXT (JSON ARRAY):
{context}


CRITICAL RULE (OVERRIDE ALL):

If ANY item in context has "type": "job":
→ You MUST show those jobs
→ NEVER say "no results"
→ NEVER ignore context


HOW TO USE CONTEXT:

Each item contains:
- title → job title
- description / full_content → job details
- slug → job link (if available)


RESPONSE RULES:

- ALWAYS show jobs if present
- Show 3–5 items
- If slug exists → clickable link
- If no slug → plain text

- DO NOT reject results
- DO NOT say "no results" if jobs exist


HTML FORMAT:

<p>Intro sentence</p>

<ul>
  <li><a href="JOB_LINK">Job Title</a></li>
</ul>


OUTPUT STRICT JSON:

{{
  "intro_html": "...",
  "results_html": "...",
  "followup": "..."
}}
"""

    try:
        r = requests.post(LLM_URL, json={
            "prompt": f"[INST] {prompt} [/INST]",
            "n_predict": 250,   
            "temperature": 0.3
        }, timeout=20)

        text = r.json().get("content", "")
        data = safe_json_parse(text)

        if data:
            return data

    except Exception as e:
        print("❌ LLM ERROR:", e)

    return {
        "intro_html": f"<p>Here are some job opportunities for '{query}':</p>",
        "results_html": "<div><p>Please try again.</p></div>",
        "followup": "Would you like jobs in a specific location?"
    }

def extract_slug(content):
    if not content:
        return ""

    match = re.search(r"[Ss]lug:\s*(\S+)", content)
    return match.group(1) if match else ""


def build_context(context):
    if not context:
        return "[]"

    structured = []

    for c in context:
        content = c.get("content") or ""

        structured.append({
            "type": (c.get("category") or "").lower(),
            "title": c.get("title"),
            "keywords": c.get("keywords"),
            "description": content[:400],
            "full_content": content,
            "slug": extract_slug(content)
        })

    return json.dumps(structured, indent=2)

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

    print("\n====== FINAL CONTEXT TO LLM ======")
    print(context_text)

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