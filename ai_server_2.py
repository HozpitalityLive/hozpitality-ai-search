# ai_server_2.py

import re
import json
from cachetools import LRUCache
import requests
import os
import faiss
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from concurrent.futures import ThreadPoolExecutor, wait

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

import redis
import hashlib
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import CrossEncoder
import logging
import time
from psycopg2.pool import PoolError
from contextlib import contextmanager



LOG_FILE = os.path.join(os.getcwd(), "app.log")

logger = logging.getLogger("ai-websocket")
logger.setLevel(logging.DEBUG)  

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
CACHE_TTL = 600

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

MODEL_PATH = "/home/dev/models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

app = FastAPI()


embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = embedder.get_sentence_embedding_dimension()


db_pool = SimpleConnectionPool(5, 20, **DB_CONFIG)

memory_indexes = {}
memory_store = {}



@contextmanager
def get_db():
    conn = get_db_conn_with_retry()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)


def get_db_conn_with_retry(retries=3, delay=0.1):
    for i in range(retries):
        try:
            return db_pool.getconn()
        except PoolError:
            logger.warning(f"[DB POOL RETRY] attempt {i+1}")
            time.sleep(delay)

    raise Exception("DB connection pool exhausted after retries")



def rerank(query, results):
    if not results:
        return []

    pairs = [(query, r["title"]) for r in results]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    return [r[0] for r in ranked]

def cache_key(query):
    return "ai:" + hashlib.md5(query.encode()).hexdigest()

def clean_html_response(text: str):
    if not text:
        return ""

    return text.replace("```html", "").replace("```", "").strip()

def simple_rerank(results, intent_type=None):
    if not results:
        return []

    intent_type = (intent_type or "").lower()

    primary = []
    secondary = []

    for r in results:
        cat = (r.get("category") or "").lower()

        if intent_type and intent_type in cat:
            primary.append(r)
        else:
            secondary.append(r)

    return (primary + secondary)[:6]

def detect_mode(query, intent_type, context):
    q = query.lower().strip()

    if len(q) <= 3 or q in ["hi", "hello", "hey", "ok", "thanks"]:
        return "chat"

    if intent_type == "faq":
        return "faq"

    if intent_type in ["professional", "company", "supplier"]:
        return "profile"

    if intent_type == "job":
        return "list"

    if len(context) == 1:
        return "single"

    return "list"

def call_ollama(prompt, stream=False, model="hozpitality-llama", max_tokens=200):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens
        }
    }

    if not stream:
        res = requests.post(url, json=payload, timeout=60)
        return res.json().get("response", "")

    def generator():
        with requests.post(url, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    yield data.get("response", "")

    return generator()

def detect_followup_llm(query: str, last_ai_response: str = ""):
    prompt = f"""
You are a STRICT JSON classifier for conversational intent.

User Reply: "{query}"

Previous AI Response:
"{last_ai_response[:300]}"

TASK:
Classify if this is a follow-up.

RULES:

FOLLOW-UP = TRUE if:
- short replies: yes, ok, sure, continue
- vague replies: "tell me more", "details"
- refers to previous answer

FOLLOW-UP = FALSE if:
- new entity (new name, new job, new topic)
- contains location or job keywords
- full meaningful query

TYPE:
- expand → more detail
- refine → filter/search change
- switch → new topic
- chat → casual

OUTPUT JSON:
{{
  "is_followup": true/false,
  "type": "expand|refine|switch|chat"
}}
"""

    try:
        res = call_ollama(prompt, model="hozpitality-phi3", max_tokens=60)

        match = re.search(r'\{.*\}', res, re.DOTALL)
        if match:
            return json.loads(match.group())

    except Exception as e:
        logger.error(f"Followup detect error: {e}")

    return {"is_followup": False, "type": "switch"}

def store_last_ai_response(user_id, org_id, response):
    key = f"last_ai:{org_id}:{user_id}"
    redis_client.setex(key, 600, response)

def get_last_ai_response(user_id, org_id):
    key = f"last_ai:{org_id}:{user_id}"
    return redis_client.get(key) or ""

def store_last_context(user_id, org_id, intent_type, context):
    key = f"ctx:{org_id}:{user_id}"
    redis_client.setex(key, 600, json.dumps({
        "intent": intent_type,
        "context": context[:3]
    }))

def get_last_context(user_id, org_id):
    key = f"ctx:{org_id}:{user_id}"
    data = redis_client.get(key)
    return json.loads(data) if data else None

def normalize_query_llm(query: str):
    cache_k = cache_key("norm:" + query)
    cached = redis_client.get(cache_k)
    if cached:
        return json.loads(cached)

    prompt = f"""
You are a search query optimizer.

User Query: "{query}"

TASK:
1. Fix spelling mistakes
2. Normalize to hospitality terms
3. Remove unnecessary words
4. Keep intent clear

OUTPUT JSON:
{{
  "normalized": "clean optimized query"
}}

Return ONLY JSON.
"""

    try:
        res = call_ollama(prompt, model="hozpitality-phi3", max_tokens=60)

        match = re.search(r'\{[\s\S]*?\}', res)
        if match:
            data = json.loads(match.group())
            redis_client.setex(cache_k, 600, json.dumps(data))
            return data
    except Exception as e:
        print("Normalize error:", e)

    return {"normalized": query}

def generate_synonyms_llm(query: str):
    cache_k = cache_key("syn:" + query)
    cached = redis_client.get(cache_k)
    if cached:
        return json.loads(cached)

    prompt = f"""
You are a search expansion engine.

User Query: "{query}"

TASK:
Generate 3-5 relevant synonyms or related search terms 
specific to hospitality jobs / industry.

OUTPUT JSON:
{{
  "synonyms": ["term1", "term2", "term3"]
}}

Return ONLY JSON.
"""

    try:
        
        res = call_ollama(prompt, model="hozpitality-phi3", max_tokens=80)
        text = res
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            redis_client.setex(cache_k, 600, json.dumps(data))
            return data
    except Exception as e:
        print("Synonym error:", e)

    return {"synonyms": []}


def build_final_query(query: str):
    norm = normalize_query_llm(query)
    normalized = norm.get("normalized", query)

    syn = generate_synonyms_llm(normalized)
    synonyms = syn.get("synonyms", [])

    final_query = normalized + " " + " ".join(synonyms)

    return final_query.strip()



def detect_intent_llm(query: str):
    categories = ['job', 'article', 'professional', 'faq', 'company', 'event', 'supplier', 'product', 'awards']

    prompt = f"""
You are a STRICT JSON API for search intent classification for a hospitality platform.

User Query: "{query}"
Categories: {categories}

CRITICAL RULES (FOLLOW STRICTLY):

1. SPELLING FIX
- Fix spelling mistakes ONLY
- DO NOT change structure or meaning

2. INTENT DETECTION

A. PROFESSIONAL (VERY IMPORTANT PRIORITY)
- If query is about a PERSON
- If query starts with "who is"
- If query looks like a NAME (2 words like "raj bhatt")
- If asking about a person profile

Examples:
"who is raj bhatt"
"raj bhatt"
"vikas khanna chef"

→ intent = "SEARCH"
→ type = "professional"


B. FAQ
- Queries starting with:
  "how to", "how do", "steps", "process"


C. JOB
- ONLY if explicitly job intent:
  "jobs", "hiring", "vacancy", "apply", "opening"

🚫 IMPORTANT:
- DO NOT classify as job if person name is present
- Example:
  "who is raj bhatt" ❌ NOT job


D. COMPANY
- If searching for company info


E. DEFAULT
- Otherwise → SEARCH + best matching type

3. TYPE
Must be EXACTLY one of:
{categories}

4. LOCATION
- Extract ONLY city or country
- single word
- else empty string

5. KEYWORDS
- 2–4 important words
- lowercase
- REMOVE filler words
- KEEP names and roles
- SPACE separated (NO commas)

Examples:
"who is raj bhatt"
→ "raj bhatt"

"jobs for chef in dubai"
→ "chef dubai"

OUTPUT (STRICT JSON ONLY)
{{
"intent": "SEARCH",
"type": "professional",
"keywords": "raj bhatt",
"location": "",
"rephrased_query": "who is raj bhatt"
}}
"""

    try:
        answer = call_ollama(prompt, model="hozpitality-llama", max_tokens=200)

        match = re.search(r'\{.*\}', answer, re.DOTALL)
        if match:
            data = json.loads(match.group())

            keywords = data.get("keywords", "")
            keywords = keywords.replace(",", " ").lower()
            keywords = " ".join(keywords.split())

            location = (data.get("location") or "").lower().strip()

            rq = (data.get("rephrased_query") or query).lower()
            rq = re.sub(r'\s+', ' ', rq).strip()

            cleaned = {
                "intent": data.get("intent", "SEARCH"),
                "type": data.get("type", "article"),
                "keywords": keywords,
                "location": location,
                "rephrased_query": rq
            }

            q_lower = query.lower().strip()

            if (
                q_lower.startswith("who is") or
                re.match(r"^[a-z]+ [a-z]+$", q_lower)
            ):
                cleaned["type"] = "professional"

            logger.info(f"Intent detected: {cleaned}")
            return cleaned

        raise ValueError("Invalid JSON")

    except Exception as e:
        logger.error(f"Intent error: {e}")

        return {
            "intent": "SEARCH",
            "type": "article",
            "keywords": query.lower(),
            "location": "",
            "rephrased_query": query.lower()
        }

def tenant_key(user_id, org_id):
    return f"{org_id}:{user_id}"

def get_memory(user_id, org_id):
    key = tenant_key(user_id, org_id)
    if key not in memory_indexes:
        memory_indexes[key] = faiss.IndexFlatL2(EMBED_DIM)
        memory_store[key] = []
    return memory_indexes[key], memory_store[key]

def store_memory(user_id, org_id, text):
    idx, store = get_memory(user_id, org_id)
    vec = np.array([get_embedding(text)], dtype="float32")
    faiss.normalize_L2(vec)
    idx.add(vec)
    store.append(text)
    if len(store) > 50:
        store.pop(0)

def retrieve_memory(user_id, org_id, query):
    idx, store = get_memory(user_id, org_id)
    if not store:
        return []
    q_vec = np.array([get_embedding(query)], dtype="float32")
    faiss.normalize_L2(q_vec)
    D, I = idx.search(q_vec, 5)
    return [store[i] for i in I[0] if i < len(store)]

embedding_cache = LRUCache(maxsize=5000)

def get_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]

    vec = embedder.encode([text], normalize_embeddings=True)[0]

    embedding_cache[text] = vec
    return vec

def create_conversation(user_id, title):
    conn = None
    try:
        conn = get_db_conn_with_retry()
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO master_search_usersearchconversation (user_id, title, created_at, updated_at)
        VALUES (%s, %s, NOW(), NOW())
        RETURNING id
        """, (user_id, title))

        cid = cur.fetchone()[0]
        conn.commit()
        cur.close()

        return cid

    except Exception as e:
        logger.error(f"[DB ERROR create_conversation] {e}", exc_info=True)
        return None

    finally:
        if conn:
            db_pool.putconn(conn)

def save_message(conversation_id, role, content):
    conn = None
    try:
        conn = get_db_conn_with_retry()
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO master_search_usersearchmessage (conversation_id, role, content, created_at)
        VALUES (%s, %s, %s, NOW())
        """, (conversation_id, role, content))

        conn.commit()
        cur.close()

    except Exception as e:
        logger.error(f"[DB ERROR save_message] {e}", exc_info=True)

    finally:
        if conn:
            db_pool.putconn(conn)

def search_web(query):
    key = cache_key("web:" + query)
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)

    ALLOWED_DOMAINS = [
        "hozpitality.com",
        "hozpitalityexcellenceawards.com",
        "instagram.com/hozpitalitygroup",
        "www.instagram.com/hozpitalitygroup",

        "linkedin.com/company/hozpitalitygroup",
        "www.linkedin.com/company/hozpitalitygroup",
    ]

    domain_query = query + " " + " OR ".join([f"site:{d}" for d in ALLOWED_DOMAINS])

    try:
        res = requests.post(
            "https://google.serper.dev/search",
            json={"q": domain_query},
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            },
            timeout=2
        )

        data = res.json()

        results = [{
            "title": r.get("title"),
            "content": (r.get("snippet") or "")[:150],
            "category": "web",
            "location": r.get("link")
        } for r in data.get("organic", [])[:3]]

        results = [
            r for r in results
            if any(domain in (r.get("location") or "") for domain in ALLOWED_DOMAINS)
        ]

        redis_client.setex(key, 300, json.dumps(results))

        return results

    except Exception as e:
        print("Web error:", e)
        return []

def build_url(category, slug):
    if not slug:
        return ""

    category = (category or "").lower()

    if "job" in category:
        return f"https://www.hozpitality.com/jobs/details/{slug}/"
    elif "article" in category:
        return f"https://www.hozpitality.com/articles/details/{slug}/"
    elif "event" in category:
        return f"https://www.hozpitality.com/events/details/{slug}/"
    elif "company" in category or "professional" in category or "supplier" in category:
        return f"https://www.hozpitality.com/profile/{slug}/"
    elif "award" in category:
        return "https://www.hozpitality.com/awards"

    return ""

def search_db(query, intent_type=None, location=None):
    key = cache_key(f"db:{query}:{intent_type}:{location}")
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)

    conn = get_db_conn_with_retry()

    try:
        cur = conn.cursor()

        query = query.replace(",", " ").lower().strip()

        words = [
            w.strip()
            for w in query.split()
            if w and len(w) > 2 and w != location
        ]

        full_phrase = " ".join(words)

        def execute(sql, params):
            logger.info("SQL: %s", sql)
            logger.info("Params: %s", params)

            cur.execute(sql, params)
            rows = cur.fetchall()

            return [{
                "title": r[0],
                "content": (r[1] or "")[:150],
                "category": r[2],
                "location": r[3],
                "url": build_url(r[2], r[4])
            } for r in rows]

        def base_sql():
            sql = """
            SELECT title, content, category_text, location_text, slug
            FROM master_search_mastersearchindex
            WHERE is_live = TRUE
            """
            params = []

            if intent_type:
                sql += " AND LOWER(category_text) LIKE %s"
                params.append(f"%{intent_type.lower()}%")

            if location:
                sql += " AND LOWER(location_text) LIKE %s"
                params.append(f"%{location.lower()}%")

            return sql, params

        if full_phrase:
            sql, params = base_sql()

            sql += """
            AND (
                LOWER(title) LIKE %s
                OR LOWER(content) LIKE %s
            )
            ORDER BY 
                CASE 
                    WHEN LOWER(title) = %s THEN 0
                    WHEN LOWER(title) LIKE %s THEN 1
                    ELSE 2
                END
            LIMIT 6
            """

            params.extend([
                f"%{full_phrase}%",
                f"%{full_phrase}%",
                full_phrase,
                f"%{full_phrase}%"
            ])

            results = execute(sql, params)

            if results:
                return results

        if words:
            sql, params = base_sql()

            conditions = []
            for w in words:
                conditions.append("(LOWER(title) LIKE %s OR LOWER(content) LIKE %s)")
                params.extend([f"%{w}%", f"%{w}%"])

            sql += " AND " + " AND ".join(conditions)
            sql += " LIMIT 6"

            results = execute(sql, params)

            if results:
                return results

        if words:
            sql, params = base_sql()

            conditions = []
            for w in words:
                conditions.append("(LOWER(title) LIKE %s OR LOWER(content) LIKE %s)")
                params.extend([f"%{w}%", f"%{w}%"])

            sql += " AND (" + " OR ".join(conditions) + ")"
            sql += " LIMIT 6"

            results = execute(sql, params)

            return results

        return []

    except Exception as e:
        logger.error(f"DB ERROR: {e}", exc_info=True)
        return []

    finally:
        db_pool.putconn(conn)


def build_prompt(query, memory, context, intent_type=None, mode="list"):

    memory_text = "\n".join(memory[-2:]) if memory else ""

    context_text = ""
    for i, item in enumerate(context):
        context_text += f"""
[{i+1}]
Title: {item['title']}
Details: {item['content'][:120]}
Source: {item.get('url', item.get('location'))}
"""

    return f"""
You are a smart AI assistant for Hozpitality.com.

User Query: {query}
Intent Type: {intent_type}

Context Data:
{context_text}

Memory:
{memory_text}

BEHAVIOR MODE: {mode}

IMPORTANT BEHAVIOR RULES:

1. CHAT MODE:
- If mode = "chat"
- Respond like ChatGPT (friendly conversation)
- DO NOT show results
- Just normal human response

2. SINGLE MODE:
- If mode = "single"
- Show ONE detailed result
- Explain properly (like profile/details page)
- DO NOT list multiple items

3. PROFILE MODE:
- Show 1–3 profiles
- Use avatar style

4. LIST MODE:
- Show multiple results (jobs/articles)

5. FAQ MODE:
- Use bullet steps

STRICT:
- NEVER mix styles
- NEVER show results in chat mode

STRICT INSTRUCTIONS

1. OUTPUT MUST BE PURE HTML ONLY (NO MARKDOWN, NO ```)

2. RESPONSE STRUCTURE (MANDATORY):

<div class="ai-response">

  <!-- INTRO -->
  <div class="ai-intro">
    <p>Write 1-2 lines introduction based on the query.</p>
  </div>

  <!-- RESULTS -->
  <div class="ai-results">

CASE 1: PROFESSIONAL / COMPANY / SUPPLIER

IF intent_type is "professional" OR "company" OR "supplier":

- Show PROFILE STYLE (NOT list)
- Show only top 1–3 most relevant results

FORMAT:

<div class="profile-card" style="display:flex; gap:12px; align-items:flex-start; margin-bottom:12px;">
    
    <div class="avatar" style="width:40px; height:40px; border-radius:50%; background:#ddd; display:flex; align-items:center; justify-content:center; font-weight:bold;">
        {{first letter}}
    </div>

    <div class="info">
        <a href="URL" style="font-weight:600; text-decoration:none;">Name</a>
        <p style="margin:4px 0;">Description (use content)</p>
        <span style="font-size:12px; color:gray;">Location</span>
    </div>

</div>

RULES:
- DO NOT show job-style cards
- DO NOT show more than 3 results
- Focus on identity (who is this)
- Description MUST come from content
- Avatar = first letter of name

CASE 2: JOB

IF intent_type is "job":

FORMAT:

<div class="job-card" style="margin-bottom:10px;">
    <a href="URL" style="font-weight:600;">Job Title</a>
    <p style="margin:4px 0;">Short description</p>
    <span style="font-size:12px; color:gray;">Location</span>
</div>

RULES:
- Show minimum 5 results
- Keep it list-like (multiple items)

CASE 3: FAQ

FORMAT:

<ul>
  <li>Step or answer</li>
</ul>

CASE 4: DEFAULT

- Use normal result cards (same as job but fewer items)


  </div>

  <!-- FOLLOW UP -->
  <div class="ai-followup">
    <p><strong>Follow-up:</strong> Ask one relevant question based on intent.</p>
  </div>

</div>

DO NOT:
- Do NOT use markdown
- Do NOT return plain text
- Do NOT skip description
- Do NOT mix UI types

RETURN ONLY HTML.
"""

class ChatRequest(BaseModel):
    query: str
    user_id: int
    org_id: int
    conversation_id: int = None

@app.post("/chat")
def chat(req: ChatRequest):

    query = req.query
    user_id = req.user_id
    org_id = req.org_id

    title = query[:50] if len(query) < 50 else query[:47] + "..."
    conversation_id = req.conversation_id or create_conversation(user_id, title)

    cache_k = cache_key(f"{user_id}:{org_id}:{query}")
    cached = redis_client.get(cache_k)

    if cached:
        return {"conversation_id": conversation_id, "answer": cached}

    intent_data = detect_intent_llm(query)

    clean_query = intent_data.get("rephrased_query") or intent_data.get("keywords")
    base_query = (
        intent_data.get("rephrased_query")
        or intent_data.get("keywords")
        or query
    )

    final_query = build_final_query(base_query)

    intent_type = intent_data.get("type")

    q_lower = query.lower()

    if any(k in q_lower for k in ["job", "jobs", "hiring", "vacancy"]):
        intent_type = "job"

    elif any(k in q_lower for k in ["event", "events", "conference", "expo"]):
        intent_type = "event"

    elif any(q_lower.startswith(x) for x in [
        "how to", "how do", "how can", "steps to", "process"
    ]):
        intent_type = "faq"

    location = intent_data.get("location")

    if location:
        final_query += f" {location}"

    memory = retrieve_memory(user_id, org_id, final_query)

    with ThreadPoolExecutor() as executor:
        db_future = executor.submit(search_db, final_query, intent_type, location)
        web_future = executor.submit(search_web, final_query)

        done, _ = wait([db_future, web_future], timeout=2)

        db_context = db_future.result()

        web_context = web_future.result() if web_future in done else []

    if intent_type in ["job", "company", "professional"]:
        combined = db_context

    elif intent_type == "faq":
        if len(db_context) < 2:
            combined = db_context + web_context[:2]
        else:
            combined = db_context

    else:
        combined = db_context + web_context[:2]


    context = rerank(final_query, combined)

    if not context:
        answer = "You can check the official website or latest announcements for more details."
    else:
        try:
            mode = detect_mode(query, intent_type, context)
            prompt = build_prompt(query, memory, context, intent_type, mode)

            answer = call_ollama(prompt, model="hozpitality-llama", max_tokens=200)
        except Exception as e:
            print("LLM error:", e)
            answer = "Error generating response."

    save_message(conversation_id, "user", query)
    save_message(conversation_id, "assistant", answer)

    store_memory(user_id, org_id, query)
    store_memory(user_id, org_id, answer)

    redis_client.setex(cache_k, CACHE_TTL, answer)

    return {"conversation_id": conversation_id, "answer": answer}

@app.post("/chat-stream")
def chat_stream(req: ChatRequest):

    query = req.query
    user_id = req.user_id
    org_id = req.org_id


    intent_data = detect_intent_llm(query)

    base_query = (
        intent_data.get("rephrased_query")
        or intent_data.get("keywords")
        or query
    )

    final_query = build_final_query(base_query)
    intent_type = intent_data.get("type")

    q_lower = query.lower()

    if any(k in q_lower for k in ["job", "jobs", "hiring", "vacancy"]):
        intent_type = "job"

    elif any(k in q_lower for k in ["event", "events", "conference", "expo"]):
        intent_type = "event"

    elif any(q_lower.startswith(x) for x in [
        "how to", "how do", "how can", "steps to", "process"
    ]):
        intent_type = "faq"

    location = intent_data.get("location")

    if location:
        final_query += f" {location}"

    with ThreadPoolExecutor() as executor:
        db_future = executor.submit(search_db, final_query, intent_type, location)
        web_future = executor.submit(search_web, final_query)

        done, _ = wait([db_future, web_future], timeout=2)

        db_context = db_future.result()

        web_context = web_future.result() if web_future in done else []

    if intent_type in ["job", "company", "professional"]:
        combined = db_context

    elif intent_type == "faq":
        if len(db_context) < 2:
            combined = db_context + web_context[:2]
        else:
            combined = db_context

    else:
        combined = db_context + web_context[:2]


    context = rerank(final_query, combined)

    memory = retrieve_memory(user_id, org_id, query)
    
    mode = detect_mode(query, intent_type, context)

    prompt = build_prompt(query, memory, context, intent_type, mode)

    def generate():
        for token in call_ollama(prompt, stream=True, model="hozpitality-llama", max_tokens=200):
            yield token


    return StreamingResponse(generate(), media_type="text/plain")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):

    origin = websocket.headers.get("origin")
    logger.info(f"[WS CONNECT] Origin: {origin}")

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            logger.debug(f"[WS RECEIVED] Raw Data: {data}")

            query = data["query"]
            user_id = data["user_id"]
            org_id = data["org_id"]

            logger.info(f"[QUERY] User:{user_id} Org:{org_id} → {query}")

            conversation_id = data.get("conversation_id")

            if not conversation_id:
                conversation_id = create_conversation(user_id, query[:30])
            logger.debug(f"[CONVERSATION] ID: {conversation_id}")

            intent_data = detect_intent_llm(query)
            logger.debug(f"[INTENT] {intent_data}")

            base_query = intent_data.get("keywords") or query

            logger.debug(f"[BASE QUERY] {base_query}")

            final_query = base_query
            intent_type = intent_data.get("type")

            last_ai = get_last_ai_response(user_id, org_id)
            followup = detect_followup_llm(query, last_ai)

            logger.info(f"[FOLLOWUP DETECT] {followup}")

            if followup.get("is_followup"):
                last_ctx = get_last_context(user_id, org_id)

                if last_ctx:
                    intent_type = last_ctx.get("intent")
                    context = last_ctx.get("context")

                    logger.info("[FOLLOW-UP MODE ACTIVATED]")

                    ftype = followup.get("type")

                    if ftype == "expand":
                        mode = "single"
                    elif ftype == "refine":
                        mode = "list"
                    elif ftype == "chat":
                        mode = "chat"
                    else:
                        mode = "single"

                    prompt = build_prompt(
                        query,
                        retrieve_memory(user_id, org_id, query),
                        context,
                        intent_type,
                        mode
                    )

                    response = call_ollama(prompt, model="hozpitality-llama", max_tokens=800)

                    clean_html = clean_html_response(response)

                    await websocket.send_json({
                        "type": "final",
                        "data": {
                            "type": intent_type,
                            "html": clean_html
                        },
                        "conversation_id": conversation_id
                    })

                    store_last_ai_response(user_id, org_id, response)

                    continue

            q_lower = query.lower()

            if any(k in q_lower for k in ["job", "jobs", "hiring", "vacancy"]):
                intent_type = "job"

            elif any(k in q_lower for k in ["event", "events", "conference", "expo"]):
                intent_type = "event"

            elif any(q_lower.startswith(x) for x in [
                "how to", "how do", "how can", "steps to", "process"
            ]):
                intent_type = "faq"

            location = intent_data.get("location")

            if location:
                final_query += f" {location}"

            logger.info(f"[FINAL QUERY] {final_query}")
            logger.info(f"[INTENT TYPE] {intent_type} | Location: {location}")
            
            mode = detect_mode(query, intent_type, [])

            if mode == "chat":
                answer = call_ollama(query, model="hozpitality-llama", max_tokens=150)

                await websocket.send_json({
                    "type": "final",
                    "data": {
                        "type": intent_type or "chat",
                        "html": answer
                    },
                    "conversation_id": conversation_id
                })
                
                continue

            with ThreadPoolExecutor() as executor:
                db_future = executor.submit(search_db, final_query, intent_type, location)
                web_future = executor.submit(search_web, final_query)

                done, _ = wait([db_future, web_future], timeout=2)

                db_context = db_future.result()
                web_context = web_future.result() if web_future in done else []

            logger.debug(f"[DB RESULTS COUNT] {len(db_context)}")
            logger.debug(f"[WEB RESULTS COUNT] {len(web_context)}")

            if intent_type in ["job", "company", "professional"]:
                combined = db_context

            elif intent_type == "faq":
                combined = db_context if len(db_context) >= 2 else db_context + web_context[:2]

            else:
                combined = db_context + web_context[:2]

            logger.debug(f"[COMBINED COUNT] {len(combined)}")

            context = rerank(final_query, combined)
            logger.debug(f"[RERANKED COUNT] {len(context)}")

            if not context:
                logger.warning("[NO RESULTS] Empty context after rerank")

            memory = retrieve_memory(user_id, org_id, query)
            mode = detect_mode(query, intent_type, context)

            prompt = build_prompt(query, memory, context, intent_type, mode)

            logger.debug(f"[PROMPT GENERATED] Length: {len(prompt)}")

            # 🔹 LLM Call
            response = call_ollama(prompt, model="hozpitality-llama", max_tokens=700, stream=True)

            full = ""

            for token in response:
                full += token

            clean_html = clean_html_response(full)

            logger.info(f"[RESPONSE GENERATED] Length: {len(clean_html)}")

            await websocket.send_json({
                "type": "final",
                "data": {
                    "type": intent_type or "chat",
                    "html": clean_html
                },
                "conversation_id": conversation_id
            })

            save_message(conversation_id, "user", query)
            save_message(conversation_id, "assistant", full)

            store_memory(user_id, org_id, query)
            store_memory(user_id, org_id, full)

            store_last_ai_response(user_id, org_id, full)
            store_last_context(user_id, org_id, intent_type, context)

            await websocket.send_json({
                "type": "done",
                "conversation_id": conversation_id
            })

    except WebSocketDisconnect:
        logger.warning("[WS DISCONNECTED]")

    except Exception as e:
        logger.error(f"[WS ERROR] {str(e)}", exc_info=True)


@app.get("/conversations/{user_id}")
def get_conversations(user_id: int):
    try:
        with get_db() as conn:
            cur = conn.cursor()

            cur.execute("""
            SELECT id, title, updated_at
            FROM master_search_usersearchconversation
            WHERE user_id = %s
            ORDER BY updated_at DESC
            """, (user_id,))

            rows = cur.fetchall()
            cur.close()

            return [
                {"id": r[0], "title": r[1], "updated_at": str(r[2])}
                for r in rows
            ]

    except Exception as e:
        logger.error(f"[DB ERROR get_conversations] {e}", exc_info=True)
        return []


@app.get("/history/{user_id}/{conversation_id}")
def get_history(user_id: int, conversation_id: int):
    try:
        with get_db() as conn:
            cur = conn.cursor()

            cur.execute("""
            SELECT role, content, created_at
            FROM master_search_usersearchmessage
            WHERE conversation_id = %s
            ORDER BY created_at ASC
            """, (conversation_id,))

            rows = cur.fetchall()
            cur.close()

            return [
                {
                    "role": r[0],
                    "content": r[1],
                    "timestamp": str(r[2])
                }
                for r in rows
            ]

    except Exception as e:
        logger.error(f"[DB ERROR get_history] {e}", exc_info=True)
        return []


@app.websocket("/ws/test")
async def ws_test(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("connected")