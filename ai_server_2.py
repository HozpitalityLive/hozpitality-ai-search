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
from site_context import SITE_CONTEXT , ADDITIONAL_INSTRUCTION
import re




LOG_FILE = os.path.join(os.getcwd(), "app.log")

AGENT_TOOLS = [
    {"name": "job", "desc": "search jobs"},
    {"name": "article", "desc": "search articles"},
    {"name": "professional", "desc": "search people"},
    {"name": "company", "desc": "search companies"},
    {"name": "event", "desc": "search events"},
    {"name": "supplier", "desc": "search suppliers"},
    {"name": "product", "desc": "search products"},
    {"name": "awards", "desc": "search awards"},
    {"name": "faq", "desc": "answer how-to questions"}
]


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

def execute_agent_tool(tool, query, location=None):
    try:
        if tool == "faq":
            return generate_ai_fallback(query)

        return search_db(query, tool, location)

    except Exception as e:
        logger.error(f"[AGENT TOOL ERROR] {e}")
        return []

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

def agent_decide(query: str):
    prompt = f"""
You are an AI agent for Hozpitality.com.

User Query: "{query}"

Categories:
['job', 'article', 'professional', 'faq', 'company', 'event', 'supplier', 'product', 'awards']


GOAL:
- Understand what user REALLY wants
- Choose ONLY ONE category

RULES:

- "how", "steps", "process" → faq
- job search → job
- company/brand → company
- person → professional
- awards → ONLY if word exists
- otherwise choose best match

IMPORTANT:

- DO NOT hallucinate
- DO NOT misclassify job ↔ faq blindly
- Use meaning

RETURN JSON ONLY:

{{
  "type": "job",
  "query": "chef jobs",
  "location": "dubai"
}}
"""

    try:
        res = call_ollama(prompt, stream=False,model="hozpitality-llama", max_tokens=120)

        match = re.search(r'\{.*\}', res, re.DOTALL)
        if match:
            return json.loads(match.group())

    except Exception as e:
        logger.error(f"[AGENT ERROR] {e}")

    return {
        "type": "faq",
        "query": query,
        "location": ""
    }



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
    
    if intent_type == "awards":
        return "chat" 

    return "list"



def smart_chunk(text, size=12):
    words = re.findall(r'\S+\s*', text)
    chunk = ""

    for w in words:
        if len(chunk) + len(w) > size:
            yield chunk
            chunk = w
        else:
            chunk += w

    if chunk:
        yield chunk


def call_ollama(prompt, stream=True, model="hozpitality-llama", max_tokens=600):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {"temperature": 0.2, "num_predict": max_tokens}
    }

    if not stream:
        res = requests.post(url, json=payload, timeout=180)
        return res.json().get("response", "")

    def generator():
        with requests.post(url, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if not line:
                    continue

                data = json.loads(line.decode("utf-8"))
                token = data.get("response", "")

                if not token:
                    continue

                if token:
                    yield token

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

FOLLOW-UP = TRUE ONLY if:
- very short replies: yes, ok, continue, show more
- clearly refers to previous answer WITHOUT new topic

FOLLOW-UP = FALSE if:
- contains ANY domain keywords (awards, jobs, company, event, etc.)
- contains a full sentence query
- introduces a new topic

IMPORTANT:
"tell me about awards"
"I want to know more about awards"
→ MUST be FALSE (new query, NOT follow-up)

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
        res = call_ollama(prompt, stream=False,model="hozpitality-phi3", max_tokens=60)

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
        res = call_ollama(prompt,stream=False, model="hozpitality-phi3", max_tokens=60)

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
        
        res = call_ollama(prompt,stream=False, model="hozpitality-phi3", max_tokens=100)
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

    prompt = f"""
You are a STRICT JSON API for search intent classification for a hospitality platform.

User Query: "{query}"

Categories:
['job', 'article', 'professional', 'faq', 'company', 'event', 'supplier', 'product', 'awards']

CORE INSTRUCTION

- Understand what the USER actually wants (intent), not just keywords
- Classify into ONE best category from the list
- DO NOT rely on rigid keyword priority
- Use semantic understanding

INTENT GUIDELINES

1. JOB
- User is searching for job listings
- Examples:
  "hotel jobs in dubai"
  "chef hiring uae"

2. FAQ
- User is asking HOW / guidance / steps
- Even if "job" is mentioned

Examples:
"how do i find a job in la"
"how to apply for hotel jobs"
→ type = "faq"

3. PROFESSIONAL
- About a person

4. COMPANY
- About a company or brand

5. AWARDS
- ONLY if query explicitly mentions:
  "award", "awards", "nomination", "winner"

6. DEFAULT
- Choose best fit

STRICT RULES

- DO NOT hallucinate categories
- DO NOT change query meaning
- DO NOT force classification based on one keyword
- Use full query meaning

LOCATION
- Extract city or country if present
- Else ""

KEYWORDS
- 2–4 important words
- lowercase
- no commas


OUTPUT (STRICT JSON ONLY):

{{
"intent": "SEARCH",
"type": "faq",
"keywords": "find job los angeles",
"location": "los angeles",
"rephrased_query": "{query}"
}}
"""

    try:
        answer = call_ollama(prompt, model="hozpitality-llama", max_tokens=300,stream=False)

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
                if intent_type == "awards":
                    sql += " AND LOWER(category_text) = 'awards'"  
                else:
                    sql += " AND LOWER(category_text) LIKE %s"
                    params.append(f"%{intent_type.lower()}%")

            if location:
                sql += " AND LOWER(location_text) LIKE %s"
                params.append(f"%{location.lower()}%")

            return sql, params

        if full_phrase:
            sql, params = base_sql()

            if intent_type == "awards":
                sql += """
                AND LOWER(title) LIKE %s
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
                    full_phrase,
                    f"%{full_phrase}%"
                ])
            else:
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
                if intent_type == "awards":
                    conditions.append("LOWER(title) LIKE %s")
                    params.append(f"%{w}%")
                else:
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


def generate_ai_fallback(query: str):
    prompt = f"""
You are an AI assistant for Hozpitality.com.

User Query: "{query}"
Platform Context: {SITE_CONTEXT}

GOAL:
- Answer like ChatGPT (helpful, clear, complete)
- BUT strictly within Hozpitality platform


STRICT RULES:

- NEVER mention external platforms:
  Indeed, LinkedIn, Glassdoor, Naukri, Monster

- NEVER say:
  "search online", "use other websites"


INSTEAD:

- Explain using Hozpitality features:
  - job search
  - filters (location, role)
  - applying to jobs
  - company profiles


STYLE:

- Friendly and natural
- Step-by-step if needed
- Practical guidance


OUTPUT:
- HTML ONLY (no markdown)

FORMAT:

<div class="ai-response">
  <div class="ai-intro">
    <p>Helpful introduction</p>
  </div>

  <div class="ai-results">
    <ul>
      <li>Step 1</li>
      <li>Step 2</li>
    </ul>
  </div>

  <div class="ai-followup">
    <p><strong>Follow-up:</strong> Ask something relevant</p>
  </div>
</div>
"""

    try:
        response = call_ollama(
            prompt,
            model="hozpitality-llama",
            max_tokens=250,
            stream=False
        )
        return clean_html_response(response)

    except Exception as e:
        logger.error(f"[AI FALLBACK ERROR] {e}")
        return "<div class='ai-response'><p>Sorry, something went wrong.</p></div>"


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

ADDITIONAL_INSTRUCTION: {ADDITIONAL_INSTRUCTION}

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

STRICT PLATFORM RULES (VERY IMPORTANT):
- You are ONLY an assistant for Hozpitality.com
- ALL answers MUST stay within Hozpitality platform

NEVER mention external platforms:
Indeed, LinkedIn, Glassdoor, Naukri, Monster, etc.

IF NO CONTEXT DATA:

- DO NOT use general internet knowledge
- DO NOT suggest external websites

INSTEAD:
- Explain how to use Hozpitality features:
  - job search
  - filters (location, role)
  - applying to jobs
  - company pages

COMPANIES:

- Marriott, Hyatt etc. are ALLOWED
- BUT ONLY if present in Context Data

FAIL RULE:

If ANY external platform is mentioned → response is INVALID

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
- Give steps ONLY using Hozpitality platform
- Do NOT give general internet advice
- Do NOT mention external websites or platforms

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
        # if len(db_context) < 2:
        #     combined = db_context + web_context[:2]
        # else:
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

            answer = call_ollama(prompt, model="hozpitality-llama", max_tokens=600)
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

            logger.info(f"[QUERY] User:{user_id} Org:{org_id} : {query}")

            conversation_id = data.get("conversation_id")

            if user_id != 0:
                if not conversation_id:
                    conversation_id = create_conversation(user_id, query[:30])
            else:
                conversation_id = None
            logger.debug(f"[CONVERSATION] ID: {conversation_id}")

            intent_data = detect_intent_llm(query)
            logger.debug(f"[INTENT] {intent_data}")

            await websocket.send_json({
                "type": "start",
                "message": "Thinking...",
                "conversation_id": conversation_id
            })

            base_query = intent_data.get("keywords") or query

            logger.debug(f"[BASE QUERY] {base_query}")

            final_query = base_query
            intent_type = intent_data.get("type")

            last_ai = get_last_ai_response(user_id, org_id)
            followup = detect_followup_llm(query, last_ai)

            logger.info(f"[FOLLOWUP DETECT] {followup}")

            if followup.get("is_followup") and any(
                k in query.lower() for k in ["award", "job", "company", "event"]
            ):
                logger.warning("Forcing follow-up FALSE due to new topic")
                followup["is_followup"] = False

            # --- FOLLOW-UP LOGIC (Streaming added) ---
            if followup.get("is_followup"):
                last_ctx = get_last_context(user_id, org_id)

                if last_ctx:
                    intent_type = last_ctx.get("intent")
                    context = last_ctx.get("context")

                    logger.info("[FOLLOW-UP MODE ACTIVATED]")

                    ftype = followup.get("type")
                    mode = "list" if ftype == "refine" else "chat" if ftype == "chat" else "single"

                    prompt = build_prompt(
                        query,
                        retrieve_memory(user_id, org_id, query),
                        context,
                        intent_type,
                        mode
                    )

                    # START STREAMING FOR FOLLOWUP
                    full_response = ""
                    for chunk in call_ollama(prompt, stream=True, model="hozpitality-llama", max_tokens=600):
                        full_response += chunk
                        await websocket.send_json({
                            "type": "token",
                            "data": chunk,   
                            "conversation_id": conversation_id
                        })

                    store_last_ai_response(user_id, org_id, full_response)
                    
                    # Final signal for this turn
                    
                    clean_html = clean_html_response(full_response)

                    await websocket.send_json({
                        "type": "final",
                        "data": {
                            "type": intent_type or "chat",
                            "html": clean_html
                        },
                        "conversation_id": conversation_id
                    })

                    await websocket.send_json({
                        "type": "done",
                        "conversation_id": conversation_id
                    })
                    continue

            location = intent_data.get("location")
            if location:
                final_query += f" {location}"

            logger.info(f"[FINAL QUERY] {final_query}")
            logger.info(f"[INTENT TYPE] {intent_type} | Location: {location}")
            
            mode = detect_mode(query, intent_type, [])

            # --- DIRECT CHAT MODE (Streaming added) ---
            if mode == "chat":
                full_response = ""
                for chunk in call_ollama(query, stream=True, model="hozpitality-llama", max_tokens=600):
                    full_response += chunk
                    await websocket.send_json({
                            "type": "token",
                            "data": chunk,   
                            "conversation_id": conversation_id
                    })
                
                
                clean_html = clean_html_response(full_response)

                await websocket.send_json({
                    "type": "final",
                    "data": {
                        "type": intent_type or "chat",
                        "html": clean_html
                    },
                    "conversation_id": conversation_id
                })

                await websocket.send_json({
                    "type": "done",
                    "conversation_id": conversation_id
                })
                continue

            with ThreadPoolExecutor() as executor:
                db_future = executor.submit(search_db, final_query, intent_type, location)
                web_future = executor.submit(search_web, final_query)

                done, _ = wait([db_future, web_future], timeout=2)

                db_context = db_future.result()
                web_context = web_future.result() if web_future in done else []

            if intent_type in ["job", "company", "professional", "faq", "awards"]:
                combined = db_context
            else:
                combined = db_context + web_context[:2]

            context = rerank(final_query, combined)

            # --- NO RESULTS / FALLBACK ---
            if not context:
                logger.warning("[NO RESULTS] - switching to AI mode")

                prompt = f"""
            User Query: "{query}"

            Give a helpful response within Hozpitality platform.
            Return HTML only.
            """

                full_response = ""

                for chunk in call_ollama(prompt, stream=True, model="hozpitality-llama", max_tokens=600):
                    full_response += chunk

                    await websocket.send_json({
                        "type": "token",
                        "data": chunk,
                        "conversation_id": conversation_id
                    })

                clean_html = clean_html_response(full_response)

                if user_id != 0 and conversation_id:
                    save_message(conversation_id, "user", query)
                    save_message(conversation_id, "assistant", clean_html)

                if user_id != 0:
                    store_memory(user_id, org_id, query)
                    store_memory(user_id, org_id, clean_html)
                    store_last_ai_response(user_id, org_id, clean_html)
                    store_last_context(user_id, org_id, intent_type, [])
    


                await websocket.send_json({
                    "type": "final",
                    "data": {
                        "type": "chat",
                        "html": clean_html
                    },
                    "conversation_id": conversation_id
                })

                await websocket.send_json({
                    "type": "done",
                    "conversation_id": conversation_id
                })

                continue

            # (Streaming added) ---
            memory = retrieve_memory(user_id, org_id, query)
            mode = detect_mode(query, intent_type, context)
            prompt = build_prompt(query, memory, context, intent_type, mode)

            try:
                full_response = ""
                # Looping through generator chunks
                for chunk in call_ollama(prompt, stream=True, model="hozpitality-llama", max_tokens=600):
                    full_response += chunk
                    await websocket.send_json({
                        "type": "token",
                        "data": chunk,  
                        "conversation_id": conversation_id
                    })

                
                clean_html = clean_html_response(full_response)


                await websocket.send_json({
                    "type": "final",
                    "data": {
                        "type": intent_type or "chat",
                        "html": clean_html
                    },
                    "conversation_id": conversation_id
                })

            except Exception as e:
                logger.error(f"[LLM ERROR] {e}")
                clean_html = "Something went wrong. Please try again."
                await websocket.send_json({
                    "type": "final",
                    "data": {"type": "error", "html": clean_html},
                    "conversation_id": conversation_id
                })

            #  SAVE DATA (Full response used for saving)
            if user_id != 0 and conversation_id:
                save_message(conversation_id, "user", query)
                save_message(conversation_id, "assistant", clean_html)

            if user_id != 0:
                store_memory(user_id, org_id, query)
                store_memory(user_id, org_id, clean_html)
                store_last_ai_response(user_id, org_id, clean_html)
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


@app.get("/")
def health():
    return {"status": "ok"}