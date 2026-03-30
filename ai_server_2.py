# ai_server_2.py

import re
import json
import openai
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


openai.api_base = "http://localhost:8000/v1"
openai.api_key = ""



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


db_pool = SimpleConnectionPool(1, 5, **DB_CONFIG)

memory_indexes = {}
memory_store = {}

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
        res = openai.ChatCompletion.create(
            model="google/gemma-2b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )

        text = res["choices"][0]["message"]["content"]

        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
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
        res = openai.ChatCompletion.create(
            model="google/gemma-2b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.2
        )

        text = res["choices"][0]["message"]["content"]

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
You are a strict JSON API for search processing.

User Query: "{query}"
Categories: {categories}

YOUR TASK (STRICT):

1. SPELLING FIX ONLY:
- Fix spelling mistakes
- DO NOT merge words
- DO NOT remove words
- DO NOT add new words
- Keep sentence structure same

Example:
"find a job for cheiif in dubai"
→ "find a job for chef in dubai"

2. INTENT:
- FAQ if starts with "how to", "how do", "steps", "process"
- professional if "who is"
- job if contains job-related words
- else SEARCH

3. TYPE:
Must be ONE of: {categories}

4. LOCATION:
- Extract city or country only
- Return single word if possible
- Example: "dubai", "india", "mumbai"

5. KEYWORDS:
- Extract 2–4 important search terms
- lowercase only
- remove filler words like: find, a, the, for, in
- KEEP important roles (chef, manager etc.)

Example:
"find a job for chef in dubai"
→ "chef dubai"

STRICT OUTPUT (NO TEXT, ONLY JSON):

{{
"intent": "SEARCH",
"type": "job",
"keywords": "chef dubai",
"location": "dubai",
"rephrased_query": "find a job for chef in dubai"
}}
"""

    try:
        response = openai.ChatCompletion.create(
            model="google/gemma-2b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0  
        )

        text = response["choices"][0]["message"]["content"].strip()

        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)

        if match:
            data = json.loads(match.group())

            rq = data.get("rephrased_query", "").lower()

            rq = re.sub(r'\bin([a-z]+)', r'in \1', rq)

            data["rephrased_query"] = rq.strip()

            logger.info(f"Intent detected: {data}")

            return data

        raise ValueError("Invalid JSON")

    except Exception as e:
        logger.error(f"Intent error: {e}")

        return {
            "intent": "SEARCH",
            "type": "article",
            "keywords": query.lower().strip(),
            "location": "",
            "rephrased_query": query.strip()
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
    conn = db_pool.getconn()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO master_search_usersearchconversation (user_id, title, created_at, updated_at)
    VALUES (%s, %s, NOW(), NOW())
    RETURNING id
    """, (user_id, title))

    cid = cur.fetchone()[0]
    conn.commit()
    cur.close()
    db_pool.putconn(conn)

    return cid

def save_message(conversation_id, role, content):
    conn = db_pool.getconn()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO master_search_usersearchmessage (conversation_id, role, content, created_at)
    VALUES (%s, %s, %s, NOW())
    """, (conversation_id, role, content))

    conn.commit()
    cur.close()
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

    conn = db_pool.getconn()
    try:
        cur = conn.cursor()

        sql = """
        SELECT title, content, category_text, location_text, slug
        FROM master_search_mastersearchindex
        WHERE is_live = TRUE
        """

        params = []

        if intent_type == "job":
            sql += " AND LOWER(category_text) LIKE '%job%'"

        elif intent_type == "article":
            sql += " AND LOWER(category_text) LIKE '%article%'"

        elif intent_type == "company":
            sql += " AND LOWER(category_text) LIKE '%company%'"

        elif intent_type == "event":
            sql += " AND LOWER(category_text) LIKE '%event%'"

        elif intent_type == "supplier":
            sql += " AND LOWER(category_text) LIKE '%supplier%'"

        elif intent_type == "product":
            sql += " AND LOWER(category_text) LIKE '%product%'"

        elif intent_type == "awards":
            sql += " AND LOWER(category_text) LIKE '%award%'"

        elif intent_type == "professional":
            sql += " AND LOWER(category_text) LIKE '%professional%'"

        if location:
            sql += " AND LOWER(location_text) LIKE %s"
            params.append(f"%{location.lower()}%")

        words = [
            w for w in re.split(r"[,\s]+", query.lower())
            if w and len(w) > 2
        ]

        if words:
            conditions = []

            for w in words:
                conditions.append("LOWER(title) LIKE %s")
                params.append(f"%{w}%")

            for w in words:
                conditions.append("LOWER(content) LIKE %s")
                params.append(f"%{w}%")

            sql += " AND (" + " OR ".join(conditions) + ")"

        sql += """
        ORDER BY 
            CASE 
                WHEN LOWER(title) LIKE %s THEN 0
                ELSE 1
            END
        LIMIT 6
        """

        if words:
            params.append(f"%{words[0]}%")
        else:
            params.append("%")

        logger.info("SQL: %s", sql)
        logger.info("Params: %s", params)

        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()

        result = [{
            "title": r[0],
            "content": (r[1] or "")[:150],
            "category": r[2],
            "location": r[3],
            "url": build_url(r[2], r[4])
        } for r in rows]

        redis_client.setex(key, CACHE_TTL, json.dumps(result))

        return result

    except Exception as e:
        print("DB ERROR:", e)
        return []

    finally:
        db_pool.putconn(conn)


def build_prompt(query, memory, context):

    memory_text = "\n".join(memory[-2:]) if memory else ""

    context_text = ""
    for i, item in enumerate(context):
        context_text += f"""
[{i+1}]
Title: {item['title']}
Details: {item['content'][:80]}
Source: {item.get('url', item.get('location'))}
"""

    return f"""
You are a smart AI assistant for Hozpitality.com.

User Query: {query}

Context Data:
{context_text}

Memory:
{memory_text}

STRICT INSTRUCTIONS:

1. OUTPUT MUST BE PURE HTML ONLY (NO MARKDOWN, NO ```)

2. RESPONSE STRUCTURE (MANDATORY):

<div class="ai-response">

  <!-- INTRO -->
  <div class="ai-intro">
    <p>Write 2-3 lines introduction explaining the answer clearly.</p>
  </div>

  <!-- RESULTS -->
  <div class="ai-results">
    Use ANY clean HTML format based on data:
    - paragraphs OR
    - cards (div blocks) OR
    - table OR
    - list (only if needed)

    Each result MUST include:
    - Title (clickable <a>)
    - Short description (1-2 lines)
    - Optional location
  </div>

  <!-- FOLLOW UP -->
  <div class="ai-followup">
    <p><strong>Follow-up:</strong> Ask ONE relevant question.</p>
  </div>

</div>

3. JOB RULE:
- If query is job-related → show at least 5 results

4. DO NOT:
- Use markdown (**, ###, ``` etc)
- Do not return plain text
- Do not wrap in code block

5. KEEP UI:
- Clean
- Minimal
- Readable spacing

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
            prompt = build_prompt(query, memory, context)
            response = openai.ChatCompletion.create(
                model="google/gemma-2b-it",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )

            answer = response["choices"][0]["message"]["content"]
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
    prompt = build_prompt(query, memory, context)

    def generate():
        response = openai.ChatCompletion.create(
            model="google/gemma-2b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            stream=True
        )

        for chunk in response:
            token = chunk["choices"][0]["delta"].get("content", "")
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

            conversation_id = data.get("conversation_id") or create_conversation(user_id, query[:30])
            logger.debug(f"[CONVERSATION] ID: {conversation_id}")

            # 🔹 Intent Detection
            intent_data = detect_intent_llm(query)
            logger.debug(f"[INTENT] {intent_data}")

            base_query = intent_data.get("keywords") or query

            logger.debug(f"[BASE QUERY] {base_query}")

            # final_query = build_final_query(base_query)
            final_query = base_query
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

            logger.info(f"[FINAL QUERY] {final_query}")
            logger.info(f"[INTENT TYPE] {intent_type} | Location: {location}")

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

            memory = []
            prompt = build_prompt(query, memory, context)

            logger.debug(f"[PROMPT GENERATED] Length: {len(prompt)}")

            # 🔹 LLM Call
            response = openai.ChatCompletion.create(
                model="google/gemma-2b-it",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700,
                stream=True
            )

            full = ""

            for chunk in response:
                token = chunk["choices"][0]["delta"].get("content", "")
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
    conn = db_pool.getconn()
    cur = conn.cursor()

    cur.execute("""
    SELECT id, title, updated_at
    FROM master_search_usersearchconversation
    WHERE user_id = %s
    ORDER BY updated_at DESC
    """, (user_id,))

    rows = cur.fetchall()

    return [
        {"id": r[0], "title": r[1], "updated_at": str(r[2])}
        for r in rows
    ]


@app.get("/history/{user_id}/{conversation_id}")
def get_history(user_id: int, conversation_id: int):
    conn = db_pool.getconn()
    cur = conn.cursor()

    cur.execute("""
    SELECT role, content, created_at
    FROM master_search_usersearchmessage
    WHERE conversation_id = %s
    ORDER BY created_at ASC
    """, (conversation_id,))

    rows = cur.fetchall()

    return [
        {
            "role": r[0],
            "content": r[1],
            "timestamp": str(r[2])
        }
        for r in rows
    ]


@app.websocket("/ws/test")
async def ws_test(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("connected")