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
from llama_cpp import Llama

import redis
import hashlib
import numpy as np
from dotenv import load_dotenv
load_dotenv()



redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
CACHE_TTL = 600

def cache_key(query):
    return "ai:" + hashlib.md5(query.encode()).hexdigest()

def simple_rerank(results):
    return results[:3]

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

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=os.cpu_count(),
    n_batch=512
)

db_pool = SimpleConnectionPool(1, 5, **DB_CONFIG)

memory_indexes = {}
memory_store = {}

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
        import json
        return json.loads(cached)

    domain_query = f"{query} (site:hozpitality.com OR site:instagram.com OR site:linkedin.com)"

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

        import json
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

def search_db(query):
    key = cache_key("db:" + query)
    cached = redis_client.get(key)
    if cached:
        import json
        return json.loads(cached)

    conn = db_pool.getconn()
    try:
        cur = conn.cursor()

        q_vec = get_embedding(query).tolist()

        cur.execute("""
        SELECT title, content, category_text, location_text, slug
        FROM master_search_mastersearchindex
        WHERE is_live = TRUE
        ORDER BY embedding <-> %s::vector
        LIMIT 3
        """, (q_vec,))

        rows = cur.fetchall()
        cur.close()

        result = [{
            "title": r[0],
            "content": (r[1] or "")[:150],
            "category": r[2],
            "location": r[3],
            "url": build_url(r[2], r[4])
        } for r in rows]

        import json
        redis_client.setex(key, CACHE_TTL, json.dumps(result))

        return result

    finally:
        db_pool.putconn(conn)


def build_prompt(query, memory, context):

    memory_text = "\n".join(memory[-2:]) if memory else ""

    context_text = ""
    for i, item in enumerate(context):
        context_text += f"""
[{i+1}]
Title: {item['title']}
Details: {item['content']}
Source: {item.get('url', item.get('location'))}
"""

    return f"""
You are a smart AI assistant for Hozpitality.com like ChatGPT or Gemini.

Query: {query}

Context:
{context_text}

Memory:
{memory_text}

INSTRUCTIONS:

1. Start with a short helpful introduction answering the query
2. Then show structured results (bullet points or sections)
3. Each result should include:
   - Title
   - Short explanation
   - Link (if available)
4. DO NOT invent missing steps
5. If data is incomplete → explain clearly but naturally
6. Keep tone helpful and human-like

FORMAT:

- Intro (2–3 lines)
- Then results list
- Then a helpful closing line
- Then ONE follow-up question

STYLE:

- Conversational but factual
- Clear and structured
- Helpful tone

ANSWER:
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

    memory = retrieve_memory(user_id, org_id, query)

    with ThreadPoolExecutor() as executor:
        db_future = executor.submit(search_db, query)
        web_future = executor.submit(search_web, query)

        done, _ = wait([db_future, web_future], timeout=2)

        db_context = db_future.result()

        web_context = web_future.result() if web_future in done else []

    context = simple_rerank(db_context + web_context)

    if not context:
        answer = "You can check the official website or latest announcements for more details."
    else:
        try:
            prompt = build_prompt(query, memory, context)
            output = llm(prompt, max_tokens=200)
            answer = output["choices"][0]["text"].strip()
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

    with ThreadPoolExecutor() as executor:
        db_future = executor.submit(search_db, query)
        web_future = executor.submit(search_web, query)

        done, _ = wait([db_future, web_future], timeout=2)

        db_context = db_future.result()

        web_context = web_future.result() if web_future in done else []

    context = simple_rerank(db_context + web_context)

    memory = retrieve_memory(user_id, org_id, query)
    prompt = build_prompt(query, memory, context)

    def generate():
        stream = llm(prompt, max_tokens=200, stream=True)
        for chunk in stream:
            yield chunk["choices"][0]["text"]


    return StreamingResponse(generate(), media_type="text/plain")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):

    origin = websocket.headers.get("origin")
    print("Incoming origin:", origin)

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            query = data["query"]
            user_id = data["user_id"]
            org_id = data["org_id"]

            conversation_id = data.get("conversation_id") or create_conversation(user_id, query[:30])

            with ThreadPoolExecutor() as executor:
                db_future = executor.submit(search_db, query)
                web_future = executor.submit(search_web, query)

                done, _ = wait([db_future, web_future], timeout=2)

                db_context = db_future.result()

                web_context = web_future.result() if web_future in done else []

            context = simple_rerank(db_context + web_context)

            memory = retrieve_memory(user_id, org_id, query)
            prompt = build_prompt(query, memory, context)

            stream = llm(prompt, max_tokens=200, stream=True)

            full = ""  # ✅ FIXED

            for chunk in stream:
                token = chunk["choices"][0]["text"]
                full += token

                await websocket.send_json({
                    "type": "token",
                    "data": token
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
        print("WebSocket disconnected")


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