# ai_server_2.py

import os
import faiss
import psycopg2
from psycopg2.pool import SimpleConnectionPool

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

import redis
import hashlib
from fastapi import WebSocket , WebSocketDisconnect
import numpy as np
from dotenv import load_dotenv

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

CACHE_TTL = 60

def cache_key(query):
    return "ai:" + hashlib.md5(query.encode()).hexdigest()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


MODEL_PATH = "/home/dev/models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MAX_MEMORY = 50

app = FastAPI()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = embedder.get_sentence_embedding_dimension()

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=4
)

load_dotenv()

db_pool = SimpleConnectionPool(1, 5, **DB_CONFIG)

@app.on_event("shutdown")
def shutdown():
    if db_pool:
        db_pool.closeall()

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
    
    vec = np.array([get_embedding(text)]).astype("float32")
    idx.add(vec)
    
    store.append(text)
    if len(store) > MAX_MEMORY:
        store.pop(0)

def retrieve_memory(user_id, org_id, query):
    idx, store = get_memory(user_id, org_id)
    if not store:
        return []
    q_vec = embedder.encode([query])
    D, I = idx.search(q_vec, 5)
    return [store[i] for i in I[0] if i < len(store)]

embedding_cache = {}

MAX_EMBED_CACHE = 1000

def get_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]

    vec = embedder.encode([text])[0]

    if len(embedding_cache) > MAX_EMBED_CACHE:
        embedding_cache.clear()

    embedding_cache[text] = vec
    return vec

def rerank_results(query, rows):
    q_vec = get_embedding(query)

    scored = []

    for r in rows:
        text = f"{r[0]} {r[1]} {r[2]} {r[3]}"
        doc_vec = get_embedding(text)

        score = float(q_vec @ doc_vec)

        category = (r[2] or "").lower()

        if "job" in category:
            score *= 1.1
        elif "professional" in category:
            score *= 1.2
        elif "company" in category:
            score *= 1.05

        scored.append((score, r))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [r for _, r in scored[:3]]
    
def search_db(query):
    key = cache_key("search:" + query)

    cached = redis_client.get(key)
    if cached:
        import json
        return json.loads(cached)

    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()

        q_vec = get_embedding(query).tolist()

        cur.execute("""
        SELECT title, content, category_text, location_text
        FROM master_search_mastersearchindex
        WHERE is_live = TRUE
        ORDER BY embedding <-> %s::vector
        LIMIT 5
        """, (q_vec,))

        rows = cur.fetchall()
        cur.close()

        # 🔥 APPLY RERANKING
        rows = rerank_results(query, rows)

        # 🔥 STRUCTURED CONTEXT
        result = []
        for r in rows:
            result.append({
                "title": r[0],
                "content": (r[1] or "")[:300],
                "category": r[2],
                "location": r[3]
            })

        import json
        redis_client.setex(key, CACHE_TTL, json.dumps(result))

        return result

    except Exception as e:
        print("DB search error:", e)
        return []

    finally:
        if conn:
            db_pool.putconn(conn)

def create_conversation(user_id, title):
    conn = db_pool.getconn()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO user_search_conversation (user_id, title, created_at, updated_at)
    VALUES (%s, %s, NOW(), NOW())
    RETURNING id
    """, (user_id, title))

    conversation_id = cur.fetchone()[0]

    conn.commit()
    cur.close()
    db_pool.putconn(conn)

    return conversation_id


def save_message(conversation_id, role, content):
    conn = db_pool.getconn()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO user_search_message (conversation_id, role, content, created_at)
    VALUES (%s, %s, %s, NOW())
    """, (conversation_id, role, content))

    conn.commit()
    cur.close()
    db_pool.putconn(conn)


def generate_title(query):
    out = llm(f"Generate short title (max 6 words): {query}", max_tokens=20)
    return out["choices"][0]["text"].strip()


def build_prompt(query, memory, context):

    memory_text = "\n".join(memory[-3:]) if memory else ""

    context_text = ""

    for i, item in enumerate(context):
        context_text += f"""
[{i+1}] {item['category'].upper()}
Title: {item['title']}
Location: {item['location']}
Details: {item['content']}
"""

    return f"""
You are an advanced AI assistant like ChatGPT.

USER QUERY:
{query}

USER MEMORY:
{memory_text}

RETRIEVED DATA:
{context_text}

INSTRUCTIONS:

1. Understand user intent deeply
2. Use retrieved data when relevant
3. If multiple results exist → compare or summarize
4. If no exact data → still answer intelligently
5. Use clean HTML (<p>, <ul>, <b>)
6. Be natural, helpful, human-like
7. Do NOT copy raw database text

RESPONSE STYLE:

- Start with a direct answer
- Then expand with helpful context
- Use bullet points if useful

FINAL ANSWER:
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

    conversation_id = req.conversation_id
    if not conversation_id:
        title = generate_title(query)
        conversation_id = create_conversation(user_id, title)
        
    cache_k = cache_key(f"{user_id}:{org_id}:{query}")

    cached_answer = redis_client.get(cache_k)
    if cached_answer:
        save_message(conversation_id, "user", query)
        save_message(conversation_id, "assistant", cached_answer)

        store_memory(user_id, org_id, query)
        store_memory(user_id, org_id, cached_answer)

        return {
            "conversation_id": conversation_id,
            "answer": cached_answer,
            "cached": True
        }

    memory = retrieve_memory(user_id, org_id, query)
    context = search_db(query)

    prompt = build_prompt(query, memory, context)

    output = llm(prompt, max_tokens=500)
    answer = output["choices"][0]["text"].strip()

    save_message(conversation_id, "user", query)
    save_message(conversation_id, "assistant", answer)

    store_memory(user_id, org_id, query)
    store_memory(user_id, org_id, answer)


    redis_client.setex(cache_k, CACHE_TTL, answer)

    return {
        "conversation_id": conversation_id,
        "answer": answer
    }


@app.post("/chat-stream")
def chat_stream(req: ChatRequest):

    query = req.query
    user_id = req.user_id
    org_id = req.org_id

    conversation_id = req.conversation_id
    if not conversation_id:
        title = generate_title(query)
        conversation_id = create_conversation(user_id, title)

    memory = retrieve_memory(user_id, org_id, query)
    context = search_db(query)

    prompt = build_prompt(query, memory, context)

    def generate():
        full = ""

        stream = llm(prompt, max_tokens=500, stream=True)

        for chunk in stream:
            token = chunk["choices"][0]["text"]
            full += token
            yield token

        save_message(conversation_id, "user", query)
        save_message(conversation_id, "assistant", full)

        store_memory(user_id, org_id, query)
        store_memory(user_id, org_id, full)

    return StreamingResponse(generate(), media_type="text/plain")



@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            query = data["query"]
            user_id = data["user_id"]
            org_id = data["org_id"]
            conversation_id = data.get("conversation_id")

            if not conversation_id:
                title = generate_title(query)
                conversation_id = create_conversation(user_id, title)

            memory = retrieve_memory(user_id, org_id, query)
            context = search_db(query)

            prompt = build_prompt(query, memory, context)

            full = ""

            stream = llm(prompt, max_tokens=500, stream=True)

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
