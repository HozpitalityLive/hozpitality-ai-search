# ai_v2.py

import os
import json
import faiss
import time
import redis
import numpy as np
import requests
from fastapi import FastAPI, WebSocket
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from psycopg2.pool import SimpleConnectionPool
import psycopg2
import torch
from sentence_transformers import SentenceTransformer
import traceback
from elasticsearch.helpers import bulk

print("🔥 CUDA AVAILABLE:", torch.cuda.is_available())
print("🔥 GPU NAME:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

device = "cuda" if torch.cuda.is_available() else "cpu"

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

app = FastAPI()

redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)
es = Elasticsearch(
    "http://elasticsearch:9200"
)

EMBED_DIM = embedder.get_sentence_embedding_dimension()
global index, res
cpu_index = faiss.IndexFlatIP(EMBED_DIM)
if device == "cuda":
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
else:
    res = None
    index = cpu_index

documents = []

OLLAMA_URL = "http://ollama:11434/api/generate"

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

db_pool = SimpleConnectionPool(1, 10, **DB_CONFIG)

def load_data(force_reindex=False):
    global documents, index

    print("🔄 Loading data (optimized mode)")

    documents = []

    # 🔹 Reset FAISS only (fast)
    cpu_index = faiss.IndexFlatIP(EMBED_DIM)
    if device == "cuda":
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        index = cpu_index

    # 🔹 Check ES index
    index_exists = es.indices.exists(index="hozpitality")

    if force_reindex:
        print("🔥 FORCE REINDEX → deleting ES index")
        try:
            es.indices.delete(index="hozpitality")
            index_exists = False
        except Exception as e:
            print("❌ ES DELETE ERROR:", e)

    # 🔹 Create index ONLY if missing
    if not index_exists:
        try:
            print("📦 Creating ES index...")

            es.indices.create(
                index="hozpitality",
                body={
                    "mappings": {
                        "properties": {
                            "title": {"type": "text"},
                            "content": {"type": "text"},
                            "category": {"type": "keyword"},
                            "location": {"type": "keyword"}
                        }
                    }
                }
            )

            print("✅ ES index created")

        except Exception as e:
            print("⚠️ ES create skipped:", e)

    conn = db_pool.getconn()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, title, content, category_text, location_text, slug
        FROM master_search_mastersearchindex
        WHERE is_live = TRUE
    """)

    rows = cur.fetchall()
    texts = []
    actions = []

    print(f"📊 Rows fetched: {len(rows)}")

    for r in rows:
        text = (r[1] or "") + " " + (r[2] or "")
        texts.append(text)

        doc = {
            "id": r[0],
            "title": r[1],
            "content": (r[2] or "")[:200],
            "category": r[3],
            "location": r[4],
            "slug": r[5],
            "score": 1.0
        }

        documents.append(doc)

        # 🔹 Only index into ES if new or forced
        if not index_exists or force_reindex:
            actions.append({
                "_index": "hozpitality",
                "_id": r[0],
                "_source": {
                    "title": r[1],
                    "content": r[2],
                    "category": r[3],
                    "location": r[4]
                }
            })

    # 🔹 Bulk insert only if needed
    if actions:
        print("⚡ Bulk indexing ES...")
        bulk(es, actions)
        es.indices.refresh(index="hozpitality")

    # 🔹 Build FAISS (always needed)
    print("⚡ Building FAISS index...")
    vectors = embedder.encode(texts, normalize_embeddings=True)
    index.add(np.array(vectors))

    db_pool.putconn(conn)

    print(f"✅ Done | Docs: {len(documents)} | FAISS: {index.ntotal}")

def detect_mode(query: str):
    q = query.lower().strip()

    if len(q) <= 3 or q in ["hi", "hello", "hey", "ok", "thanks"]:
        return "chat"

    if any(x in q for x in ["how", "why", "steps", "process"]):
        return "faq"

    if any(x in q for x in ["job", "jobs", "company", "hotel", "restaurant"]):
        return "search"

    return "chat"

def choose_model(query, results):
    q = query.lower()

    if len(q) < 40:
        return "phi3-hoz"

    if any(x in q for x in ["how", "why", "steps", "process"]):
        return "llama3-hoz"

    if len(results) > 5:
        return "llama3-hoz"

    return "llama3-hoz"


def understand_query(q):
    q = q.lower()

    intent = "general"
    if "job" in q: intent = "job"
    elif "event" in q: intent = "event"
    elif "company" in q: intent = "company"

    return intent


def vector_search(query):
    query_vec = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec, 10)

    results = []

    for idx in I[0]:
        if idx == -1:
            continue

        if idx >= index.ntotal:
            print(f"⚠️ FAISS OUT OF RANGE: {idx} >= {index.ntotal}")
            continue

        if idx >= len(documents):
            print(f"⚠️ DOC MISMATCH: {idx} >= {len(documents)}")
            continue

        try:
            doc = documents[idx].copy()
            results.append(doc)
        except Exception as e:
            print(f"❌ DOC ACCESS ERROR: {idx}", e)

    return results


def elastic_search(query):
    res = es.search(index="hozpitality", query={
        "multi_match": {
            "query": query,
            "fields": ["title^2", "content"]
        }
    }, size=50)

    results = []
    for hit in res["hits"]["hits"]:
        doc = hit["_source"]
        doc["score"] = hit["_score"]
        results.append(doc)

    return results


def hybrid_search(query):
    try:
        vec = vector_search(query)
    except Exception as e:
        print("❌ VECTOR SEARCH ERROR:", e)
        vec = []
    esr = elastic_search(query)

    combined = {}

    for r in vec:
        r["score"] = r.get("score", 1.0) 
        combined[r["title"]] = r
        combined[r["title"]]["score"] *= 0.6

    for r in esr:
        if r["title"] in combined:
            combined[r["title"]]["score"] += r["score"] * 0.4
        else:
            combined[r["title"]] = r

    return sorted(combined.values(), key=lambda x: x["score"], reverse=True)


def personalize(user_id, results):
    key = f"user_pref:{user_id}"
    prefs = redis_client.get(key)

    if not prefs:
        return results

    prefs = json.loads(prefs)

    for r in results:
        if r["category"] in prefs:
            r["score"] += prefs[r["category"]] * 0.5

    return results


def track_click(user_id, category):
    key = f"user_pref:{user_id}"
    prefs = redis_client.get(key)

    if prefs:
        prefs = json.loads(prefs)
    else:
        prefs = {}

    prefs[category] = prefs.get(category, 0) + 1
    redis_client.set(key, json.dumps(prefs))


def get_cache(q):
    return redis_client.get(f"search:{q}")

def set_cache(q, data):
    redis_client.setex(f"search:{q}", 300, json.dumps(data))

def generate_intro(query: str):
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "phi3-hoz",
                "prompt": f"""
You are an AI assistant for Hozpitality.

TASK:
- Give a helpful, natural intro in 1–2 lines
- Do NOT repeat the user query
- Do NOT add explanation
- Do NOT use bullet points
- Keep it conversational

User:
{query}
""",
                "stream": False
            },
            timeout=3  
        )

        text = res.json().get("response", "").strip()

        text = text.replace("\n", " ").strip()

        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        return text

    except Exception as e:
        print("❌ INTRO ERROR:", e)
        return "Let me quickly help you with that..."

async def stream_answer(ws, query, results):
    import httpx

    context = ""
    for i, r in enumerate(results[:5]):
        context += f"{i+1}. {r['title']} - {r['content']}\n"

    model = choose_model(query, results)

    prompt = f"""
You are an intelligent AI assistant for Hozpitality.

IMPORTANT:
- Continue the answer naturally
- DO NOT repeat the introduction
- DO NOT restart the answer
- Assume the answer has already started

User Query:
{query}

Context:
{context}
"""

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True
                }
            ) as response:

                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)

                        if "response" in data:
                            await ws.send_json({
                                "type": "token",
                                "data": data["response"]
                            })

                        if data.get("done"):
                            break

    except Exception as e:
        print("❌ STREAM ERROR:", e)
        await ws.send_json({
            "type": "token",
            "data": "Error generating response"
        })

def generate_answer(query, results):
    context = ""
    for i, r in enumerate(results[:5]):
        context += f"{i+1}. {r['title']} - {r['content']}\n"

    model = choose_model(query, results)

    prompt = f"""
You are an intelligent AI assistant for Hozpitality.

User Query:
{query}

Context (if available):
{context}

INSTRUCTIONS:

- If context is relevant → use it
- If context is empty → answer like ChatGPT
- Be helpful, natural, conversational
- Do NOT say "no data found"
- Do NOT restrict yourself

If results exist:
- Mention useful insights from data

If no results:
- Answer intelligently using general knowledge

"""

    res = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })

    return res.json().get("response", "")





@app.post("/track-click")
def click(user_id: int, category: str):
    track_click(user_id, category)
    return {"status": "ok"}


@app.websocket("/ws/ai-search")
async def ws_search(ws: WebSocket):
    await ws.accept()
    print("✅ WebSocket connected")

    try:
        while True:
            raw = await ws.receive_text()
            print("📩 RAW:", raw)

            try:
                data = json.loads(raw)
            except:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            query = data.get("query", "").strip()
            user_id = data.get("user_id", 0)

            if not query:
                await ws.send_json({"type": "error", "message": "Query missing"})
                continue

            print(f"🔍 Query: {query}")

            mode = detect_mode(query)
            print(f"🧠 MODE: {mode}")

            results = []
            total = 0

            intro = generate_intro(query)

            if intro:
                await ws.send_json({
                    "type": "token",
                    "data": intro + "\n\n"
                })

            if mode == "search":
                try:
                    results = hybrid_search(query)
                    results = personalize(user_id, results)
                    total = len(results)
                except Exception as e:
                    print("❌ SEARCH ERROR:", e)

                await ws.send_json({
                    "type": "meta",
                    "total": total
                })

                for r in results[:10]:
                    await ws.send_json({
                        "type": "result",
                        "data": r
                    })

            await stream_answer(ws, query, results)

            await ws.send_json({
                "type": "done",
                "total": total
            })

    except Exception as e:
        print("❌ WS ERROR:", str(e))

    finally:
        print("🔌 Connection closed")
        await ws.close()


@app.on_event("startup")
def startup():
    print("\n🚀 ===== STARTUP BEGIN =====", flush=True)

    print("⏳ Waiting for Elasticsearch...", flush=True)

    es_ready = False

    for i in range(30):  
        try:
            print(f"🔁 Elasticsearch {i+1}/30 - pinging ES...", flush=True)

            if es.ping():
                print("✅ Elasticsearch ping successful", flush=True)

                try:
                    health = es.cluster.health()
                    print(f"📊 ES Health: {health['status']}", flush=True)
                except Exception as e:
                    print(f"⚠️ Failed to get ES health: {e}", flush=True)

                print("⏳ Extra wait for ES readiness (5s)...", flush=True)
                time.sleep(5)

                es_ready = True
                break

        except Exception as e:
            print(f"❌ ES ping failed: {e}", flush=True)

        time.sleep(2)

    if not es_ready:
        print("❌ Elasticsearch NOT reachable after retries", flush=True)
    else:
        print("✅ Elasticsearch is fully ready", flush=True)

    try:
        exists = es.indices.exists(index="hozpitality")
        print(f"🔍 Index exists before load: {exists}", flush=True)
    except Exception as e:
        print(f"❌ Failed to check index existence: {e}", flush=True)

    print("🚀 Calling load_data()...", flush=True)

    try:
        load_data()
        print("✅ load_data() completed", flush=True)
    except Exception as e:
        print("❌ load_data() FAILED:", str(e), flush=True)
        traceback.print_exc()

    try:
        exists = es.indices.exists(index="hozpitality")
        print(f"🔍 Index exists after load: {exists}", flush=True)

        if exists:
            count = es.count(index="hozpitality")["count"]
            print(f"📊 Indexed documents: {count}", flush=True)

    except Exception as e:
        print(f"❌ Post-load verification failed: {e}", flush=True)

    print("🏁 ===== STARTUP END =====\n", flush=True)



@app.post("/reindex")
def reindex():
    load_data(force_reindex=True)
    return {"status": "reindexed"}