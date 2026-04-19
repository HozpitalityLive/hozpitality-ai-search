# ai_v2.py

import os
import json
import faiss
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

print("🔥 CUDA AVAILABLE:", torch.cuda.is_available())
print("🔥 GPU NAME:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

device = "cuda" if torch.cuda.is_available() else "cpu"

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

app = FastAPI()

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
es = Elasticsearch("http://localhost:9200")

EMBED_DIM = embedder.get_sentence_embedding_dimension()

res = faiss.StandardGpuResources()
cpu_index = faiss.IndexFlatIP(EMBED_DIM)
index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
documents = []

OLLAMA_URL = "http://localhost:11434/api/generate"

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

db_pool = SimpleConnectionPool(1, 10, **DB_CONFIG)

def load_data():
    conn = db_pool.getconn()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, title, content, category_text, location_text, slug
        FROM master_search_mastersearchindex
        WHERE is_live = TRUE
    """)

    rows = cur.fetchall()
    texts = []

    for r in rows:
        text = (r[1] or "") + " " + (r[2] or "")
        texts.append(text)

        doc = {
            "id": r[0],
            "title": r[1],
            "content": (r[2] or "")[:200],
            "category": r[3],
            "location": r[4],
            "slug": r[5]
        }

        documents.append(doc)

        es.index(index="hozpitality", id=r[0], document=doc)

    vectors = embedder.encode(texts, normalize_embeddings=True)
    index.add(np.array(vectors))

    print(f"✅ Loaded {len(rows)} records")


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


def vector_search(query, k=50):
    q_vec = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(q_vec), k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= len(documents): continue
        doc = documents[idx].copy()
        doc["score"] = float(scores[0][i])
        results.append(doc)

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
    vec = vector_search(query)
    esr = elastic_search(query)

    combined = {}

    for r in vec:
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


def generate_answer(query, results):
    context = ""
    for i, r in enumerate(results[:5]):
        context += f"{i+1}. {r['title']} - {r['content']}\n"

    model = choose_model(query, results)

    print(f"🧠 Using model: {model}")

    prompt = f"""
You are an AI assistant for Hozpitality.

User Query:
{query}

Context:
{context}

Answer clearly like ChatGPT and helpfully using ONLY context.
"""

    res = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })

    return res.json().get("response", "")


@app.get("/ai-search")
def ai_search(q: str, user_id: int = 0):

    cached = get_cache(q)
    if cached:
        return json.loads(cached)

    intent = understand_query(q)

    results = hybrid_search(q)
    results = personalize(user_id, results)

    answer = generate_answer(q, results)

    final = {
        "query": q,
        "intent": intent,
        "answer": answer,
        "total": len(results),
        "results": results[:10]
    }

    set_cache(q, final)

    return final


@app.post("/track-click")
def click(user_id: int, category: str):
    track_click(user_id, category)
    return {"status": "ok"}


@app.websocket("/ws/ai-search")
async def ws_search(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()
            print("📩 RAW:", data)

            query = data["query"]
            user_id = data.get("user_id", 0)

            results = hybrid_search(query)
            results = personalize(user_id, results)

            for r in results[:10]:
                await ws.send_json({
                    "type": "result",
                    "data": r
                })

            answer = generate_answer(query, results)

            for chunk in answer.split():
                await ws.send_json({
                    "type": "token",
                    "data": chunk + " "
                })

            await ws.send_json({
                "type": "done",
                "total": len(results)
            })

    except Exception as e:
        print("❌ WS ERROR:", str(e))

        await ws.send_json({
            "status": "error",
            "message": str(e)
        })

    finally:
        await ws.close()


@app.on_event("startup")
def startup():
    load_data()