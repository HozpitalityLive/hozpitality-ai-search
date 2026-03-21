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
        SELECT id, title, category_text, ai_keywords
        FROM master_search_mastersearchindex
        WHERE is_live = TRUE
    """)

    print(f"[FAISS] Rows received: {len(rows)}")

    embeddings = []
    vector_data = []

    for i, r in enumerate(rows):
        text = f"{r[1]} {r[2]} {r[3]}"
        print(f"[FAISS] Processing row {i}: {r[1]}")

        emb = get_embedding(text)
        embeddings.append(emb)

        vector_data.append({
            "id": r[0],
            "title": r[1],
            "category": r[2],
            "keywords": r[3]
        })

    if not embeddings:
        print("⚠️ No embeddings found — check DB data or query")
        return

    vectors = np.array(embeddings)
    print("[FAISS] Vector shape:", vectors.shape)

    vector_index = faiss.IndexFlatL2(vectors.shape[1])
    vector_index.add(vectors)

    print("✅ FAISS READY:", len(vector_data))

def semantic_search(query, k=5):
    print(f"\n[SEARCH] Semantic search for: {query}")

    if vector_index is None:
        print("⚠️ FAISS index is not initialized")
        return []

    q_vec = np.array([get_embedding(query)])
    print("[SEARCH] Query embedding shape:", q_vec.shape)

    distances, indices = vector_index.search(q_vec, k)

    print("[SEARCH] Indices:", indices)
    print("[SEARCH] Distances:", distances)

    return [vector_data[i] for i in indices[0] if i < len(vector_data)]

def fetch_db_context():
    rows = safe_db_execute("""
        SELECT DISTINCT title, category_text, ai_keywords
        FROM master_search_mastersearchindex
        WHERE is_live = TRUE
        ORDER BY id DESC
        LIMIT 5
    """)

    return [{"title": r[0], "category": r[1], "keywords": r[2]} for r in rows]

def hybrid_search(query):
    print("\n[HYBRID] Running hybrid search")

    sem = semantic_search(query)
    db = fetch_db_context()

    print(f"[HYBRID] Semantic results: {len(sem)}")
    print(f"[HYBRID] DB fallback results: {len(db)}")

    combined = {}

    for r in sem:
        combined[r["title"]] = {"data": r, "score": 0.7}

    for r in db:
        if r["title"] in combined:
            combined[r["title"]]["score"] += 0.3
        else:
            combined[r["title"]] = {"data": r, "score": 0.3}

    final = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:5]

    print(f"[HYBRID] Final results: {len(final)}")

    return [x["data"] for x in final]

def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None

def generate_full_ai(query, context, history):

    prompt = f"""
You are an AI assistant.

User Query:
{query}

Context:
{context}

History:
{history}

TASK:
1. Understand intent
2. Detect action (apply_job, vote_award, nominate_award, none)
3. Answer clearly

Return JSON:
{{
 "answer": "...",
 "action": "..."
}}
"""

    print("\n[LLM] Sending request")
    print("[LLM] Query:", query)
    print("[LLM] Context:", context)
    print("[LLM] History:", history)

    try:
        r = requests.post(LLM_URL, json={
            "prompt": f"[INST] {prompt} [/INST]",
            "n_predict": 140,
            "temperature": 0.3
        }, timeout=8)

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

def build_context(context):
    return "\n".join([f"{i+1}. {c['title']}" for i, c in enumerate(context)]) if context else "No strong results."

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

    return {
        "answer": ai.get("answer"),
        "action": ai.get("action"),
        "context": context
    }

@app.on_event("startup")
def startup():
    build_vector_index()

@app.get("/")
def health():
    return {"status": "FINAL OPTIMIZED AI READY"}