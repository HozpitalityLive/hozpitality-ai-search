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

from ai_server import (
    search_db,
    apply_priority_sorting,
    create_conversation,
    save_message,
    store_last_ai_response,
    get_last_ai_response,
    store_last_context,
    get_last_context,
    retrieve_memory,
    store_memory,
    detect_followup_llm
)
import asyncio

print("🔥 CUDA AVAILABLE:", torch.cuda.is_available())
print("🔥 GPU NAME:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

device = "cuda" if torch.cuda.is_available() else "cpu"

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

app = FastAPI()

redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

es = Elasticsearch(
    "http://elasticsearch:9200",
    request_timeout=120,
    max_retries=5,
    retry_on_timeout=True
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

def chunked_bulk(es, actions, chunk_size=1000):
    for i in range(0, len(actions), chunk_size):
        chunk = actions[i:i+chunk_size]

        print(f"⚡ ES chunk {i} → {i+len(chunk)}", flush=True)

        bulk(
            es,
            chunk,
            request_timeout=120
        )

        time.sleep(0.2) 

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
                            "location": {"type": "keyword"},
                            "slug": {"type": "keyword"} 
                        }
                    }
                },
                request_timeout=30
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

        category_raw = (r[3] or "").lower()

        if "job" in category_raw:
            category = "job"
        elif "company" in category_raw:
            category = "company"
        elif "candidate" in category_raw or "profile" in category_raw:
            category = "professional"
        elif "supplier" in category_raw:
            category = "supplier"
        elif "product" in category_raw:
            category = "product"
        elif "event" in category_raw:
            category = "event"
        elif "article" in category_raw or "blog" in category_raw:
            category = "article"
        elif "award" in category_raw:
            category = "award"
        elif "faq" in category_raw:
            category = "faq"
        else:
            category = "general"

        text = " ".join([
            r[1] or "",
            r[2] or "",
            r[3] or "",  
            category, 
            r[4] or "",   
        ])
        texts.append(text)

        doc = {
            "id": r[0],
            "title": r[1],
            "content": (r[2] or "")[:200],
            "category": category,
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
                    "category": category,
                    "location": r[4],
                    "slug": r[5]
                }
            })

    # 🔹 Bulk insert only if needed
    if actions:
        print("⚡ Bulk indexing ES...")
        chunked_bulk(es, actions)
        es.indices.refresh(index="hozpitality")

    # 🔹 Build FAISS (always needed)
    print("⚡ Building FAISS index...")
    vectors = embedder.encode(texts, normalize_embeddings=True)
    index.add(np.array(vectors))

    db_pool.putconn(conn)

    print(f"✅ Done | Docs: {len(documents)} | FAISS: {index.ntotal}")

async def safe_send(ws, data):
    try:
        if ws.client_state.name == "CONNECTED":
            await ws.send_json(data)
    except:
        pass

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


def detect_intent_llm(query: str):
    q = query.lower().strip()

    print("🎯 Detect Intent LLM:", query, flush=True)

    job_keywords = [
        "job", "jobs", "hiring", "vacancy", "vacancies",
        "apply", "opening", "career", "position"
    ]

    professional_keywords = [
        "candidate", "candidates", "profile", "profiles",
        "cv", "resume", "talent"
    ]

    company_keywords = [
        "company", "companies", "hotel", "restaurant",
        "brand", "group"
    ]

    supplier_keywords = [
        "supplier", "suppliers", "vendor", "vendors"
    ]

    product_keywords = [
        "product", "products", "equipment", "items"
    ]

    event_keywords = [
        "event", "events", "conference", "expo"
    ]

    article_keywords = [
        "article", "blog", "news"
    ]

    faq_keywords = [
        "how", "why", "what", "guide", "steps", "process"
    ]

    if any(k in q for k in job_keywords):
        return "job"

    if any(k in q for k in professional_keywords):
        return "professional"

    if any(k in q for k in supplier_keywords):
        return "supplier"

    if any(k in q for k in product_keywords):
        return "product"

    if any(k in q for k in event_keywords):
        return "event"

    if any(k in q for k in article_keywords):
        return "article"

    if any(k in q for k in company_keywords):
        return "company"

    if any(k in q for k in faq_keywords):
        return "faq"

    cache_k = f"intent:{query}"
    cached = redis_client.get(cache_k)

    if cached:
        print("🎯 Cached:", cached, flush=True)
        return cached
        

    prompt = f"""
You are an intent classifier for a hospitality platform.

User Query: "{query}"

Available categories:
- job
- company
- professional
- supplier
- product
- event
- article
- award
- faq
- general

RULES:
- Return ONLY ONE category
- No explanation
- No extra text

Examples:
"find waiter job in dubai" → job
"best hotels in dubai" → company
"chef profiles" → professional
"hotel equipment suppliers" → supplier
"what is hospitality" → faq

OUTPUT:
"""

    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "phi3-hoz",
                "prompt": prompt,
                "stream": False
            },
            timeout=4
        )

        intent = res.json().get("response", "").strip().lower()

        print("🎯  Response:", res, flush=True)
        print("🎯 Intent Response:", intent, flush=True)

        valid = {
            "job","company","professional","supplier",
            "product","event","article","award","faq","general"
        }

        if intent not in valid:
            intent = "general"

        redis_client.setex(cache_k, 600, intent)
        return intent

    except Exception as e:
        print("❌ intent error:", e)
        return "general"


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
    


STRICT_DATA_RULES = """
### DATA INTEGRITY RULES (CRITICAL)

- ONLY use data from CONTEXT
- DO NOT generate:
  - salary
  - benefits
  - company claims
  - statistics
- If data not present → DO NOT mention it

- NEVER assume
- NEVER estimate
- NEVER hallucinate

- If unsure → omit

### END RULES
"""

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
- Keep it conversational

{STRICT_DATA_RULES}

User:
{query}
""",
                "stream": False
            },
            timeout=4 
        )

        text = res.json().get("response", "").strip()
        text = text.replace("\n", " ").strip()

        return text

    except Exception as e:
        return ""


# async def stream_answer(ws, query, results):
#     import httpx

#     MAX_TOKENS = 1200
#     count = 0

#     # ✅ Better structured context
#     context = ""
#     for i, r in enumerate(results[:5]):
#         context += f"""
# {i+1}.
# Title: {r.get('title')}
# Category: {r.get('category')}
# Location: {r.get('location')}
# Content: {r.get('content')}
# """

#     model = "llama3-hoz" if results else "phi3-hoz"

#     prompt = f"""
# You are an intelligent AI assistant for Hozpitality.

# IMPORTANT:
# - Continue the answer naturally
# - DO NOT repeat the introduction
# - DO NOT restart
# - Intro is already shown

# User Query:
# {query}

# Context:
# {context}
# """

#     try:
#         async with httpx.AsyncClient(timeout=None) as client:
#             async with client.stream(
#                 "POST",
#                 OLLAMA_URL,
#                 json={
#                     "model": model,
#                     "prompt": prompt,
#                     "stream": True,
#                     "options": {
#                         "num_predict": 900
#                     }
#                 }
#             ) as response:

#                 async for line in response.aiter_lines():

#                     if ws.client_state.name != "CONNECTED":
#                         return

#                     if not line:
#                         continue

#                     data = json.loads(line)

#                     if "response" in data:
#                         chunk = data["response"]

#                         count += len(chunk) / 3
#                         if count > MAX_TOKENS:
#                             break

#                         await safe_send(ws, {
#                             "type": "token",
#                             "data": chunk
#                         })

#                     if data.get("done"):
#                         break

#     except Exception as e:
#         print("❌ STREAM ERROR:", e)

def build_link(slug, category):
    base = "https://www.hozpitality.com"

    if category == "job":
        return f"{base}/jobs/{slug}"
    elif category == "company":
        return f"{base}/company/{slug}"
    elif category == "product":
        return f"{base}/product/{slug}"
    elif category == "supplier":
        return f"{base}/supplier/{slug}"
    elif category == "professional":
        return f"{base}/{slug}"
    elif category == "article":
        return f"{base}/article/{slug}"
    elif category == "event":
        return f"{base}/event/{slug}"
    elif category == "award":
        return f"{base}/award/{slug}"
    else:
        return f"{base}/{slug}"

# async def stream_answer(ws, query, results):
#     import httpx

#     MAX_TOKENS = 1200
#     count = 0

#     print("results and query", query , results)

#     context = ""
#     for i, r in enumerate(results[:5]):
#         context += f"""
# {i+1}.
# Title: {r.get('title')}
# Content: {r.get('content')}
# Location: {r.get('location')}
# URL: https://www.hozpitality.com/{r.get('slug', '')}
# """

#     model = "llama3-hoz" if results else "phi3-hoz"

#     prompt = f"""
# You are an AI search assistant for Hozpitality.

# STRICT RULES:

# 1. You MUST use the provided CONTEXT data
# 2. You MUST include clickable links
# 3. You MUST NOT invent jobs/products/companies
# 4. You MUST NOT give generic career advice if results exist
# 5. You MUST NOT hallucinate

# OUTPUT RULES:

# - Write like a natural chat response (not list)
# - Mention 2–4 relevant results
# - Each result MUST be clickable

# LINK FORMAT (MANDATORY):
# <a href="URL" target="_blank">TITLE</a>

# - NEVER show raw URL
# - NEVER break HTML

# STYLE:
# - conversational
# - helpful
# - clean

# {STRICT_DATA_RULES}

# USER QUERY:
# {query}

# CONTEXT:
# {context}
# """

#     try:
#         async with httpx.AsyncClient(timeout=None) as client:
#             async with client.stream(
#                 "POST",
#                 OLLAMA_URL,
#                 json={
#                     "model": model,
#                     "prompt": prompt,
#                     "stream": True,
#                     "options": {"num_predict": 600}
#                 }
#             ) as response:

#                 async for line in response.aiter_lines():

#                     if ws.client_state.name != "CONNECTED":
#                         return

#                     if not line:
#                         continue

#                     data = json.loads(line)

#                     if "response" in data:
#                         chunk = data["response"]

#                         count += len(chunk) / 3
#                         if count > MAX_TOKENS:
#                             break

#                         await safe_send(ws, {
#                             "type": "token",
#                             "data": chunk
#                         })

#                     if data.get("done"):
#                         break

#     except Exception as e:
#         print("❌ STREAM ERROR:", e)

# async def stream_answer(ws, query, results):
#     import httpx

#     MAX_TOKENS = 1200
#     count = 0

#     print("🚀 ENTERED STREAM ANSWER", flush=True)
#     print("QUERY:", query, flush=True)
#     print("RESULT COUNT:", len(results), flush=True)
#     print("RESULT :", results, flush=True)

#     context_items = []
#     for r in results[:5]:

        
#         url = build_link(r.get("slug"), r.get("category"))

#         print("🔗 GENERATED URL:", url, flush=True)

#         context_items.append({
#             "title": r.get("title"),
#             "location": r.get("location"),
#             "category": r.get("category"),
#             "url": url
#         })
    
#     for r in results[:5]:
#         print("📄 STREAM ITEM:", {
#             "title": r.get("title"),
#             "category": r.get("category"),
#             "slug": r.get("slug")
#         }, flush=True)

#     context_json = json.dumps(context_items, indent=2)

#     model = "llama3-hoz"  

#     prompt = f"""
# You are a AI Search Assitant for Hozpitality and STRICT search result formatter.

# CRITICAL RULES:

# - ONLY use the provided JSON data
# - DO NOT create new jobs, companies, or links
# - DO NOT modify URLs
# - DO NOT hallucinate anything
# - If data is missing → skip it
# - Return clean Markdown

# OUTPUT RULES:

# - Write a natural conversational paragraph
# - Mention 4-5 items from JSON
# - Each item must be clickable Markdown link

# LINK FORMAT:
# [TITLE](URL)

# - DO NOT output bullet points
# - DO NOT output raw JSON
# - DO NOT invent anything

# USER QUERY:
# {query}

# DATA:
# {context_json}
# """

#     try:
#         async with httpx.AsyncClient(timeout=None) as client:
#             async with client.stream(
#                 "POST",
#                 OLLAMA_URL,
#                 json={
#                     "model": model,
#                     "prompt": prompt,
#                     "stream": True,
#                     "options": {"num_predict": 800}
#                 }
#             ) as response:

#                 async for line in response.aiter_lines():
#                     print("📡 STREAM LOOP RUNNING", flush=True)
#                     print("RAW:", line, flush=True)
#                     if ws.client_state.name != "CONNECTED":
#                         return

#                     if not line:
#                         continue

#                     data = json.loads(line)

#                     if "response" in data:
#                         chunk = data["response"]

#                         count += len(chunk) / 3
#                         if count > MAX_TOKENS:
#                             break

#                         await safe_send(ws, {
#                             "type": "token",
#                             "data": chunk
#                         })

#                     if data.get("done"):
#                         break

#     except Exception as e:
#         print("❌ STREAM ERROR:", e)



async def stream_answer(ws, query, results, memory=None):
    import httpx

    print("🚀 HYBRID CHATGPT MODE", flush=True)

    full_response = ""   

    context_items = []
    for r in results[:5]:
        context_items.append({
            "title": r.get("title"),
            "location": r.get("location"),
            "category": r.get("category"),
            "url": build_link(r.get("slug"), r.get("category"))
        })

    intro_text = "Here are some relevant results:\n\n"
    full_response += intro_text

    await safe_send(ws, {
        "type": "token",
        "data": intro_text
    })

    for r in context_items:
        line = f"[{r['title']}]({r['url']})\n\n"
        full_response += line

        await safe_send(ws, {
            "type": "token",
            "data": line
        })

    draft_prompt = f"""
Explain briefly what these results are about in 2-3 lines.

Query: {query}

Titles:
{[r['title'] for r in context_items]}
"""

    try:
        async with httpx.AsyncClient(timeout=None) as client:

            # ⚡ FAST DRAFT
            res = await client.post(
                OLLAMA_URL,
                json={
                    "model": "phi3-hoz",
                    "prompt": draft_prompt,
                    "stream": False,
                    "options": {"num_predict": 80}
                }
            )

            draft = res.json().get("response", "").strip()

            if draft:
                full_response += "\n" + draft + "\n\n"

                await safe_send(ws, {
                    "type": "token",
                    "data": "\n" + draft + "\n\n"
                })

            improve_prompt = f"""
Improve and expand this answer naturally.

User Query:
{query}

Previous Context (if any):
{memory}

Existing Answer:
{draft}

Add more useful detail using these results:
{context_items}

Ask ONE helpful follow-up question from context.
Short. Natural. Relevant.

Rules:
- Keep conversational
- Use markdown links
- Do not repeat
"""

            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={
                    "model": "phi3-hoz",
                    "prompt": improve_prompt,
                    "stream": True,
                    "options": {"num_predict": 300}
                }
            ) as response:

                buffer = ""

                async for line in response.aiter_lines():

                    if ws.client_state.name != "CONNECTED":
                        return full_response

                    if not line:
                        continue

                    data = json.loads(line)

                    if "response" in data:
                        chunk = data["response"]
                        buffer += chunk
                        full_response += chunk   

                        if len(buffer) > 120:
                            await safe_send(ws, {
                                "type": "token",
                                "data": buffer
                            })
                            buffer = ""

                    if data.get("done"):
                        break

                if buffer:
                    await safe_send(ws, {
                        "type": "token",
                        "data": buffer
                    })

    except Exception as e:
        print("❌ STREAM ERROR:", e)

    return full_response   

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

def expand_query_llm(query: str):
    cache_k = f"expand:{query}"
    cached = redis_client.get(cache_k)

    print("🧩 Expanding query cached...", cached, flush=True)

    if cached:
        return json.loads(cached)

    prompt = f"""
You are a strict JSON generator.

Extract structured search data from the query.

USER QUERY:
{query}

REQUIRED OUTPUT FORMAT (STRICT):

{{
  "normalized": "string",
  "roles": ["string"],
  "locations": ["string"]
}}

RULES:
- ONLY return valid JSON
- DO NOT add extra keys
- DO NOT rename keys
- DO NOT create new fields
- "roles" must be a list of job titles (e.g., "cook", "chef")
- "locations" must be cities/countries (e.g., "Dubai")
- If nothing found → return empty list []

INVALID EXAMPLES (DO NOT DO):
❌ "roits"
❌ "job_roles"
❌ "places"

VALID EXAMPLE:
{{
  "normalized": "cook jobs in dubai",
  "roles": ["cook"],
  "locations": ["Dubai"]
}}
"""

    try:
        res = requests.post(
            OLLAMA_URL,
            json={"model": "phi3-hoz", "prompt": prompt, "stream": False},
            timeout=20
        )

        import re
        match = re.search(r'\{.*\}', res.json().get("response", ""), re.DOTALL)

        print("🧩 Expanding query response...", res, flush=True)

        if match:
            data = json.loads(match.group())

            print("🧩 Expanding data response...", data, flush=True)

            data["roles"] = data.get("roles", [])[:5]
            data["locations"] = data.get("locations", [])[:3]


            redis_client.setex(cache_k, 600, json.dumps(data))
            return data

    except Exception as e:
        print("❌ expand_query_llm error:", e)

    return {
        "normalized": query,
        "roles": [],
        "locations": []
    }


def filter_by_intent(results, intent):
    if intent == "general":
        return results

    return [r for r in results if r.get("category") == intent]

# def elastic_search_v2(query_data, intent):
#     should = []

#     should.append({
#         "multi_match": {
#             "query": query_data["normalized"],
#             "fields": ["title^4", "content^2"],
#             "operator": "or",
#             "minimum_should_match": "60%"
#         }
#     })

#     for role in query_data.get("roles", []):
#         should.append({
#             "match": {
#                 "content": {
#                     "query": role,
#                     "boost": 1.5
#                 }
#             }
#         })

#     for loc in query_data.get("locations", []):
#         should.append({
#             "match": {
#                 "location": {
#                     "query": loc,
#                     "boost": 2
#                 }
#             }
#         })

#     filters = []

#     if intent != "general":
#         filters.append({"term": {"category": intent}})

#     print("🔎 ES QUERY:", json.dumps({
#         "normalized": query_data["normalized"],
#         "roles": query_data.get("roles"),
#         "locations": query_data.get("locations"),
#         "intent": intent
#     }, indent=2), flush=True)

#     res = es.search(
#         index="hozpitality",
#         query={
#             "bool": {
#                 "must": should,
#                 "filter": filters
#             }
#         },
#         size=30
#     )

#     print(f"📊 ES RAW HITS: {len(res['hits']['hits'])}", flush=True)

#     results = []
#     for hit in res["hits"]["hits"]:
#         print("➡️ ES HIT:", {
#             "title": hit["_source"].get("title"),
#             "category": hit["_source"].get("category"),
#             "slug": hit["_source"].get("slug") ,
#             "score": hit["_score"]
#         }, flush=True)
#         doc = hit["_source"]
#         doc["slug"] = hit["_source"].get("slug")  # ensure exists
#         doc["bm25_score"] = hit["_score"]

#         print("🧾 FINAL DOC:", doc, flush=True)
#         results.append(doc)

#     return results

def elastic_search_v2(query_data, intent):
    must_clauses = []
    should_clauses = []

    must_clauses.append({
        "multi_match": {
            "query": query_data["normalized"],
            "fields": ["title^4", "content^2"],
            "operator": "or",
            "minimum_should_match": "60%"   
        }
    })

    for role in query_data.get("roles", []):
        should_clauses.append({
            "match": {
                "content": {
                    "query": role,
                    "boost": 1.5
                }
            }
        })

    for loc in query_data.get("locations", []):
        should_clauses.append({
            "match": {
                "location": {
                    "query": loc,
                    "boost": 2
                }
            }
        })

    filters = []
    if intent != "general":
        filters.append({"term": {"category": intent}})

    body = {
        "query": {
            "bool": {
                "must": must_clauses,
                "should": should_clauses,
                "filter": filters
            }
        }
    }

    print("🔎 ES FINAL QUERY:", json.dumps(body, indent=2), flush=True)

    res = es.search(
        index="hozpitality",
        body=body,
        size=30
    )

    hits = res["hits"]["hits"]

    print(f"📊 ES RAW HITS: {len(hits)}", flush=True)

    results = []
    for hit in hits:
        src = hit["_source"]

        print("➡️ ES HIT:", {
            "title": src.get("title"),
            "category": src.get("category"),
            "slug": src.get("slug"),
            "score": hit["_score"]
        }, flush=True)

        doc = src.copy()
        doc["bm25_score"] = hit["_score"]

        results.append(doc)

    return results

def vector_search_v2(query_data):
    full_text = " ".join([
        query_data["normalized"],
        " ".join(query_data.get("roles", [])),
        " ".join(query_data.get("locations", []))
    ])

    print("🧠 VECTOR SEARCH INPUT:", full_text, flush=True)


    query_vec = embedder.encode([full_text], normalize_embeddings=True)

    D, I = index.search(query_vec, 20)

    results = []
    for idx, score in zip(I[0], D[0]):
        print("🧬 VECTOR HIT:", idx, float(score), flush=True)

        if idx == -1:
            continue

        if idx >= len(documents):
            continue

        try:
            doc = documents[idx].copy()
            doc["vector_score"] = float(score)
            results.append(doc)
        except Exception as e:
            print("❌ VECTOR DOC ERROR:", e)

    return results

def hybrid_search_v2(query_data, intent):
    es_results = elastic_search_v2(query_data, intent)
    vec_results = vector_search_v2(query_data)

    combined = {}

    for r in es_results:
        combined[r["title"]] = r

    for r in vec_results:
        if r["title"] in combined:
            combined[r["title"]]["vector_score"] = r.get("vector_score", 0)
        else:
            combined[r["title"]] = r

    final = list(combined.values())

    bm25_scores = [r.get("bm25_score", 0) for r in final]
    vec_scores = [r.get("vector_score", 0) for r in final]

    def norm(arr):
        if not arr:
            return arr
        mn, mx = min(arr), max(arr)
        return [(x - mn) / (mx - mn + 1e-6) for x in arr]

    bm25_norm = norm(bm25_scores)
    vec_norm = norm(vec_scores)

    for i, r in enumerate(final):
        r["final_score"] = (
            0.6 * bm25_norm[i] +
            0.3 * vec_norm[i] +
            (0.1 if r.get("is_paid") else 0)
        )

    seen = set()
    unique = []

    for r in final:
        key = r["title"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return sorted(unique, key=lambda x: x["final_score"], reverse=True)

def hybrid_rank(es_results, vec_results):
    combined = {}
    print("⚖️ HYBRID INPUT:", len(es_results), len(vec_results), flush=True)

    for r in es_results:
        combined[r["title"]] = r

    for r in vec_results:
        if r["title"] in combined:
            combined[r["title"]]["vector_score"] = r.get("vector_score", 0)
        else:
            combined[r["title"]] = r

    final = list(combined.values())

    bm25_scores = [r.get("bm25_score", 0) for r in final]
    vec_scores = [r.get("vector_score", 0) for r in final]

    def norm(arr):
        if not arr:
            return arr
        mn, mx = min(arr), max(arr)
        return [(x - mn) / (mx - mn + 1e-6) for x in arr]

    bm25_norm = norm(bm25_scores)
    vec_norm = norm(vec_scores)

    for i, r in enumerate(final):
        r["final_score"] = (
            0.6 * bm25_norm[i] +
            0.3 * vec_norm[i] +
            (0.1 if r.get("is_paid") else 0)
        )

    for r in final[:5]:
        print("🏆 FINAL SCORE:", {
            "title": r.get("title"),
            "bm25": r.get("bm25_score"),
            "vector": r.get("vector_score"),
            "final": r.get("final_score"),
            "category": r.get("category")
        }, flush=True)

    return sorted(final, key=lambda x: x["final_score"], reverse=True)

@app.websocket("/ws/ai-search")
async def ws_search(ws: WebSocket):
    await ws.accept()
    print("✅ WebSocket connected")

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except:
                print("⚠️ Client disconnected during receive")
                break

            try:
                data = json.loads(raw)
            except:
                await safe_send(ws, {"type": "error", "message": "Invalid JSON"})
                continue

            query = data.get("query", "").strip()
            user_id = data.get("user_id", 0)
            org_id = data.get("org_id", 0)
            conversation_id = data.get("conversation_id")

            if not conversation_id:
                title = query[:50]
                conversation_id = create_conversation(user_id, title)

            await safe_send(ws, {
                "type": "conversation",
                "conversation_id": conversation_id
            })

            last_ai = get_last_ai_response(user_id, org_id)
            last_ctx = get_last_context(user_id, org_id)
            memory = retrieve_memory(user_id, org_id, query)

            follow = detect_followup_llm(query, last_ai)

            
            if follow.get("is_followup") and last_ctx:
                print("🔁 FOLLOW-UP DETECTED", flush=True)
                intent = last_ctx.get("intent")
            else:
                intent = detect_intent_llm(query)

            print(f"🔍 Query: {query}")

            if not query:
                await safe_send(ws, {"type": "error", "message": "Query missing"})
                continue

            print(f"🔍 Query: {query}")
            print("🧠 RAW INPUT:", data, flush=True)
            print("👤 USER ID:", user_id, flush=True)

            intro = ""
            try:
                intro = await asyncio.to_thread(generate_intro, query)
            except:
                pass

            if intro:
                await safe_send(ws, {
                    "type": "token",
                    "data": intro + "\n\n"
                })

            results = []
            total = 0

            try:


                save_message(conversation_id, "user", query)
                store_memory(user_id, org_id, query)
                print("🧭 Detecting intent...", flush=True)
                print("🎯 INTENT:", intent, flush=True)

                print("🧩 Expanding query...", flush=True)
                query_data = expand_query_llm(query)

                if not query_data.get("roles") and not query_data.get("locations"):
                    print("⚠️ LLM FAILED → USING FALLBACK", flush=True)
                    query_data = {
                        "normalized": query,
                        "roles": [],
                        "locations": []
                    }

                print("📦 QUERY DATA:", json.dumps(query_data, indent=2), flush=True)

                es_results = []
                vec_results = []

                try:
                    es_results = await asyncio.to_thread(elastic_search_v2, query_data, intent)
                except Exception as e:
                    print("❌ ES ERROR:", e)

                try:
                    vec_results = await asyncio.to_thread(vector_search_v2, query_data)
                except Exception as e:
                    print("❌ VECTOR ERROR:", e)

                print("📦 BEFORE FILTER:", len(results), flush=True)

                es_results = filter_by_intent(es_results, intent)
                vec_results = filter_by_intent(vec_results, intent)

                print("📦 AFTER FILTER:", len(results), flush=True)

                if vec_results:
                    results = hybrid_rank(es_results, vec_results)
                else:
                    print("⚠️ VECTOR FAILED → USING ES ONLY", flush=True)
                    results = es_results

                if not results:
                    print("❌ NO RESULTS FOUND", flush=True)
                    print("🔍 DEBUG QUERY:", query, flush=True)
                    print("🧠 QUERY DATA:", query_data, flush=True)
                    print("🎯 INTENT:", intent, flush=True)

                if not results:
                    print("❌ NO RESULTS → STOPPING PIPELINE", flush=True)

                    await safe_send(ws, {
                        "type": "token",
                        "data": (
                            "I couldn’t find any results matching your search. "
                            "You may want to refine your query or try different keywords."
                        )
                    })

                    await safe_send(ws, {
                        "type": "done",
                        "total": 0
                    })

                    continue

                results = apply_priority_sorting(results)
                store_last_context(user_id, org_id, intent, results[:3])

                total = len(results)

            except Exception as e:
                print("❌ SEARCH ERROR:", e)

            if ws.client_state.name != "CONNECTED":
                break

            if results:
                print("✅ USING REAL DATA:", len(results), flush=True)
            else:
                if not results:
                    print("❌ NO RESULTS → STOPPING PIPELINE", flush=True)

                    await safe_send(ws, {
                        "type": "token",
                        "data": (
                            "I couldn’t find any results matching your search. "
                            "You may want to refine your query or try different keywords."
                        )
                    })

                    await safe_send(ws, {
                        "type": "done",
                        "total": 0
                    })

                    continue
            
            print("🚨 BEFORE STREAM", len(results), query, flush=True)
            memory_text = ""

            if memory:
                memory_text = "\n".join([f"- {m}" for m in memory[:5]])

            ai_response = await stream_answer(
                ws,
                query,
                results,
                memory_text
            )

            if ai_response:
                save_message(conversation_id, "assistant", ai_response)
                store_last_ai_response(user_id, org_id, ai_response)
                store_memory(user_id, org_id, ai_response)

            await safe_send(ws, {
                "type": "done",
                "total": total
            })

    except Exception as e:
        print("❌ WS ERROR:", str(e))

    finally:
        print("🔌 Connection closed")
        pass

def load_faiss_only():
    global documents, index

    print("⚡ Loading FAISS only (no ES)", flush=True)

    documents = []

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
        category_raw = (r[3] or "").lower()

        if "job" in category_raw:
            category = "job"
        elif "company" in category_raw:
            category = "company"
        elif "candidate" in category_raw or "profile" in category_raw:
            category = "professional"
        elif "supplier" in category_raw:
            category = "supplier"
        elif "product" in category_raw:
            category = "product"
        elif "event" in category_raw:
            category = "event"
        elif "article" in category_raw or "blog" in category_raw:
            category = "article"
        elif "award" in category_raw:
            category = "award"
        elif "faq" in category_raw:
            category = "faq"
        else:
            category = "general"

        text = " ".join([
            r[1] or "",
            r[2] or "",
            category, 
            r[4] or "",
            r[5] or ""
        ])
        texts.append(text)

        documents.append({
            "id": r[0],
            "title": r[1],
            "content": (r[2] or "")[:200],
            "category": category,
            "location": r[4],
            "slug": r[5],
        })

    vectors = embedder.encode(texts, normalize_embeddings=True)
    index.add(np.array(vectors))

    db_pool.putconn(conn)

    print(f"✅ FAISS ready: {index.ntotal}", flush=True)

@app.on_event("startup")
def startup():
    print("🚀 API started (NO ES TOUCH)", flush=True)
    load_faiss_only()