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

import re
from collections import Counter

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
        elif "award" in category_raw or "awards" in category_raw:
            category = "awards"
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

            print("📤 SENDING:", data, flush=True)

            await ws.send_json(data)

    except Exception as e:
        print("❌ SEND ERROR:", e, flush=True)



# def detect_intent_llm(query: str):
#     q = query.lower().strip()

#     print("🎯 Detect Intent LLM:", query, flush=True)

#     job_keywords = [
#         "job", "jobs", "hiring", "vacancy", "vacancies",
#         "apply", "opening", "career", "position"
#     ]

#     professional_keywords = [
#         "candidate", "candidates", "profile", "profiles",
#         "cv", "resume", "talent"
#     ]

#     company_keywords = [
#         "company", "companies", "hotel", "restaurant",
#         "brand", "group"
#     ]

#     supplier_keywords = [
#         "supplier", "suppliers", "vendor", "vendors"
#     ]

#     product_keywords = [
#         "product", "products", "equipment", "items"
#     ]

#     event_keywords = [
#         "event", "events", "conference", "expo"
#     ]

#     article_keywords = [
#         "article", "blog", "news"
#     ]

#     faq_keywords = [
#         "how", "why", "what", "guide", "steps", "process"
#     ]

#     if any(k in q for k in job_keywords):
#         return "job"

#     if any(k in q for k in professional_keywords):
#         return "professional"

#     if any(k in q for k in supplier_keywords):
#         return "supplier"

#     if any(k in q for k in product_keywords):
#         return "product"

#     if any(k in q for k in event_keywords):
#         return "event"

#     if any(k in q for k in article_keywords):
#         return "article"

#     if any(k in q for k in company_keywords):
#         return "company"

#     if any(k in q for k in faq_keywords):
#         return "faq"

#     cache_k = f"intent:{query}"
#     cached = redis_client.get(cache_k)

#     if cached:
#         print("🎯 Cached:", cached, flush=True)
#         return cached
        

#     prompt = f"""
# You are an intent classifier for a hospitality platform.

# User Query: "{query}"

# Available categories:
# - job
# - company
# - professional
# - supplier
# - product
# - event
# - article
# - award
# - faq
# - general

# RULES:
# - Return ONLY ONE category
# - No explanation
# - No extra text

# Examples:
# "find waiter job in dubai" → job
# "best hotels in dubai" → company
# "chef profiles" → professional
# "hotel equipment suppliers" → supplier
# "what is hospitality" → faq

# OUTPUT:
# """

#     try:
#         res = requests.post(
#             OLLAMA_URL,
#             json={
#                 "model": "phi3-hoz",
#                 "prompt": prompt,
#                 "stream": False
#             },
#             timeout=4
#         )

#         intent = res.json().get("response", "").strip().lower()

#         print("🎯  Response:", res, flush=True)
#         print("🎯 Intent Response:", intent, flush=True)

#         valid = {
#             "job","company","professional","supplier",
#             "product","event","article","award","faq","general"
#         }

#         if intent not in valid:
#             intent = "general"

#         redis_client.setex(cache_k, 600, intent)
#         return intent

#     except Exception as e:
#         print("❌ intent error:", e)
#         return "general"


def detect_intent_llm(query: str):

    query = query.strip()

    cache_key = f"intent:{query.lower()}"

    cached = redis_client.get(cache_key)

    if cached:
        return cached

    prompt = f"""
You are an intent classifier.

Classify the query into EXACTLY ONE intent.

VALID INTENTS:
- greeting
- search
- faq
- chat

RULES:
- Return ONLY ONE WORD
- No explanation
- No markdown
- No punctuation

EXAMPLES:

hi
-> greeting

hello
-> greeting

find waiter jobs in dubai
-> search

hotels in qatar
-> search

how to apply for hospitality jobs
-> faq

what is hospitality
-> faq

tell me something interesting
-> chat

USER QUERY:
{query}

INTENT:
"""

    try:

        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "phi3-hoz",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 5,
                    "top_p": 0.1
                }
            },
            timeout=2
        )

        print(
            "🧠 OLLAMA RAW:",
            res.json(),
            flush=True
        )

        intent = (
            res.json()
            .get("response", "")
            .strip()
            .lower()
        )

        valid = {
            "greeting",
            "search",
            "faq",
            "chat"
        }

        if intent not in valid:
            intent = "chat"

        redis_client.setex(
            cache_key,
            300,
            intent
        )

        print("🎯 FINAL INTENT:", intent, flush=True)

        return intent

    except Exception as e:

        print("❌ intent error:", e)

        return "chat"


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

# def generate_intro(query: str):
#     try:
#         res = requests.post(
#             OLLAMA_URL,
#             json={
#                 "model": "phi3-hoz",
#                 "prompt": f"""
# You are an AI assistant for Hozpitality.

# TASK:
# - Give a helpful, natural intro in 1–2 lines
# - Do NOT repeat the user query
# - Do NOT add explanation
# - Keep it conversational

# {STRICT_DATA_RULES}

# User:
# {query}
# """,
#                 "stream": False
#             },
#             timeout=4 
#         )

#         text = res.json().get("response", "").strip()
#         text = text.replace("\n", " ").strip()

#         return text

#     except Exception as e:
#         return ""

def generate_intro(
    query: str,
    intent: str,
    results=None
):

    results = results or []

    context_titles = []

    for r in results[:5]:

        title = r.get("title")

        if title:
            context_titles.append(title)

    prompt = f"""
You are Hozpitality AI.

Generate a SHORT conversational intro.

RULES:
- 1 sentence only
- max 20 words
- natural
- conversational
- dynamic wording
- NO markdown
- NO fake information
- NO hallucinations
- DO NOT invent jobs or companies
- ONLY use provided context
- NEVER explain
- NEVER repeat the full user query

INTENT:
{intent}

USER QUERY:
{query}

SEARCH RESULTS:
{context_titles}

GOOD EXAMPLES:
- I found a few relevant hospitality opportunities for you.
- Here are some matching hospitality results.
- I found several relevant listings you may like.
- These results look relevant to your search.

BAD EXAMPLES:
- The FBI recommends...
- Hospitality is theoretically...
- Random paragraphs
- Fake claims

OUTPUT:
"""

    try:

        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "phi3-hoz",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "num_predict": 40
                }
            },
            timeout=2
        )

        text = (
            res.json()
            .get("response", "")
            .strip()
        )

        text = text.replace("\n", " ").strip()

        if len(text) > 120:
            text = text[:120]

        return text

    except Exception as e:

        print("❌ intro error:", e)

        return ""


def build_link(slug, category):
    base = "https://www.hozpitality.com"

    if category == "job":
        return f"{base}/jobs/details/{slug}"
    elif category == "company":
        return f"{base}/company/{slug}"
    elif category == "product":
        return f"{base}/product/details/{slug}"
    elif category == "supplier":
        return f"{base}/supplier/{slug}"
    elif category == "professional":
        return f"{base}/{slug}"
    elif category == "article":
        return f"{base}/article/details/{slug}"
    elif category == "event":
        return f"{base}/event/details/{slug}"
    elif category == "awards":
        return f"{base}/awards/"
    else:
        return f"{base}/{slug}"



# async def stream_answer(ws, query, results, memory=None):
#     import httpx

#     print("🚀 HYBRID CHATGPT MODE", flush=True)

#     full_response = ""   

#     context_items = []
#     for r in results[:5]:
#         context_items.append({
#             "title": r.get("title"),
#             "location": r.get("location"),
#             "category": r.get("category"),
#             "url": build_link(r.get("slug"), r.get("category"))
#         })

#     intro_text = "Here are some relevant results:\n\n"
#     full_response += intro_text

#     await safe_send(ws, {
#         "type": "token",
#         "data": intro_text
#     })

#     for r in context_items:
#         line = f"[{r['title']}]({r['url']})\n\n"
#         full_response += line

#         await safe_send(ws, {
#             "type": "token",
#             "data": line
#         })

#     draft_prompt = f"""
# Explain briefly what these results are about in 2-3 lines.

# Query: {query}

# Titles:
# {[r['title'] for r in context_items]}
# """

#     try:
#         async with httpx.AsyncClient(timeout=None) as client:

#             # ⚡ FAST DRAFT
#             res = await client.post(
#                 OLLAMA_URL,
#                 json={
#                     "model": "phi3-hoz",
#                     "prompt": draft_prompt,
#                     "stream": False,
#                     "options": {"num_predict": 80}
#                 }
#             )

#             draft = res.json().get("response", "").strip()

#             if draft:
#                 full_response += "\n" + draft + "\n\n"

#                 await safe_send(ws, {
#                     "type": "token",
#                     "data": "\n" + draft + "\n\n"
#                 })

#             improve_prompt = f"""
#             You are a STRICT search assistant.
# Improve and expand this answer naturally.

# CRITICAL RULES:
# - ONLY use the provided results
# - DO NOT add external knowledge
# - DO NOT invent jobs, companies, or stories
# - DO NOT generalize beyond results
# - If information not in results → skip it

# User Query:
# {query}

# Previous Context (if any):
# {memory}

# Existing Answer:
# {draft}

# Add more useful detail using these results:
# {context_items}

# TASK:
# - Explain results briefly (4-5 lines)
# - Keep it factual
# - Do NOT hallucinate

# Ask ONE helpful follow-up question from context.
# Short. Natural. Relevant.

# Rules:
# - Keep conversational
# - Use markdown links
# - Do not repeat
# """

#             async with client.stream(
#                 "POST",
#                 OLLAMA_URL,
#                 json={
#                     "model": "phi3-hoz",
#                     "prompt": improve_prompt,
#                     "stream": True,
#                     "options": {"num_predict": 300}
#                 }
#             ) as response:

#                 buffer = ""

#                 async for line in response.aiter_lines():

#                     if ws.client_state.name != "CONNECTED":
#                         return full_response

#                     if not line:
#                         continue

#                     data = json.loads(line)

#                     if "response" in data:
#                         chunk = data["response"]
#                         buffer += chunk
#                         full_response += chunk   

#                         if len(buffer) > 120:
#                             await safe_send(ws, {
#                                 "type": "token",
#                                 "data": buffer
#                             })
#                             buffer = ""

#                     if data.get("done"):
#                         break

#                 if buffer:
#                     await safe_send(ws, {
#                         "type": "token",
#                         "data": buffer
#                     })

#     except Exception as e:
#         print("❌ STREAM ERROR:", e)

#     return full_response   


async def stream_answer(
    ws,
    query,
    intent,
    memory_text,
    results=None
):

    full_response = ""

    if intent == "greeting":

        prompt = f"""
Reply naturally to this greeting.

Memory Text: {memory_text}


RULES:
- short response
- friendly
- conversational
- under 15 words
- no markdown

USER:
{query}
"""

        try:

            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": "llama3-hoz",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "num_predict": 30
                    }
                },
                timeout=2
            )

            text = (
                res.json()
                .get("response", "")
                .strip()
            )

        except:
            text = "Hello 👋"

        await safe_send(ws, {
            "type": "token",
            "data": text
        })

        return text

    if intent in ["faq", "chat"]:

        prompt = f"""
You are Hozpitality AI.

RULES:
- helpful
- concise
- conversational
- markdown allowed
- no fake information
- no hallucinations
- if unsure say you don't know

USER:
{query}
"""

        try:

            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": "llama3-hoz",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 200
                    }
                },
                timeout=5
            )

            text = (
                res.json()
                .get("response", "")
                .strip()
            )

        except:
            text = "I couldn't answer that right now."

        await safe_send(ws, {
            "type": "token",
            "data": text
        })

        return text

    if intent == "search":

        if not results:

            text = (
                "No relevant results found. "
                "Try different keywords."
            )

            await safe_send(ws, {
                "type": "token",
                "data": text
            })

            return text

        intro = f"## Search Results ({len(results)})\n\n"

        full_response += intro

        await safe_send(ws, {
            "type": "token",
            "data": intro
        })

        for r in results[:10]:

            title = r.get("title", "Untitled")

            category = r.get("category", "")

            location = r.get("location", "")

            slug = r.get("slug")

            url = build_link(
                slug,
                category
            )

            line = f"### [{title}]({url})\n"

            if location:
                line += f"📍 {location}\n"

            if category:
                line += f"🏷️ {category}\n"

            line += "\n"

            full_response += line

            await safe_send(ws, {
                "type": "token",
                "data": line
            })

        return full_response

    return ""






@app.post("/track-click")
def click(user_id: int, category: str):
    track_click(user_id, category)
    return {"status": "ok"}


# def expand_query_llm(query: str):

#     cache_key = f"expand:{query.lower()}"

#     cached = redis_client.get(cache_key)

#     if cached:
#         return json.loads(cached)

#     prompt = f"""
# Extract structured search data from this query.

# RETURN STRICT JSON ONLY.

# FORMAT:

# {{
#   "normalized": "string",
#   "roles": [],
#   "locations": [],
#   "intent": "search | faq | chat",
#   "category": "job | company | article | professional | general"
# }}

# RULES:
# - valid JSON only
# - no markdown
# - no explanation
# - normalized = cleaned search query
# - roles = job titles only
# - locations = city/country only
# - intent:
#     - search = searching jobs/companies/articles/profiles
#     - faq = informational/how-to questions
#     - chat = casual conversation
# - category:
#     - job
#     - company
#     - article
#     - professional
#     - general
# - if unsure use:
#     "intent": "chat"
#     "category": "general"

# QUERY:
# {query}
# """

#     try:

#         res = requests.post(
#             OLLAMA_URL,
#             json={
#                 "model": "phi3-hoz",
#                 "prompt": prompt,
#                 "stream": False,
#                 "options": {
#                     "temperature": 0,
#                     "num_predict": 120
#                 }
#             },
#             timeout=15
#         )

#         raw = res.json().get(
#             "response",
#             ""
#         )

#         import re

#         match = re.search(
#             r'\{.*\}',
#             raw,
#             re.DOTALL
#         )

#         if match:

#             data = json.loads(
#                 match.group()
#             )

#             # SAFE FALLBACKS

#             if "normalized" not in data:
#                 data["normalized"] = query

#             if "roles" not in data:
#                 data["roles"] = []

#             if "locations" not in data:
#                 data["locations"] = []

#             if "intent" not in data:
#                 data["intent"] = "chat"

#             if "category" not in data:
#                 data["category"] = "general"

#             redis_client.setex(
#                 cache_key,
#                 300,
#                 json.dumps(data)
#             )

#             print(
#                 "🧠 QUERY DATA:",
#                 json.dumps(data, indent=2),
#                 flush=True
#             )

#             return data

#     except Exception as e:

#         print("❌ expand error:", e)

#     return {
#         "normalized": query,
#         "roles": [],
#         "locations": [],
#         "intent": "chat",
#         "category": "general"
#     }


def expand_query_llm(query: str):

    cache_key = f"expand:{query.lower()}"

    cached = redis_client.get(cache_key)

    if cached:
        return json.loads(cached)

    q = query.lower().strip()

    try:

        es_query = {

            "size": 15,

            "_source": [
                "title",
                "location",
                "category",
                "ai_keywords"
            ],

            "query": {

                "multi_match": {

                    "query": q,

                    "fields": [

                        "title^5",

                        "ai_keywords^4",

                        "location^3",

                        "content"
                    ],

                    "operator": "or",

                    "fuzziness": "AUTO"
                }
            }
        }

        res = es.search(
            index="hozpitality",
            body=es_query
        )

        hits = res["hits"]["hits"]

    except Exception as e:

        print(
            "❌ ES expand search failed:",
            repr(e),
            flush=True
        )

        hits = []


    role_candidates = []

    location_candidates = []

    category_counter = Counter()

    for hit in hits:

        src = hit["_source"]

        title = (
            src.get("title")
            or ""
        ).lower()

        location = (
            src.get("location")
            or ""
        ).lower()

        category = (
            src.get("category")
            or "general"
        )

        category_counter[
            category
        ] += 1

        title_parts = re.split(
            r'[^a-zA-Z]+',
            title
        )

        for p in title_parts:

            p = p.strip()

            if len(p) > 2:

                role_candidates.append(p)

        if location:

            location_candidates.append(
                location
            )

            parts = re.split(
                r'[,/|-]',
                location
            )

            for p in parts:

                p = p.strip()

                if len(p) > 2:

                    location_candidates.append(
                        p
                    )

    query_tokens = set(

        re.findall(
            r"[a-zA-Z]+",
            q
        )
    )

    detected_roles = []

    for role in role_candidates:

        role_tokens = set(

            re.findall(
                r"[a-zA-Z]+",
                role
            )
        )

        if query_tokens.intersection(
            role_tokens
        ):

            detected_roles.append(
                role
            )

    detected_locations = []

    for loc in location_candidates:

        if loc in q:

            detected_locations.append(
                loc
            )

    category = "general"

    tokens = set(
        re.findall(
            r"[a-zA-Z]+",
            q
        )
    )

    q_lower = q.lower()
    
    if "award" in q_lower or "awards" in q_lower:
        category = "awards"
        print(f"DEBUG: Award keyword detected! Category locked to: {category}", flush=True)
        
    elif "supplier" in q_lower or "suppliers" in q_lower:
        category = ["supplier", "company"] 
        print(f"DEBUG: Supplier/Company keyword detected! Category: {category}", flush=True)
        
    elif "event" in q_lower or "events" in q_lower:
        category = "event"
        print(f"DEBUG: Event keyword detected! Category locked to: {category}", flush=True)

    else:

        job_words = {
            "job",
            "jobs",
            "vacancy",
            "vacancies",
            "hiring",
            "career",
            "careers"
        }

        article_words = {
            "article",
            "articles",
            "blog",
            "blogs",
            "news"
        }

        professional_words = {
            "candidate",
            "profile",
            "resume",
            "cv",
            "person",
            "who"
        }

        company_words = {
            "company",
            "companies",
            "hotel",
            "hotels",
            "group",
            "restaurant",
            "restaurants"
        }

        faq_words = {
            "how",
            "what",
            "why",
            "guide",
            "steps",
            "process",
            "tutorial",
            "help",
            "memberships",
            "membership"
        }

        if tokens.intersection(job_words):

            category = "job"

        elif tokens.intersection(article_words):

            category = "article"

        elif tokens.intersection(professional_words):

            category = "professional"

        elif tokens.intersection(company_words):

            category = "company"

        elif tokens.intersection(faq_words):
            category = "faq"
            print(f"DEBUG: FAQ detected, category: {category}", flush=True)

        elif category == "general" and 1 < len(tokens) <= 3:
            
            if not tokens.intersection(company_words):
                category = "professional"

        elif category_counter:

            category = (
                category_counter
                .most_common(1)[0][0]
            )

    faq_words = {
        "how",
        "what",
        "why",
        "guide",
        "steps",
        "process",
        "tutorial",
        "help",
        "memberships",
        "membership"
    }

    greeting_words = {
        "hi",
        "hello",
        "hey",
        "hola"
    }

    search_words = {
        "job",
        "jobs",
        "vacancy",
        "vacancies",
        "hiring",
        "hotel",
        "hotels",
        "restaurant",
        "restaurants",
        "company",
        "companies",
        "chef",
        "waiter",
        "manager"
    }

    intent = "chat"


    if q in greeting_words:

        intent = "greeting"


    elif tokens.intersection(faq_words):

        intent = "faq"


    elif (
        hits
        or
        tokens.intersection(search_words)
    ):

        intent = "search"


    else:

        intent = "chat"


    data = {

        "normalized": query,

        "roles": list(
            set(detected_roles)
        )[:5],

        "locations": list(
            set(detected_locations)
        )[:5],

        "intent": intent,

        "category": category
    }


    if (
        intent == "search"
        and
        not data["roles"]
        and
        not data["locations"]
        and
        len(query_tokens) > 2
    ):

        try:

            prompt = f"""
Extract structured search data.

Return STRICT JSON only.

FORMAT:

{{
  "roles": [],
  "locations": []
}}

QUERY:
{query}
"""
            
            res = requests.post(

                OLLAMA_URL,

                json={

                    "model": "phi3-hoz",

                    "prompt": prompt,

                    "stream": False,

                    "options": {

                        "temperature": 0,

                        "num_predict": 40
                    }
                },

                timeout=2
            )

            raw = (
                res.json()
                .get("response", "")
            )

            match = re.search(
                r'\{.*\}',
                raw,
                re.DOTALL
            )

            if match:

                llm_data = json.loads(
                    match.group()
                )

                data["roles"].extend(
                    llm_data.get(
                        "roles",
                        []
                    )
                )

                data["locations"].extend(
                    llm_data.get(
                        "locations",
                        []
                    )
                )

                data["roles"] = list(
                    set(data["roles"])
                )[:5]

                data["locations"] = list(
                    set(data["locations"])
                )[:5]

        except Exception as e:

            print(
                "⚠️ optional refine failed:",
                repr(e),
                flush=True
            )


    redis_client.setex(

        cache_key,

        300,

        json.dumps(data)
    )

    print(
        "🧠 QUERY DATA:",
        json.dumps(
            data,
            indent=2
        ),
        flush=True
    )

    return data

def elastic_search_v2(query_data, category):

    must_clauses = []
    should_clauses = []
    filters = []

    normalized_query = (
        query_data.get("normalized") or ""
    ).strip()

    profile_country = (
        query_data.get(
            "profile_country"
        )
    )
    
    detected_locations = query_data.get("locations", [])

    if not normalized_query:
        return []

    must_clauses.append({
        "multi_match": {
            "query": normalized_query,
            "fields": [
                "title^5",
                "ai_keywords^6",
                "content^2",
                "location^3",
                "user_name^2"
            ],
            "operator": "or",
            "fuzziness": "AUTO"
        }
    })

    for role in query_data.get("roles", []):

        should_clauses.append({
            "match": {
                "title": {
                    "query": role,
                    "boost": 6
                }
            }
        })

        should_clauses.append({
            "match": {
                "content": {
                    "query": role,
                    "boost": 2
                }
            }
        })

    if detected_locations:
        location_filters = []
        for loc in detected_locations:
            location_filters.append({
                "match": {
                    "location": loc.lower() 
                }
            })
        
        filters.append({
            "bool": {
                "should": location_filters,
                "minimum_should_match": 1
            }
        })
        print(f"DEBUG: Flexible Location Filter Applied: {detected_locations}", flush=True)

    elif query_data.get("locations"): 
        for loc in query_data.get("locations"):
            should_clauses.append({
                "match": {
                    "location": {
                        "query": loc,
                        "boost": 4
                    }
                }
            })
    
    if profile_country:

        should_clauses.append({

            "term": {
                "location": {
                    "value": profile_country,
                    "boost": 5
                }
            }
        })

    if category:

        filters.append({
            "terms": {
                "category": category if isinstance(category, list) else [category]
            }
        })

    # body = {
    #     "size": 30,
    #     "query": {
    #         "bool": {
    #             "must": must_clauses,
    #             "should": should_clauses,
    #             "filter": filters,
    #             "minimum_should_match": 0
    #         }
    #     }
    # }

    body = {
        "size": 30,
        "query": {
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": [
                                {
                                    "bool": {
                                        "must": must_clauses,
                                        "filter": filters
                                    }
                                },
                                {
                                    "match_phrase": {
                                        "title": {
                                            "query": normalized_query,
                                            "boost": 500,
                                            "slop": 0
                                        }
                                    }
                                }
                            ],
                            "minimum_should_match": 1
                        }
                    }
                ],
                "should": should_clauses
            }
        }
    }

    print(
        "🔎 ES FINAL QUERY:",
        json.dumps(body, indent=2),
        flush=True
    )

    try:

        res = es.search(
            index="hozpitality",
            body=body
        )

        hits = res["hits"]["hits"]

        print(
            f"📊 ES RAW HITS: {len(hits)}",
            flush=True
        )

        results = []

        for hit in hits:

            src = hit["_source"]

            print("➡️ ES HIT:", {
                "title": src.get("title"),
                "category": src.get("category"),
                "score": hit["_score"]
            }, flush=True)

            doc = src.copy()

            doc["bm25_score"] = hit["_score"]

            results.append(doc)

        return results

    except Exception as e:

        print("❌ ES ERROR:", e)

        return []


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



@app.websocket("/ws/ai-search")
async def ws_search(ws: WebSocket):

    await ws.accept()

    print("✅ WebSocket connected", flush=True)

    try:

        while True:

            try:
                raw = await ws.receive_text()

            except Exception as e:
                print("⚠️ Client disconnected during receive:", e, flush=True)
                break

            try:
                data = json.loads(raw)

            except Exception as e:

                print("❌ INVALID JSON:", e, flush=True)

                await safe_send(ws, {
                    "type": "error",
                    "message": "Invalid JSON"
                })

                continue

            query = data.get("query", "").strip()

            user_id = data.get("user_id", 0)

            org_id = data.get("org_id", 0)

            conversation_id = data.get("conversation_id")

            print("\n==============================", flush=True)
            print("📩 NEW MESSAGE RECEIVED", flush=True)
            print("==============================", flush=True)

            print("🔍 Query:", query, flush=True)
            print("👤 USER ID:", user_id, flush=True)
            print("🏢 ORG ID:", org_id, flush=True)

            if not query:

                print("❌ EMPTY QUERY", flush=True)

                await safe_send(ws, {
                    "type": "error",
                    "message": "Query missing"
                })

                continue

            if not conversation_id:

                title = query[:50]

                conversation_id = create_conversation(
                    user_id,
                    title
                )

                print(
                    "🆕 CREATED CONVERSATION:",
                    conversation_id,
                    flush=True
                )

            await safe_send(ws, {
                "type": "conversation",
                "conversation_id": conversation_id
            })


            last_ai = get_last_ai_response(
                user_id,
                org_id
            )

            last_ctx = get_last_context(
                user_id,
                org_id
            )

            memory = retrieve_memory(
                user_id,
                org_id,
                query
            )

            print(
                "🧠 MEMORY ITEMS:",
                len(memory) if memory else 0,
                flush=True
            )

            follow = detect_followup_llm(
                query,
                last_ai
            )

            print(
                "🔁 FOLLOWUP DETECTION:",
                follow,
                flush=True
            )

            if follow.get("is_followup") and last_ctx:

                print(
                    "🔁 FOLLOW-UP DETECTED",
                    flush=True
                )

                intent = last_ctx.get("intent")

            else:

                print(
                    "🧭 RUNNING INTENT DETECTION",
                    flush=True
                )

                intent = detect_intent_llm(query)

            print(
                "🎯 FINAL INTENT:",
                intent,
                flush=True
            )

            results = []

            total = 0

            try:

                save_message(
                    conversation_id,
                    "user",
                    query
                )

                store_memory(
                    user_id,
                    org_id,
                    query
                )

                print(
                    "💾 USER MESSAGE SAVED",
                    flush=True
                )

            except Exception as e:

                print(
                    "❌ SAVE MESSAGE ERROR:",
                    e,
                    flush=True
                )

            try:

                if intent == "search":

                    print(
                        "🔎 STARTING SEARCH PIPELINE",
                        flush=True
                    )

                    query_data = expand_query_llm(query)

                    print(
                        "📦 QUERY DATA:",
                        json.dumps(query_data, indent=2),
                        flush=True
                    )

                    if (
                        not query_data.get("roles")
                        and
                        not query_data.get("locations")
                    ):

                        print(
                            "⚠️ USING QUERY FALLBACK",
                            flush=True
                        )

                        query_data = {
                            "normalized": query,
                            "roles": [],
                            "locations": []
                        }

                    try:

                        print(
                            "⚡ RUNNING ELASTIC SEARCH",
                            flush=True
                        )

                        results = await asyncio.to_thread(
                            elastic_search_v2,
                            query_data,
                            query_data.get("category")
                        )

                        print(
                            "📊 RAW SEARCH RESULTS:",
                            results,
                            flush=True
                        )

                        print(
                            "📊 RAW SEARCH RESULTS Counts:",
                            len(results),
                            flush=True
                        )

                    except Exception as e:

                        print(
                            "❌ ELASTIC SEARCH ERROR:",
                            e,
                            flush=True
                        )

                        results = []

                    try:

                        results = apply_priority_sorting(results)

                    except Exception as e:

                        print(
                            "❌ SORT ERROR:",
                            e,
                            flush=True
                        )

                    total = len(results)

                    print(
                        "✅ FINAL RESULTS:",
                        total,
                        flush=True
                    )

                    intro = ""

                    try:

                        intro = await asyncio.to_thread(
                            generate_intro,
                            query,
                            intent,
                            results[:5]
                        )

                        print(
                            "💬 INTRO:",
                            intro,
                            flush=True
                        )

                    except Exception as e:

                        print(
                            "❌ INTRO ERROR:",
                            e,
                            flush=True
                        )

                    if intro:

                        await safe_send(ws, {
                            "type": "token",
                            "data": intro + "\n\n"
                        })

                    if not results:

                        print(
                            "❌ NO SEARCH RESULTS",
                            flush=True
                        )

                        await safe_send(ws, {
                            "type": "token",
                            "data": (
                                "I couldn’t find any results "
                                "matching your search. "
                                "Try different keywords."
                            )
                        })

                        await safe_send(ws, {
                            "type": "done",
                            "total": 0
                        })

                        continue

                    try:

                        store_last_context(
                            user_id,
                            org_id,
                            intent,
                            results[:3]
                        )

                        print(
                            "🧠 CONTEXT STORED",
                            flush=True
                        )

                    except Exception as e:

                        print(
                            "❌ CONTEXT STORE ERROR:",
                            e,
                            flush=True
                        )

                else:

                    print(
                        f"💬 NON-SEARCH INTENT: {intent}",
                        flush=True
                    )

            except Exception as e:

                print(
                    "❌ SEARCH PIPELINE ERROR:",
                    e,
                    flush=True
                )

            if ws.client_state.name != "CONNECTED":

                print(
                    "⚠️ SOCKET DISCONNECTED",
                    flush=True
                )

                break

            if intent == "search":

                if results:

                    print(
                        "✅ USING SEARCH RESULTS:",
                        len(results),
                        flush=True
                    )

                else:

                    print(
                        "⚠️ SEARCH INTENT BUT EMPTY RESULTS",
                        flush=True
                    )

            else:

                print(
                    "💬 SKIPPING RESULT VALIDATION "
                    "FOR NON-SEARCH INTENT",
                    flush=True
                )

            memory_text = ""

            if memory:

                memory_text = "\n".join([
                    f"- {m}"
                    for m in memory[:5]
                ])

            print(
                "🧠 MEMORY TEXT READY",
                flush=True
            )

            print(
                "🚀 CALLING STREAM ANSWER",
                flush=True
            )

            ai_response = await stream_answer(
                ws=ws,
                query=query,
                intent=intent,
                memory_text=memory_text,
                results=results
            )

            print(
                "✅ STREAM COMPLETE",
                flush=True
            )

            if ai_response:

                try:

                    save_message(
                        conversation_id,
                        "assistant",
                        ai_response
                    )

                    store_last_ai_response(
                        user_id,
                        org_id,
                        ai_response
                    )

                    store_memory(
                        user_id,
                        org_id,
                        ai_response
                    )

                    print(
                        "💾 AI RESPONSE SAVED",
                        flush=True
                    )

                except Exception as e:

                    print(
                        "❌ AI SAVE ERROR:",
                        e,
                        flush=True
                    )

            await safe_send(ws, {
                "type": "done",
                "total": total
            })

            print(
                "✅ REQUEST COMPLETE",
                flush=True
            )

    except Exception as e:

        print(
            "❌ WS ERROR:",
            str(e),
            flush=True
        )

        traceback.print_exc()

    finally:

        print(
            "🔌 Connection closed",
            flush=True
        )

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
        elif "awards" in category_raw:
            category = "awards"
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