# ai_v3.py

import os
import json
import asyncio
import traceback

import redis
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ai_v2 import db_pool

from pydantic import BaseModel
from typing import Optional
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch

ES_HOST_SYNC = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
es_sync = Elasticsearch([ES_HOST_SYNC])
BULK_BUFFER = []
BUFFER_LOCK = asyncio.Lock()
BATCH_TIMEOUT = 180.0

from ai_server import (
    search_db,
    apply_priority_sorting,
    create_conversation,
    save_message,
    retrieve_memory,
    store_memory,
    store_last_ai_response,
)

from ai_v2 import (
    elastic_search_v2,
    expand_query_llm
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SyncPayload(BaseModel):
    id: int                                 
    title: Optional[str] = ""               
    content: Optional[str] = ""               
    category: Optional[str] = "general"   
    location: Optional[str] = ""        
    slug: Optional[str] = ""                 
    user_name: Optional[str] = ""            
    ai_keywords: Optional[str] = ""          
    is_live: bool = True                     
    is_deleted: bool = False                 
    entity_type: Optional[str] = ""

async def automatic_bulk_flusher_loop():
    """Background worker thread flusher loop"""
    global BULK_BUFFER
    while True:
        await asyncio.sleep(BATCH_TIMEOUT)
        
        async with BUFFER_LOCK:
            if not BULK_BUFFER:
                continue
                
            try:
                actions = []
                for doc in BULK_BUFFER:
                    doc_dict = doc.dict()
                    print(f"🚀 Processing ID {doc_dict['id']} | Title: {doc_dict['title'][:20]}", flush=True)
                    if doc_dict["is_deleted"] or not doc_dict["is_live"]:
                        actions.append({
                            "_op_type": "delete",        
                            "_index": "hozpitality",     
                            "_id": str(doc_dict["id"])           
                        })
                    else:
                        category_clean = (doc_dict["category"] or "general").lower().strip()
                        entity_clean = (doc_dict["entity_type"] or category_clean).lower().strip()

                        actions.append({
                            "_index": "hozpitality",     
                            "_id": str(doc_dict["id"]),          
                            "_source": {                 
                                "title": doc_dict["title"] or "",
                                "content": doc_dict["content"] or "",
                                "category": category_clean, 
                                "location": doc_dict["location"] or "",
                                "slug": doc_dict["slug"] or "",
                                "user_name": doc_dict["user_name"] or "",
                                "ai_keywords": doc_dict["ai_keywords"] or "",
                                "entity_type": entity_clean
                            }
                        })
                
                if actions:
                    success, errors = bulk(es_sync, actions, request_timeout=60, raise_on_error=False)
                    es_sync.indices.refresh(index="hozpitality")
                    print(f"⚡ Real-time Sync: {success} actions processed into Elasticsearch index [hozpitality].", flush=True)
                
                BULK_BUFFER.clear()
                
            except Exception as e:
                print(f"❌ Failed to execute bulk sync: {e}", flush=True)
                traceback.print_exc()

@app.on_event("startup")
async def start_bulk_worker():
    asyncio.create_task(automatic_bulk_flusher_loop())

@app.post("/api/v1/bulk-sync-es/")
async def receive_single_post_api(data: SyncPayload):
    global BULK_BUFFER
    
    print(f"--- DEBUG DEBUG DEBUG ---", flush=True)
    print(f"📥 Received Job ID: {data.id}", flush=True)
    print(f"📦 Payload Data: {data.dict()}", flush=True) 
    
    async with BUFFER_LOCK:
        is_duplicate = any(item.id == data.id for item in BULK_BUFFER)
        BULK_BUFFER.append(data)
        
    print(f"✅ Buffered. Total Queue Size: {len(BULK_BUFFER)}", flush=True)
    print(f"🔄 Is Duplicate ID? {is_duplicate}", flush=True)
    print(f"--- DEBUG END ---", flush=True)
            
    return {"status": "buffered", "queue_size": len(BULK_BUFFER)}

OLLAMA_URL = "http://ollama:11434/api/generate"

MODEL_CHAT = "llama3-hoz:latest"

redis_client = redis.Redis(
    host="redis",
    port=6379,
    decode_responses=True
)

async def safe_send(ws, data):

    try:

        if ws.client_state.name == "CONNECTED":

            await ws.send_json(data)

    except Exception as e:

        print("❌ SEND ERROR:", e, flush=True)

def detect_intent(query: str):

    q = query.lower().strip()

    greetings = [
        "hi",
        "hello",
        "hey",
        "yo",
        "hola"
    ]

    faq_words = [
        "how",
        "what",
        "why",
        "guide",
        "steps",
        "process"
    ]

    search_words = [
        "job",
        "jobs",
        "hiring",
        "vacancy",
        "hotel",
        "restaurant",
        "chef",
        "waiter",
        "dubai",
        "company",
        "salary",
        "resume",
        "cv"
    ]

    if q in greetings:
        return "greeting"

    if any(w in q for w in faq_words):
        return "faq"

    if any(w in q for w in search_words):
        return "search"

    return "chat"

def generate_followups(category):

    if category == "job":

        return [
            "Show luxury hotel jobs",
            "Show jobs in Dubai",
            "Show visa sponsorship jobs"
        ]

    if category == "company":

        return [
            "Show hotel groups",
            "Show hiring companies"
        ]

    return [
        "Show more results",
        "Refine this search"
    ]

async def stream_chat_response(
    ws,
    query,
    memory_text="",
    intent="chat"
):

    if intent == "faq":

        prompt = f"""
You are Hozpitality AI.

RULES:
- answer step-by-step
- use numbered points
- practical guidance
- concise
- conversational
- hospitality focused
- no hallucinations
- under 150 words

MEMORY:
{memory_text}

USER:
{query}

ASSISTANT:
"""

    else:

        prompt = f"""
You are Hozpitality AI.

RULES:
- conversational
- concise
- helpful
- friendly
- no hallucinations
- under 120 words

MEMORY:
{memory_text}

USER:
{query}

ASSISTANT:
"""

    full = ""

    try:

        async with httpx.AsyncClient(
            timeout=None
        ) as client:

            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={
                    "model": MODEL_CHAT,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 180
                    }
                }
            ) as response:

                async for line in response.aiter_lines():

                    if not line:
                        continue

                    try:

                        data = json.loads(line)

                    except:
                        continue

                    token = data.get("response")

                    if token:

                        full += token

                        await safe_send(ws, {
                            "type": "stream",
                            "token": token
                        })

    except Exception as e:

        print("❌ STREAM ERROR:", e)

        await safe_send(ws, {
            "type": "stream",
            "token": "Something went wrong."
        })

    return full


def get_user_profile(user_id):

    conn = None
    cur = None

    try:

        conn = db_pool.getconn()

        cur = conn.cursor()

        cur.execute("""

            SELECT

                c.name as country,

                d.name as department,

                jr.name as role,

                jl.name as level

            FROM user_accounts ua

            LEFT JOIN countries c
                ON c.id = ua.current_country_id

            LEFT JOIN professionals p
                ON p.useraccount_ptr_id = ua.id

            LEFT JOIN departments d
                ON d.id = p.department_id

            LEFT JOIN job_role jr
                ON jr.id = p.job_role_id

            LEFT JOIN job_levels jl
                ON jl.id = p.job_level_id

            WHERE ua.id = %s

        """, (user_id,))

        row = cur.fetchone()

        if not row:

            return {}

        return {

            "country": (row[0] or "").lower(),

            "department":
                (row[1] or "").lower(),

            "role":
                (row[2] or "").lower(),

            "level":
                (row[3] or "").lower()
        }

    except Exception as e:

        print(
            "❌ PROFILE ERROR:",
            e,
            flush=True
        )

        return {}

    finally:

        try:
            if cur:
                cur.close()
        except:
            pass

        try:
            if conn:
                db_pool.putconn(conn)
        except:
            pass


async def handle_search(
    ws,
    query,
    query_data
):

    try:

        cache_key = (
            f"search:{query.lower()}"
        )

        cached = redis_client.get(
            cache_key
        )

        if cached:

            cached_data = json.loads(
                cached
            )

            await safe_send(ws, {
                "type": "search",
                "data": cached_data
            })

            await safe_send(ws, {
                "type": "done"
            })

            return

        category = query_data.get(
            "category",
            "general"
        )

        print(
            "🧠 QUERY DATA:",
            json.dumps(
                query_data,
                indent=2
            ),
            flush=True
        )

        # =================================
        # PRIMARY SEARCH
        # =================================

        results = await asyncio.to_thread(
            elastic_search_v2,
            query_data,
            category
        )

        print(
            f"📊 PRIMARY RESULTS: {len(results)}",
            flush=True
        )

        # =================================
        # BROADEN SEARCH
        # =================================

        if len(results) < 3:

            print(
                "🌍 BROADENING SEARCH",
                flush=True
            )

            broader_query = (
                query_data.copy()
            )

            broader_query[
                "locations"
            ] = []

            broader_query[
                "profile_country"
            ] = None

            more_results = (
                await asyncio.to_thread(
                    elastic_search_v2,
                    broader_query,
                    category
                )
            )

            existing = set()

            for r in results:

                existing.add(
                    r.get("slug")
                )

            for r in more_results:

                slug = r.get("slug")

                if slug not in existing:

                    results.append(r)

        # =================================
        # SORT
        # =================================

        results = apply_priority_sorting(
            results
        )

        results = results[:8]

        clean_results = []

        for r in results:

            result_category = (
                r.get("category")
                or "general"
            )

            slug = (
                r.get("slug")
                or ""
            )

            if result_category == "job":

                url = (
                    "https://www.hozpitality.com/"
                    f"jobs/details/{slug}"
                )

            elif result_category == "company":

                url = (
                    "https://www.hozpitality.com/"
                    f"company/{slug}"
                )

            elif result_category == "professional":
                url = f"https://www.hozpitality.com/professional/{slug}/about"

            elif result_category == "event":
                url = f"https://www.hozpitality.com/events/details/{slug}"

            elif result_category == "article":
                url = f"https://www.hozpitality.com/articles/details/{slug}"
            
            elif result_category == "awards":
                url = f"https://www.hozpitality.com/awards/"

            else:

                url = (
                    "https://www.hozpitality.com/"
                    f"{slug}"
                )

            clean_results.append({

                "title":
                    r.get("title", ""),

                "url":
                    url,

                "snippet":
                    (
                        r.get("content")
                        or ""
                    )[:180],

                "location":
                    r.get("location", ""),

                "category":
                    result_category
            })

        # =================================
        # PERSONALIZED INTRO
        # =================================

        if clean_results:

            if (
                category == "job"
                and
                query_data.get(
                    "roles"
                )
                and
                not query_data.get(
                    "explicit_role"
                )
            ):

                role = (
                    query_data["roles"][0]
                )

                intro = (
                    f"I found hospitality jobs "
                    f"matching your "
                    f"{role} profile."
                )

            else:

                intro = (
                    f"I found "
                    f"{len(clean_results)} "
                    f"relevant hospitality "
                    f"results."
                )

        else:

            intro = (
                "I couldn't find "
                "matching results."
            )

        # =================================
        # FALLBACK MESSAGE
        # =================================

        if (
            len(clean_results) < 3
            and
            category == "job"
        ):

            intro += (
                " I also included related "
                "hospitality opportunities "
                "outside your profile."
            )

        # =================================
        # FOLLOWUPS
        # =================================

        followups = generate_followups(
            category
        )

        payload = {

            "message": intro,

            "results": clean_results,

            "followups": followups
        }

        # =================================
        # SEND SEARCH RESULTS
        # =================================

        await safe_send(ws, {

            "type": "search",

            "data": payload
        })

        # =================================
        # CACHE
        # =================================

        redis_client.setex(
            cache_key,
            300,
            json.dumps(payload)
        )

        # =================================
        # STREAM AI EXPLANATION
        # =================================

        if clean_results:

            prompt = f"""
You are Hozpitality AI.

USER QUERY:
{query}

RESULTS:
{json.dumps(clean_results[:5])}

TASK:
- explain results naturally
- conversational
- under 80 words
- no hallucinations
- mention hospitality trends only from results
"""

            async with httpx.AsyncClient(
                timeout=None
            ) as client:

                async with client.stream(
                    "POST",
                    OLLAMA_URL,
                    json={
                        "model": MODEL_CHAT,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.4,
                            "num_predict": 120
                        }
                    }
                ) as response:

                    async for line in response.aiter_lines():

                        if not line:
                            continue

                        try:

                            data = json.loads(
                                line
                            )

                        except:
                            continue

                        token = data.get(
                            "response"
                        )

                        if (
                            token
                            and
                            ws.client_state.name
                            == "CONNECTED"
                        ):

                            await safe_send(ws, {
                                "type": "stream",
                                "token": token
                            })

        await safe_send(ws, {
            "type": "done"
        })

    except Exception as e:

        print(
            "❌ SEARCH ERROR:",
            e,
            flush=True
        )

        traceback.print_exc()

        await safe_send(ws, {

            "type": "search",

            "data": {

                "message":
                    "Something went wrong.",

                "results": [],

                "followups": []
            }
        })

        await safe_send(ws, {
            "type": "done"
        })



@app.websocket("/ws/ai-search")
async def websocket_ai_search(
    ws: WebSocket
):

    await ws.accept()

    print("✅ WebSocket connected", flush=True)

    try:

        while True:

            raw = await ws.receive_text()

            data = json.loads(raw)

            query = data.get(
                "query",
                ""
            ).strip()

            user_id = data.get(
                "user_id",
                0
            )

            org_id = data.get(
                "org_id",
                0
            )

            if not query:
                continue

            print("\n====================", flush=True)
            print("📩 QUERY:", query, flush=True)
            print("====================\n", flush=True)

            conversation_id = data.get(
                "conversation_id"
            )

            if not conversation_id:

                title = query[:50]

                conversation_id = create_conversation(
                    user_id,
                    title
                )

                await safe_send(ws, {
                    "type": "conversation",
                    "conversation_id": conversation_id
                })

            save_message(
                conversation_id,
                "user",
                query
            )

            memory_items = retrieve_memory(
                user_id,
                org_id,
                query
            )

            memory_text = "\n".join(
                memory_items[-3:]
            )

            fast_intent = detect_intent(query)

            if fast_intent == "greeting":

                greeting = (
                    "Hi! I can help you find hozpitality jobs, "
                    "companies, candidates, and industry insights."
                )

                await safe_send(ws, {

                    "type": "message",

                    "data": {

                        "message": greeting,

                        "results": [],

                        "followups": [
                            "Find hotel jobs in Dubai",
                            "Show waiter jobs",
                            "Find hospitality companies"
                        ]
                    }
                })

                await safe_send(ws, {
                    "type": "done"
                })

                continue

            try:

                query_data = await asyncio.wait_for(

                    asyncio.to_thread(
                        expand_query_llm,
                        query
                    ),

                    timeout=15

                )

            except Exception as e:

                print(
                    f"❌ expand error: {repr(e)}",
                    flush=True
                )

                query_data = {

                    "normalized": query,

                    "roles": [],

                    "locations": [],

                    "intent": detect_intent(query),

                    "category": (
                        "job"
                        if "job" in query.lower()
                        else "general"
                    )
                }

            profile = await asyncio.to_thread(
                get_user_profile,
                user_id
            )

            print(
                "👤 PROFILE:",
                profile,
                flush=True
            )

            intent = query_data.get(
                "intent",
                "chat"
            )

            category = query_data.get(
                "category",
                "general"
            )

            explicit_roles = (
                query_data.get("roles") or []
            )

            has_explicit_role = (
                len(explicit_roles) > 0
            )

            query_data["explicit_role"] = (
                has_explicit_role
            )

            explicit_locations = (
                query_data.get("locations") or []
            )

            has_explicit_location = (
                len(explicit_locations) > 0
            )


            if (
                intent == "search"
                and
                category == "job"
                and
                not has_explicit_role
            ):

                profile_role = profile.get(
                    "role"
                )

                if profile_role:

                    query_data["roles"].append(
                        profile_role
                    )

                    print(
                        "🎯 PROFILE ROLE:",
                        profile_role,
                        flush=True
                    )

            if (
                intent == "search"
                and
                category == "job"
                and
                not has_explicit_role
            ):

                profile_department = profile.get(
                    "department"
                )

                if profile_department:

                    query_data["roles"].append(
                        profile_department
                    )

                    print(
                        "🏨 PROFILE DEPARTMENT:",
                        profile_department,
                        flush=True
                    )

            if (
                intent == "search"
                and
                category == "job"
                and
                not has_explicit_location
            ):

                profile_country = profile.get(
                    "country"
                )

                if profile_country:

                    query_data[
                        "profile_country"
                    ] = profile_country

                    print(
                        "🌍 PROFILE COUNTRY:",
                        profile_country,
                        flush=True
                    )

            print(
                "🧠 FINAL QUERY DATA:",
                json.dumps(query_data, indent=2),
                flush=True
            )

            print(
                "🎯 FINAL INTENT:",
                intent,
                flush=True
            )

            print(
                "📂 FINAL CATEGORY:",
                category,
                flush=True
            )


            if intent == "search":

                await handle_search(
                    ws,
                    query,
                    query_data
                )

                continue

            answer = await stream_chat_response(
                ws,
                query,
                memory_text,
                intent
            )

            answer = answer.strip()

            if answer:

                save_message(
                    conversation_id,
                    "assistant",
                    answer
                )

                store_memory(
                    user_id,
                    org_id,
                    f"USER: {query}\nAI: {answer}"
                )

                store_last_ai_response(
                    user_id,
                    org_id,
                    answer
                )

            await safe_send(ws, {
                "type": "done"
            })

    except WebSocketDisconnect:

        print(
            "❌ WebSocket disconnected",
            flush=True
        )

    except Exception as e:

        print(
            "❌ WS ERROR:",
            e,
            flush=True
        )

        traceback.print_exc()
