import os
import json
import asyncio
import traceback

import redis
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ai_server import (
    search_db,
    apply_priority_sorting,
    create_conversation,
    save_message,
    retrieve_memory,
    store_memory,
    store_last_ai_response,
)





app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





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





def detect_search_category(query: str):

    q = query.lower()

    if any(x in q for x in [
        "job",
        "jobs",
        "vacancy",
        "hiring",
        "chef",
        "waiter"
    ]):
        return "job"

    if any(x in q for x in [
        "company",
        "hotel",
        "group"
    ]):
        return "company"

    if any(x in q for x in [
        "resume",
        "cv",
        "candidate",
        "profile"
    ]):
        return "professional"

    return "general"





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
    memory_text=""
):

    prompt = f"""
You are Hozpitality AI.

RULES:
- conversational
- concise
- helpful
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





async def handle_search(
    ws,
    query,
):

    category = detect_search_category(query)

    results = search_db(
        query,
        intent_type=category
    )

    results = apply_priority_sorting(results)

    results = results[:6]

    clean_results = []

    for r in results:

        clean_results.append({

            "title": r.get("title"),

            "url": r.get("url"),

            "snippet": (
                r.get("content", "")[:180]
            ),

            "location": r.get("location"),

            "category": r.get("category")
        })

    
    
    

    if clean_results:

        intro = (
            f"I found {len(clean_results)} "
            f"relevant hospitality results."
        )

    else:

        intro = (
            "I couldn't find matching results."
        )

    
    
    

    followups = generate_followups(category)

    
    
    

    await safe_send(ws, {

        "type": "message",

        "data": {

            "message": intro,

            "results": clean_results,

            "followups": followups
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

            
            
            

            intent = detect_intent(query)

            print(
                "🎯 INTENT:",
                intent,
                flush=True
            )

            
            
            

            if intent == "search":

                await handle_search(
                    ws,
                    query
                )

                continue

            
            
            

            answer = await stream_chat_response(
                ws,
                query,
                memory_text
            )

            
            
            

            await safe_send(ws, {

                "type": "message",

                "data": {

                    "message": answer,

                    "results": [],

                    "followups": []
                }
            })

            
            
            

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