import time
from ai_v2 import es, chunked_bulk
from psycopg2.pool import SimpleConnectionPool
import os
import re

from psycopg2.pool import SimpleConnectionPool
from elasticsearch.helpers import bulk

from ai_v2 import es

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

db_pool = SimpleConnectionPool(1, 5, **DB_CONFIG)

def wait_for_es():

    print("⏳ Waiting for Elasticsearch...")

    for i in range(30):

        try:

            if es.ping():

                health = es.cluster.health(
                    wait_for_status="yellow",
                    request_timeout=5
                )

                print(
                    f"✅ ES READY: {health['status']}",
                    flush=True
                )

                return True

        except Exception as e:

            print(
                "⏳ ES NOT READY:",
                e,
                flush=True
            )

        time.sleep(2)

    return False

def normalize_category(category_raw, content_raw="", slug_raw=""):
    category_raw = (category_raw or "").lower().strip()
    content_raw = (content_raw or "").lower().strip()
    slug_raw = (slug_raw or "").lower().strip()

    if "hozpitality.com/companies" in slug_raw or slug_raw.startswith("/companies"):
        return "company"
        
    if "hozpitality.com/jobs" in slug_raw or slug_raw.startswith("/jobs"):
        return "job"
        
    if "hozpitality.com/marketplace" in slug_raw or slug_raw.startswith("/marketplace"):
        return "product"
        
    if "hozpitality.com/articles" in slug_raw or slug_raw.startswith("/articles"):
        return "article"
        
    if "hozpitality.com/events" in slug_raw or slug_raw.startswith("/events"):
        return "event"
        
    if "hozpitality.com/professional" in slug_raw or slug_raw.startswith("/professional"):
        return "professional"

    return "general"

def build_ai_keywords(
    title,
    content,
    category,
    location
):

    parts = []

    for val in [
        title,
        content,
        category,
        location
    ]:

        if val:
            parts.append(str(val))

    text = " ".join(parts).lower()

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text

def create_index():

    if es.indices.exists(index="hozpitality"):

        print(
            "⚠️ DELETING OLD INDEX",
            flush=True
        )

        es.indices.delete(
            index="hozpitality"
        )

    print(
        "📦 CREATING AI INDEX",
        flush=True
    )

    es.indices.create(

        index="hozpitality",

        body={

            "settings": {

                "analysis": {

                    "analyzer": {

                        "default": {
                            "type": "standard"
                        }
                    }
                }
            },

            "mappings": {

                "properties": {

                    "title": {
                        "type": "text"
                    },

                    "content": {
                        "type": "text"
                    },

                    "category": {
                        "type": "keyword"
                    },

                    "location": {
                        "type": "text"
                    },

                    "slug": {
                        "type": "keyword"
                    },

                    "user_name": {
                        "type": "text"
                    },

                    "ai_keywords": {
                        "type": "text"
                    },

                    "entity_type": {
                        "type": "keyword"
                    },
                    "is_EP": { "type": "boolean" },
                    "is_SP": { "type": "boolean" },
                    "is_GP": { "type": "boolean" },
                    "is_PREMIUM": { "type": "boolean" }
                }
            }
        },

        request_timeout=60
    )

    print(
        "✅ INDEX CREATED",
        flush=True
    )

def chunked_bulk_index(
    actions,
    chunk_size=1000
):

    for i in range(
        0,
        len(actions),
        chunk_size
    ):

        chunk = actions[
            i:i + chunk_size
        ]

        print(
            f"⚡ ES BULK {i} → {i + len(chunk)}",
            flush=True
        )

        bulk(
            es,
            chunk,
            request_timeout=120
        )

        time.sleep(0.2)

def run_reindex():

    print(
        "🔥 FULL AI REINDEX START",
        flush=True
    )

    if not wait_for_es():

        print(
            "❌ ES NOT READY",
            flush=True
        )

        return

    create_index()

    conn = None
    cur = None

    try:

        conn = db_pool.getconn()

        cur = conn.cursor()

        cur.execute("""
            SELECT 
                m.id, m.title, m.content, m.category_text, m.location_text, 
                m.slug, m.user_name, m.ai_keywords,
                COALESCE(pt."is_EP", FALSE) as is_ep,
                COALESCE(pt."is_SP", FALSE) as is_sp,
                COALESCE(pt."is_GP", FALSE) as is_gp,
                COALESCE(pt."is_PREMIUM", FALSE) as is_premium
            FROM master_search_mastersearchindex m
            LEFT JOIN user_accounts u ON m.object_id = u.id
            LEFT JOIN base_packagetype pt ON u.package_id = pt.id
            WHERE m.is_live = TRUE
        """)

        rows = cur.fetchall()

        print(
            f"📊 ROWS FETCHED: {len(rows)}",
            flush=True
        )

        actions = []

        for r in rows:

            category = (r[3] or "general").lower().strip()

            ai_keywords = build_ai_keywords(

                r[1],
                r[2],
                category,
                r[4]

            )

            entity_type = category

            actions.append({

                "_index": "hozpitality",

                "_id": r[0],

                "_source": {

                    "title":
                        r[1] or "",

                    "content":
                        r[2] or "",

                    "category":
                        category,

                    "location":
                        r[4] or "",

                    "slug":
                        r[5] or "",

                    "user_name":
                        r[6] or "",

                    "ai_keywords":
                        ai_keywords,

                    "entity_type":
                        entity_type,
                    "is_EP": bool(r[8]),
                    "is_SP": bool(r[9]),
                    "is_GP": bool(r[10]),
                    "is_PREMIUM": bool(r[11])
                }
            })

        print(
            f"⚡ INDEXING {len(actions)} DOCS",
            flush=True
        )

        chunked_bulk_index(actions)

        es.indices.refresh(
            index="hozpitality"
        )

        print(
            "✅ REINDEX COMPLETE",
            flush=True
        )

    except Exception as e:

        print(
            "❌ REINDEX ERROR:",
            e,
            flush=True
        )

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

if __name__ == "__main__":

    run_reindex()