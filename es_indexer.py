import time
from ai_v2 import es, chunked_bulk
from psycopg2.pool import SimpleConnectionPool
import os

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
                health = es.cluster.health(wait_for_status="yellow", request_timeout=5)
                print(f"✅ ES ready: {health['status']}")
                return True
        except Exception as e:
            print("⏳ ES not ready:", e)

        time.sleep(2)

    return False


def create_index():

    if es.indices.exists(index="hozpitality"):

        print("⚠️ Deleting old index...")

        es.indices.delete(index="hozpitality")

    print("📦 Creating optimized AI index...")

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
                    }
                }
            }
        },
        request_timeout=60
    )

    print("✅ AI index created")


def run_reindex():
    print("🔥 FULL REINDEX START")

    if not wait_for_es():
        print("❌ ES not ready")
        return

    create_index()

    conn = db_pool.getconn()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, title, content, category_text, location_text, slug , user_name
        FROM master_search_mastersearchindex
        WHERE is_live = TRUE
    """)

    rows = cur.fetchall()
    actions = []

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

        actions.append({
            "_index": "hozpitality",
            "_id": r[0],
            "_source": {
                "title": r[1],
                "content": r[2],
                "category": category,
                "location": r[4],
                "slug": r[5],
                "user_name": r[6] or ""
            }
        })

    print(f"⚡ Indexing {len(actions)} docs...")

    chunked_bulk(es, actions)

    es.indices.refresh(index="hozpitality")

    db_pool.putconn(conn)

    print("✅ REINDEX DONE")


if __name__ == "__main__":
    run_reindex()