from psycopg2.extras import RealDictCursor

from ai_v4.config.database import db_pool
from ai_v4.config.logger import logger


class PostgresSearch:

    async def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int = 20
    ):

        conn = None
        cur = None

        try:

            conn = db_pool.getconn()

            cur = conn.cursor(
                cursor_factory=RealDictCursor
            )

            sql = """

                SELECT

                    id,
                    title,
                    slug,
                    category_text,
                    location_text,
                    content,
                    ai_keywords,
                    user_name,
                    content_type_id,
                    object_id

                FROM master_search_mastersearchindex

                WHERE

                    is_live = TRUE

                    AND search_vector @@ plainto_tsquery(%s)

                LIMIT %s

            """

            cur.execute(
                sql,
                (
                    query,
                    limit
                )
            )

            rows = cur.fetchall()

            logger.info(
                f"Postgres Search Results : {len(rows)}"
            )

            results = []

            for row in rows:
                results.append({
                    "engine": "postgres",
                    "score": 0,
                    "document": dict(row)
                })

            return results

        finally:
            if cur:
                cur.close()

            if conn:
                db_pool.putconn(conn)