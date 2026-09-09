import asyncio
_model=None
def _model_load():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from ai_v4.config.settings import settings
        _model=SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model
class VectorSearch:
    def _search(self,query,filters,limit):
        from psycopg2.extras import RealDictCursor
        from ai_v4.config.database import db_pool
        conn=db_pool.getconn();cur=None
        try:
            cur=conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""SELECT 1 FROM information_schema.columns WHERE table_schema='public'
                           AND table_name IN ('master_search_mastersearchindex','master_search_index')
                           AND column_name='embedding' LIMIT 1""")
            if not cur.fetchone():return []
            emb=_model_load().encode(query,normalize_embeddings=True).tolist()
            table="master_search_mastersearchindex"
            cur.execute(f"""SELECT id,title,slug,category_text,location_text,content,user_name,object_id,created_at,
                            1-(embedding <=> %s::vector) AS score
                            FROM {table} WHERE is_live=TRUE AND embedding IS NOT NULL
                            ORDER BY embedding <=> %s::vector LIMIT %s""",(emb,emb,limit))
            return [{"engine":"vector","score":float(r.pop("score",0)),"document":dict(r)} for r in cur.fetchall()]
        except Exception:return []
        finally:
            if cur:cur.close()
            db_pool.putconn(conn)
    async def search(self,query,filters=None,limit=20):
        try:return await asyncio.to_thread(self._search,query,filters or {},limit)
        except Exception:return []
