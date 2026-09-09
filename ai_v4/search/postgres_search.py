import asyncio,re
from psycopg2.extras import RealDictCursor
from ai_v4.config.database import db_pool
from ai_v4.config.settings import settings

def _table(cur):
    cur.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name IN
    ('master_search_mastersearchindex','master_search_index') ORDER BY table_name""")
    r=cur.fetchone(); return r[0] if r else None

class PostgresSearch:
    def _search(self,query,filters,limit):
        conn=db_pool.getconn(); cur=None
        try:
            cur=conn.cursor(cursor_factory=RealDictCursor); table=_table(cur)
            if not table:return []
            where=["is_live=TRUE"]; params=[]
            text=query.strip()
            if text:
                where.append("(search_vector @@ websearch_to_tsquery('simple',%s) OR title ILIKE %s OR content ILIKE %s OR ai_keywords ILIKE %s)")
                params += [text,f"%{text}%",f"%{text}%",f"%{text}%"]
            locations=filters.get("locations") or []
            if locations:
                locparts=[]
                for loc in locations:
                    if loc.lower()=="uae": locparts += ["location_text ILIKE %s","location_text ILIKE %s","location_text ILIKE %s","location_text ILIKE %s"]
                    else: locparts.append("location_text ILIKE %s")
                vals=[]
                for loc in locations:
                    vals += ([f"%{loc}%"]*4 if loc.lower()=="uae" else [f"%{loc}%"])
                where.append("("+" OR ".join(locparts)+")"); params += vals
            for ex in filters.get("exclude_terms",[]):
                where.append("COALESCE(content,'') NOT ILIKE %s AND title NOT ILIKE %s"); params += [f"%{ex}%",f"%{ex}%"]
            d=filters.get("date") or {}
            if d.get("from"):
                where.append("created_at >= %s"); params.append(d["from"])
            if d.get("to"):
                where.append("created_at < (%s::date + INTERVAL '1 day')"); params.append(d["to"])
            cat=filters.get("category")
            if cat:
                where.append("""content_type_id IN (SELECT id FROM django_content_type WHERE model = ANY(%s))""")
                mapping={"job":["job"],"company":["company"],"professional":["professional"],"article":["article"],"event":["event"],"product":["product"],"awards":["awards","award"],"faq":["faq"]}
                params.append(mapping.get(cat,[cat]))
            sql=f"""SELECT id,title,slug,category_text,location_text,content,ai_keywords,user_name,content_type_id,object_id,created_at
                    FROM {table} WHERE {' AND '.join(where)}
                    ORDER BY created_at DESC NULLS LAST LIMIT %s"""
            params.append(min(limit,settings.MAX_SEARCH_RESULTS)); cur.execute(sql,params)
            return [dict(x) for x in cur.fetchall()]
        finally:
            if cur:cur.close()
            db_pool.putconn(conn)
    async def search(self,query,filters=None,limit=20):
        return [{"engine":"postgres","score":0.0,"document":r} for r in await asyncio.to_thread(self._search,query,filters or {},limit)]
    def _analytics(self,query,filters):
        conn=db_pool.getconn();cur=None
        try:
            cur=conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name IN
            ('jobs_job','job_job','jobs','job') ORDER BY table_name""")
            r=cur.fetchone()
            if not r:return []
            table=r[0]; q=query.lower(); where=[];params=[]
            if "available" in q or "currently" in q or "live" in q: where.append("is_live=TRUE")
            if "today" in q:
                where.append("created_at >= CURRENT_DATE AND created_at < CURRENT_DATE + INTERVAL '1 day'")
            loc=(filters.get("locations") or [])
            if loc:
                cols=self._cols(cur,table)
                values=[]
                for x in loc:
                    if x.lower()=="uae":
                        values += ["Dubai","Abu Dhabi","Sharjah","Ajman","Ras Al Khaimah","Fujairah","Umm Al Quwain","United Arab Emirates","UAE"]
                    else:
                        values.append(x)
                if "job_city" in cols:
                    where.append("("+" OR ".join(["job_city ILIKE %s"]*len(values))+")")
                    params += [f"%{x}%" for x in values]
                elif "job_address" in cols:
                    where.append("("+" OR ".join(["job_address ILIKE %s"]*len(values))+")")
                    params += [f"%{x}%" for x in values]
            if "most" in q and "compan" in q:
                cols=self._cols(cur,table)
                if "company_id" in cols:
                    sql=f"SELECT company_id,COUNT(*) AS job_count FROM {table} WHERE {' AND '.join(where) or 'TRUE'} GROUP BY company_id ORDER BY job_count DESC LIMIT 10"
                else: return []
            else:
                sql=f"SELECT COUNT(*) AS count FROM {table} WHERE {' AND '.join(where) or 'TRUE'}"
            cur.execute(sql,params);return [dict(x) for x in cur.fetchall()]
        finally:
            if cur:cur.close()
            db_pool.putconn(conn)
    def _cols(self,cur,table):
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s",(table,))
        return {r[0] for r in cur.fetchall()}
    async def analytics(self,query,filters):
        return await asyncio.to_thread(self._analytics,query,filters)
