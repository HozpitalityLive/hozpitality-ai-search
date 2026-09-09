import json
from ai_v4.config.redis import redis_client
from ai_v4.config.settings import settings

class MemoryStore:
    def __init__(self):
        self.local={}
    def load(self,user_id):
        key=f"hoz:ai:v4:{user_id}"
        try:
            raw=redis_client.get(key)
            if raw: return json.loads(raw)
        except Exception: pass
        return self.local.setdefault(str(user_id),{"conversation":[],"last_search":{},"pending_query":None})
    def save(self,user_id,memory):
        memory["conversation"]=memory.get("conversation",[])[-20:]
        key=f"hoz:ai:v4:{user_id}"
        try: redis_client.setex(key,settings.MEMORY_TTL,json.dumps(memory,default=str))
        except Exception: self.local[str(user_id)]=memory
